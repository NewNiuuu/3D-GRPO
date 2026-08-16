# -*- coding: utf-8 -*-
"""
边下边训（streaming）—— 下一批 / 训一批 / 删一批。

背景
----
UrbanVideoBench 的点云单文件中位数 125 MB，1159 个文件共 118 GB；把整个数据集
先拉全再开训，既占磁盘也把「开训」推迟了半小时以上。数据集再大一档（或换机器）
这条路就直接走不通。

做法
----
把「哪些点云文件、以什么顺序被读到」这件事从 DataLoader 手里接管过来：

    1. 按点云文件把样本分组（一个 .ply 通常对应多条 QA）
    2. 文件级 shuffle，再按 window_files 切成一个个窗口
    3. 训窗口 k 之前先把 k 下全（阻塞）；同时后台线程预取 k+1
    4. 窗口 k-2 训完即删本地文件

于是磁盘峰值 ≈ 3 个窗口的大小，而不是整个数据集。window_files=32 时，
UrbanVideo 约 3×32×125MB ≈ 12 GB。

多卡
----
文件按 rank 轮流发牌（files[rank::world_size]），各 rank 拿到的文件互不相交：
  - 带宽不浪费（同一个文件不会被 8 张卡各下一遍）
  - 删除是安全的（不会删掉别的 rank 正在用的文件）
各 rank 的样本数按最小值截齐，否则快的 rank 先跑完、DDP 在 all-reduce 上死等。

安全约定
--------
只删「本进程自己下载下来的」文件。启动时已经存在于本地的文件会被记进
_preexisting 白名单，永不删除 —— 手动下好的数据集（比如已经拉全的 AirCop）
不会被这套逻辑悄悄清掉。

依赖 azcopy（实测 136 MB/s，比 azure SDK 单线程快一个量级）。
"""
import os
import json
import shutil
import subprocess
import tempfile
import threading
from collections import defaultdict

from torch.utils.data import Sampler

DEFAULT_CONFIG = os.path.expanduser("~/.blob_config.json")


def _load_blob_config(path=DEFAULT_CONFIG):
    """读 ~/.blob_config.json，拿 sas_url / sas_token / base_prefix。

    token 只在内存里流转，绝不写进日志或命令回显（下面 _run 也不打印 URL）。
    """
    with open(path, "r") as f:
        c = json.load(f)
    return (
        c["sas_url"].rstrip("/"),
        c["sas_token"].lstrip("?"),
        c.get("base_prefix", "").strip("/"),
    )


class BlobPCDStream:
    """点云的「按需下载 + 用完即删」本地缓存。

    逻辑路径（数据 json 里写的，形如 /Pointcloud-VQA/X/y.ply）
        -> 本地  local_root + 逻辑路径
        -> 远端  base_prefix + 逻辑路径
    两边一一对应，不需要额外的映射表。
    """

    def __init__(
        self,
        local_root,
        config_path=DEFAULT_CONFIG,
        azcopy="azcopy",
        rank=0,
        delete_after=True,
        verbose=True,
    ):
        self.local_root = os.path.abspath(local_root)
        self.sas_url, self.sas_token, self.base_prefix = _load_blob_config(config_path)
        self.azcopy = shutil.which(azcopy) or azcopy
        self.rank = rank
        self.delete_after = delete_after
        self.verbose = verbose

        self._lock = threading.Lock()
        self._threads = {}        # window_key -> Thread
        self._preexisting = set()  # 启动前就有的文件：永不删除
        # azcopy 的 job plan / log 目录按 rank 隔开，8 个进程同时跑不会互相踩
        self._env = dict(os.environ)
        self._env["PATH"] = os.path.dirname(self.azcopy) + ":" + self._env.get("PATH", "")
        jobdir = os.path.join(tempfile.gettempdir(), f"azcopy_r{rank}")
        os.makedirs(jobdir, exist_ok=True)
        self._env["AZCOPY_JOB_PLAN_LOCATION"] = jobdir
        self._env["AZCOPY_LOG_LOCATION"] = jobdir

    # ---------- 路径映射 ----------
    def local_path(self, logical):
        return os.path.join(self.local_root, logical.lstrip("/"))

    def _remote_dir_url(self, logical_dir):
        blob = "/".join(p for p in (self.base_prefix, logical_dir.strip("/")) if p)
        return f"{self.sas_url}/{blob}?{self.sas_token}"

    # ---------- 下载 ----------
    def mark_preexisting(self, logicals):
        """把已经在本地的文件登记为「不是我下的」，后续 release 会跳过它们。"""
        for lg in logicals:
            if os.path.exists(self.local_path(lg)):
                self._preexisting.add(lg)

    def ensure(self, logicals, key=None):
        """阻塞地把这批文件准备好。已存在的跳过，只下缺的。"""
        if key is not None:
            t = self._threads.pop(key, None)
            if t is not None:       # 这批正被预取线程下着，等它下完
                t.join()
        missing = [lg for lg in logicals if not os.path.exists(self.local_path(lg))]
        if not missing:
            return 0
        # 按远端目录分组：一次 azcopy 调用搞定一个目录下的一批文件
        by_dir = defaultdict(list)
        for lg in missing:
            by_dir[os.path.dirname(lg)].append(os.path.basename(lg))
        n = 0
        for logical_dir, names in by_dir.items():
            n += self._download_dir(logical_dir, names)
        return n

    def _download_dir(self, logical_dir, names):
        # 目标传父目录：源 URL 以 <dir> 结尾且带 --recursive 时，azcopy 会在目标下
        # 重建 <dir> 这一层。若目标也写成 <dir>，就会多出 <dir>/<dir> 的嵌套。
        local_dir = os.path.join(self.local_root, logical_dir.lstrip("/"))
        dest = os.path.dirname(local_dir)
        os.makedirs(dest, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("\n".join(names) + "\n")
            list_file = f.name
        try:
            cmd = [
                self.azcopy, "cp", "--recursive",
                "--overwrite", "ifSourceNewer",
                "--list-of-files", list_file,
                "--output-level", "quiet",
                self._remote_dir_url(logical_dir), dest,
            ]
            r = subprocess.run(cmd, env=self._env, capture_output=True, text=True)
            if r.returncode != 0:
                # 不回显 cmd（含 SAS token），只给远端逻辑目录 + stderr
                raise RuntimeError(
                    f"azcopy 下载失败 (rc={r.returncode}) dir={logical_dir} "
                    f"files={len(names)}\n{r.stderr.strip()[:800]}"
                )
        finally:
            os.unlink(list_file)
        if self.verbose:
            print(f"[stream r{self.rank}] ↓ {len(names)} files <- {logical_dir}", flush=True)
        return len(names)

    def ensure_file(self, path):
        """单文件兜底。trainer 真要读点云时若发现文件不在，就地补下。

        正常路径下不会触发（窗口在训之前已经 ensure 过）；它防的是
        DataLoader 预取跑得比预期快、或某次下载被跳过的边角情况。
        """
        logical = path if path.startswith("/") else "/" + path
        if os.path.exists(self.local_path(logical)) or os.path.exists(path):
            return
        with self._lock:
            self.ensure([logical])

    def prefetch(self, logicals, key):
        """后台预取下一个窗口，与当前窗口的训练重叠。"""
        if key in self._threads:
            return
        t = threading.Thread(
            target=self._prefetch_worker, args=(logicals, key), daemon=True
        )
        self._threads[key] = t
        t.start()

    def _prefetch_worker(self, logicals, key):
        try:
            self.ensure(logicals)
        except Exception as e:  # 预取失败不该弄挂训练，留给 ensure_file 兜底
            print(f"[stream r{self.rank}] 预取失败({key}): {e}", flush=True)

    # ---------- 删除 ----------
    def release(self, logicals):
        if not self.delete_after:
            return 0
        n = 0
        for lg in logicals:
            if lg in self._preexisting:   # 不是我下的，不删
                continue
            p = self.local_path(lg)
            try:
                if os.path.isfile(p):
                    os.remove(p)
                    n += 1
            except OSError:
                pass
        if n and self.verbose:
            print(f"[stream r{self.rank}] ✗ 删除 {n} 个已训完的点云", flush=True)
        return n


class StreamingGroupSampler(Sampler):
    """按点云文件分组的窗口式采样器（边下边训的调度中枢）。

    产出的是 dataset 的下标序列，DataLoader 用法不变；它额外负责在合适的
    时机调用 BlobPCDStream 的 ensure / prefetch / release。

    与普通随机采样的区别：随机性从「样本级全局打乱」降级为「文件级打乱 +
    窗口内样本打乱」。这是流式训练的标准取舍（webdataset 也是这么做的）；
    window_files 越大越接近全局打乱，代价是磁盘峰值更高。
    """

    def __init__(
        self,
        dataset,
        stream,
        window_files=32,
        world_size=1,
        rank=0,
        seed=42,
        release_lag=2,
        prefetch=True,
    ):
        self.dataset = dataset
        self.stream = stream
        self.window_files = max(int(window_files), 1)
        self.world_size = max(int(world_size), 1)
        self.rank = rank
        self.seed = seed
        self.release_lag = max(int(release_lag), 1)
        self.do_prefetch = prefetch
        self.epoch = 0

        # 文件 -> 样本下标。直接读 samples，避开 __getitem__ 的字符串处理开销。
        self.file2idx = defaultdict(list)
        for i, s in enumerate(dataset.samples):
            self.file2idx[s["point_clouds"][0]].append(i)
        self.files = sorted(self.file2idx)
        self.stream.mark_preexisting(self.files)

        # 各 rank 步数必须一致，否则 DDP 会在 all-reduce 上挂死 —— 按最少的那个截齐
        counts = []
        for r in range(self.world_size):
            counts.append(
                sum(len(self.file2idx[f]) for f in self.files[r :: self.world_size])
            )
        self.num_samples = min(counts)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.num_samples

    def _plan(self):
        """本 epoch 的窗口划分：文件级打乱 -> 发牌到本 rank -> 切窗口。"""
        import random

        rng = random.Random(self.seed + self.epoch)
        files = list(self.files)
        rng.shuffle(files)
        mine = files[self.rank :: self.world_size]
        return [
            mine[i : i + self.window_files]
            for i in range(0, len(mine), self.window_files)
        ], rng

    def __iter__(self):
        windows, rng = self._plan()
        touched = []
        emitted = 0
        for w, win in enumerate(windows):
            if emitted >= self.num_samples:
                break
            self.stream.ensure(win, key=w)          # 下这一批（阻塞）
            touched.append(win)
            if self.do_prefetch and w + 1 < len(windows):
                self.stream.prefetch(windows[w + 1], key=w + 1)  # 预取下一批

            idxs = [i for f in win for i in self.file2idx[f]]
            rng.shuffle(idxs)                        # 窗口内样本打乱
            for i in idxs:
                if emitted >= self.num_samples:
                    break
                yield i
                emitted += 1

            # 滞后若干个窗口再删：DataLoader 会提前取 index，而点云是在
            # compute_loss 里才真正读的，留出余量避免删掉还没用上的文件。
            if len(touched) > self.release_lag:
                self.stream.release(touched[-(self.release_lag + 1)])

        for win in touched[-self.release_lag :]:     # 收尾：把最后几个窗口也清掉
            self.stream.release(win)


def build_stream_sampler(cfg, dataset, world_size, rank):
    """按 config 造出 (stream, sampler)；未开启流式则返回 (None, None)。"""
    if not cfg.get("stream_from_blob", False):
        return None, None
    stream = BlobPCDStream(
        local_root=cfg.get("pcd_local_root", "/home/aiscuser/nyp/pcdata"),
        config_path=cfg.get("blob_config", DEFAULT_CONFIG),
        azcopy=cfg.get("azcopy_bin", "azcopy"),
        rank=rank,
        delete_after=cfg.get("stream_delete_after", True),
        verbose=(rank == 0) or cfg.get("stream_verbose_all_ranks", False),
    )
    sampler = StreamingGroupSampler(
        dataset,
        stream,
        window_files=cfg.get("stream_window_files", 32),
        world_size=world_size,
        rank=rank,
        seed=cfg.get("seed", 42),
        release_lag=cfg.get("stream_release_lag", 2),
        prefetch=cfg.get("stream_prefetch", True),
    )
    return stream, sampler
