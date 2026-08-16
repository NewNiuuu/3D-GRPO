# -*- coding: utf-8 -*-
"""
验证 grpo_test.json 里的点云是否真能从 blob 读到。

复用 spatiallm 自己的 BlobMediaReader（resolve_path + exists），
所以判定逻辑和训练时 100% 一致。

用法：
    cd /home/aiscuser/nyp/3D-RL

    # 本地模式（点云已下载到本地时，默认走这条）
    python grpo/check_data.py
    python grpo/check_data.py --limit 20       # 只查前 20 条（快速）

    # blob 模式（点云仍在 blob 上时，先设好 3 个环境变量）
    export BLOB_CONTAINER_URL="https://yifanyang.blob.core.windows.net/yifanyang"
    export BLOB_SAS_TOKEN="....."          # 别留空！exists 需要 READ+LIST 权限
    export BLOB_BASE_PREFIX="output/liyan" # 有疑问就先试空字符串 ""
    python grpo/check_data.py --mode blob
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="data/grpo_test.json")
    ap.add_argument("--limit", type=int, default=None, help="只查前 N 条唯一路径")
    ap.add_argument(
        "--mode",
        default="auto",
        choices=["auto", "local", "blob"],
        help="auto=本地找得到就走本地，否则走 blob",
    )
    ap.add_argument("--read", action="store_true", help="额外用 open3d 真正读一遍，验证文件没损坏")
    args = ap.parse_args()

    with open(args.json) as f:
        data = json.load(f)

    # 去重 pcd_path（很多条共用同一个点云）
    paths = []
    seen = set()
    for s in data:
        p = s["point_clouds"][0]
        if p not in seen:
            seen.add(p)
            paths.append(p)
    if args.limit:
        paths = paths[: args.limit]

    # ---- 选模式 ----
    mode = args.mode
    if mode == "auto":
        mode = "local" if os.path.exists(paths[0]) else "blob"
    print(f"样本 {len(data)} 条，唯一点云 {len(paths)} 个，模式 = {mode}")
    print("-" * 70)

    missing = []

    if mode == "local":
        total = 0
        for p in paths:
            if not os.path.exists(p):
                missing.append((p, "文件不存在"))
            elif os.path.getsize(p) == 0:
                missing.append((p, "零字节"))
            else:
                total += os.path.getsize(p)
        print(f"示例路径：{paths[0]}")
        print(f"本地合计 {total/2**30:.2f} GiB\n")

        if args.read and not missing:
            from spatiallm.pcd import load_o3d_pcd

            for i, p in enumerate(paths):
                try:
                    pcd = load_o3d_pcd(p)
                    if len(pcd.points) == 0:
                        missing.append((p, "点数为 0"))
                except Exception as e:
                    missing.append((p, f"{type(e).__name__}: {e}"))
                if (i + 1) % 100 == 0:
                    print(f"  已读 {i+1}/{len(paths)} ...")
    else:
        from spatiallm.pcd.blob_utils import BlobMediaReader, has_blob_config

        if not has_blob_config():
            print("！ 没检测到 BLOB_CONTAINER_URL / BLOB_SAS_TOKEN 环境变量，先 export 再跑。")
            sys.exit(1)
        reader = BlobMediaReader()
        cfg = reader.config
        print(f"container = {cfg.container_url}")
        print(f"base_prefix = {cfg.base_prefix!r}\n")
        for i, p in enumerate(paths):
            blob_name = reader.resolve_path(p)  # 训练真正去找的 blob 名
            try:
                ok = reader.exists(p)
            except Exception as e:
                ok = False
                print(f"[{i}] 异常 {type(e).__name__}: {e}")
            if not ok:
                missing.append((p, blob_name))
            if i == 0:
                print("示例解析：")
                print(f"  json 里的 path : {p}")
                print(f"  实际找的 blob  : {blob_name}")
                print(f"  是否存在       : {ok}\n")

    print("=" * 70)
    print(f"总计 {len(paths)} 条，缺失 {len(missing)} 条")
    if missing:
        print("\n缺失清单（前 20 条）：")
        for p, why in missing[:20]:
            print(f"  ✗ {p}")
            print(f"      -> {why}")
        sys.exit(1)
    else:
        print("全部存在 ✅")


if __name__ == "__main__":
    main()
