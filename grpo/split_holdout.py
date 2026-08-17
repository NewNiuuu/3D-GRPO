# -*- coding: utf-8 -*-
"""
按点云分组切 train / test 留出集。

为什么必须按点云分组，而不是按条目随机切：
    UrbanVideoBench 4080 道题只对应 1159 个点云，平均 3.5 道题共用一个场景
    （p50=3，max=11）。按条目随机切的话，同一个场景会同时出现在 train 和 test，
    模型在训练时已经见过这个点云和相关问题，测出来的分数是被污染的。
    AirCop 同理（13578 条 / 935 个点云，比例更极端，平均 14.5 道题一个场景）。

⚠ 切完之后必须**用 train 那份重新训练**。对已经用全量数据训完的 ckpt 跑这个
   test，测的还是训练集，不能证明泛化。

用法：
    python grpo/split_holdout.py \
        --json data/UrbanVideoBench/MCQ_EmbodiedCity_AerialVLN.json \
        --test_ratio 0.1 --seed 42

    产出（与原文件同目录）：
        MCQ_EmbodiedCity_AerialVLN_train.json
        MCQ_EmbodiedCity_AerialVLN_test.json

分层：如果条目能识别出来源（UrbanVideo 看点云路径里的 AerialVLN/EmbodiedCity
前缀，AirCop 看 data_source 字段），会**按来源分别抽样**，保证 test 里两个来源的
比例和全量一致。否则 AerialVLN(0.836) 和 EmbodiedCity(0.965) 准确率差 13 个点，
随机切出来的 test 里比例一飘，整体准确率就跟着飘，两次评测没法比。
"""
import argparse
import collections
import json
import os
import random


def pcd_key(item):
    """取条目的点云标识，作为分组键。"""
    p = item.get("point_clouds")
    if isinstance(p, list):
        p = p[0] if p else ""
    return str(p)


def strata_key(item):
    """取分层键：优先用显式的 data_source，否则从点云文件名前缀猜。"""
    if item.get("data_source"):
        return str(item["data_source"])
    base = os.path.basename(pcd_key(item))
    return base.split("_")[0] if base else "unknown"


def main():
    ap = argparse.ArgumentParser("按点云分组切留出测试集")
    ap.add_argument("--json", required=True, help="原始全量 json")
    ap.add_argument("--test_ratio", type=float, default=0.1, help="test 占的点云比例")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default=None, help="默认与原文件同目录")
    args = ap.parse_args()

    data = json.load(open(args.json))
    print(f"[in ] {args.json}: {len(data)} 条")

    # 点云 -> 该点云下的所有条目下标
    groups = collections.OrderedDict()
    for i, it in enumerate(data):
        groups.setdefault(pcd_key(it), []).append(i)
    print(f"       {len(groups)} 个唯一点云，平均 {len(data)/len(groups):.1f} 条/点云")

    # 每个点云归到一个分层（同一点云下的条目来源一致，取第一条即可）
    by_stratum = collections.OrderedDict()
    for k, idxs in groups.items():
        by_stratum.setdefault(strata_key(data[idxs[0]]), []).append(k)

    rng = random.Random(args.seed)
    test_pcds = set()
    for s, pcds in by_stratum.items():
        pcds = sorted(pcds)
        rng.shuffle(pcds)
        n_test = max(1, round(len(pcds) * args.test_ratio))
        test_pcds.update(pcds[:n_test])
        print(f"       分层 {s:<16} {len(pcds):>5} 个点云 -> test {n_test}")

    train = [it for it in data if pcd_key(it) not in test_pcds]
    test = [it for it in data if pcd_key(it) in test_pcds]

    # 断言：两边点云集合无交集（分组切的全部意义就在这一行）
    tr_p = {pcd_key(it) for it in train}
    te_p = {pcd_key(it) for it in test}
    assert not (tr_p & te_p), "点云泄漏，切分逻辑有 bug"

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.json))
    stem = os.path.splitext(os.path.basename(args.json))[0]
    for name, part in (("train", train), ("test", test)):
        path = os.path.join(out_dir, f"{stem}_{name}.json")
        with open(path, "w") as f:
            json.dump(part, f, ensure_ascii=False)
        n_p = len({pcd_key(it) for it in part})
        dist = collections.Counter(strata_key(it) for it in part)
        print(
            f"[out] {path}\n"
            f"       {len(part)} 条 / {n_p} 个点云 / {len(part)/len(data)*100:.1f}% / {dict(dist)}"
        )
    print("\n⚠ 记得改 config 的 train_json 指向 *_train.json 后**重新训练**，")
    print("  否则拿 *_test.json 去评旧 ckpt，测的仍然是训练集。")


if __name__ == "__main__":
    main()
