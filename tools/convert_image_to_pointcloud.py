#!/usr/bin/env python3
"""Convert ShareGPT VQA jsonl (image field) -> SpatialLM json (point_clouds field).

For each record:
  - image  ->  point_clouds = ["/Pointcloud-VQA/<DS>/<SPLIT>/<stem>.ply"]
  - "<image>" placeholder in human turns -> "<point_cloud>"
  - drop the "image" key
Output is a pretty-printed JSON array (matching existing datasets like Floodnet).
"""
import os
import json
import argparse


def stem_of(image_value):
    """image may be a str or a 1-element list; return basename without extension."""
    if isinstance(image_value, list):
        if not image_value:
            return None
        image_value = image_value[0]
    base = os.path.basename(str(image_value))
    return os.path.splitext(base)[0]


def convert_record(rec, ds, split):
    stem = stem_of(rec.get("image"))
    if stem is None:
        raise ValueError(f"record {rec.get('id')} has empty image")
    ply_path = f"/Pointcloud-VQA/{ds}/{split}/{stem}.ply"

    conversations = []
    for turn in rec.get("conversations", []):
        val = turn.get("value", "")
        val = val.replace("<image>", "<point_cloud>")
        conversations.append({"from": turn.get("from"), "value": val})

    out = {
        "id": rec.get("id"),
        "data_source": ds,
        "point_clouds": [ply_path],
        "conversations": conversations,
    }
    return out, stem


def load_records(path):
    """Load either a jsonl (one obj/line) or a json array."""
    with open(path) as f:
        head = f.read(64).lstrip()
    if head.startswith("["):
        with open(path) as f:
            return json.load(f)
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="input jsonl/json")
    ap.add_argument("--out", required=True, help="output json array")
    ap.add_argument("--ds", required=True, help="dataset name, e.g. AAVG")
    ap.add_argument("--split", required=True, help="split, e.g. train")
    ap.add_argument("--ply_dir", default="", help="optional local ply dir to verify stems exist")
    args = ap.parse_args()

    records = load_records(args.src)
    src_stems = None
    if args.ply_dir and os.path.isdir(args.ply_dir):
        src_stems = set(
            os.path.splitext(fn)[0]
            for fn in os.listdir(args.ply_dir)
            if fn.endswith(".ply")
        )

    out_records = []
    miss = 0
    for rec in records:
        out, stem = convert_record(rec, args.ds, args.split)
        if src_stems is not None and stem not in src_stems:
            miss += 1
        out_records.append(out)

    with open(args.out, "w") as f:
        json.dump(out_records, f, indent=2, ensure_ascii=False)

    hit = len(out_records) - miss
    verify = f" | ply-verify: hit={hit} miss={miss}" if src_stems is not None else ""
    print(f"[OK] {args.src} -> {args.out} ({len(out_records)} records){verify}")


if __name__ == "__main__":
    main()
