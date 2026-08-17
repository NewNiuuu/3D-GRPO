# -*- coding: utf-8 -*-
"""
Floodnet GRPO 数据集。

每条样本产出一个 dict：
    {
        "idx": <在原始 json 里的下标，难度打标时用来定位题目>,
        "pcd_path": <点云逻辑路径，blob 或本地>,
        "prompt_text": <拼好 point-token 占位符的用户问句>,
        "answer": <GT 答案字符串，供 reward 使用>,
    }

点云不在这里加载成张量（避免 dataset 缓存巨大张量），而是在 trainer 的
collate 阶段用 load_point_cloud_tensor 惰性加载（自动走 blob）。
"""
import os
import json

from torch.utils.data import Dataset

# 与 inference.py / mm_plugin 一致的 point 占位符协议：
# <|point_start|><|point_pad|><|point_end|> 会在模型 forward 里被替换成点云特征。
POINT_S = "<|point_start|>"
POINT_PAD = "<|point_pad|>"
POINT_E = "<|point_end|>"
POINT_CLOUD_PLACEHOLDER = "<point_cloud>"


def _extract_turn(conv, role):
    for c in conv:
        if (c.get("from") or c.get("role")) == role:
            return c.get("value") or c.get("content") or ""
    return ""


class FloodnetGRPODataset(Dataset):
    def __init__(self, json_path, max_samples=None):
        with open(json_path, "r") as f:
            data = json.load(f)
        if max_samples is not None:
            data = data[:max_samples]
        self.samples = data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        conv = s["conversations"]
        human = _extract_turn(conv, "human")
        answer = _extract_turn(conv, "gpt").strip()

        # 数据里 human 文本形如 "<point_cloud>\nWhat is ...?"
        # 把 <point_cloud> 占位符替换成模型认识的 point-token 三连。
        question = human.replace(
            POINT_CLOUD_PLACEHOLDER, f"{POINT_S}{POINT_PAD}{POINT_E}"
        )
        if POINT_S not in question:  # 兜底：若没有占位符，前置一个
            question = f"{POINT_S}{POINT_PAD}{POINT_E}{question}"

        pcd_path = s["point_clouds"][0]

        return {
            "idx": idx,          # 原始 json 里的下标，用于难度打标时定位到具体题目
            "pcd_path": pcd_path,
            "prompt_text": question,
            "answer": answer,
        }
