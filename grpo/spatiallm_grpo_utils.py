# -*- coding: utf-8 -*-
"""
SpatialLM GRPO 公共工具：模型加载 + 点云预处理。

抽出来给 grpo_trainer / 数据集共用。所有点云处理逻辑与 inference.py 保持一致。
"""
import numpy as np
import torch

from spatiallm import Layout
from spatiallm.pcd import load_o3d_pcd, get_points_and_colors, cleanup_pcd, Compose


def load_spatiallm(model_path, dtype=torch.bfloat16):
    """
    在 transformers 5.x 下安全加载 SpatialLM 自定义模型。

    处理两个 5.x meta-device 兼容坑：
      1) sonata __init__ 里 torch.linspace(...).item() 在 meta 张量报错
         -> 加载期给 Tensor.item 打补丁（meta 返回占位 0.0）
      2) z-order 序列化的模块级单例 _key_lut 若在 meta context 创建，前向会
         报 "Cannot copy out of meta tensor" -> 加载后强制在真实 CPU 重建
    """
    from transformers import AutoModelForCausalLM
    import spatiallm  # noqa: F401  注册 spatiallm_qwen3

    _orig_item = torch.Tensor.item

    def _safe_item(self):
        return 0.0 if self.is_meta else _orig_item(self)

    torch.Tensor.item = _safe_item
    try:
        # transformers 4.x 用 torch_dtype，5.x 用 dtype；两版兼容。
        try:
            model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
    finally:
        torch.Tensor.item = _orig_item

    from spatiallm.model.serialization import z_order as _zo

    _zo._key_lut = _zo.KeyLUT()
    return model


def preprocess_point_cloud(points, colors, grid_size, num_bins):
    """点云 -> (N, 9) 特征张量。与 inference.py 完全一致。"""
    transform = Compose(
        [
            dict(type="PositiveShift"),
            dict(type="NormalizeColor"),
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="test",
                keys=("coord", "color"),
                return_grid_coord=True,
                max_grid_coord=num_bins,
            ),
        ]
    )
    pcd = transform({"name": "pcd", "coord": points.copy(), "color": colors.copy()})
    feat = np.concatenate([pcd["grid_coord"], pcd["coord"], pcd["color"]], axis=1)
    return torch.as_tensor(feat)


def load_point_cloud_tensor(pcd_path, num_bins, cleanup=True, max_points=0):
    """
    从路径（本地或 blob 逻辑路径）读点云 -> (N, 9) 张量。

    load_o3d_pcd 会自动判断本地/blob（见 pcd_loader）。返回单个点云张量，
    不带 batch 维；batch 维由 trainer 拼接。

    max_points>0 时对超限的点云做**均匀随机下采样**，用来给显存封顶。
    背景：体素下采样后的点数由场景尺度决定，各数据集差异极大——
      AirCop      p50 ≈ 2.0k 点
      UrbanVideo  p50 ≈ 26k、p90 ≈ 59k、尾部见过 22 万点
    Sonata 是 fp32 且注意力随点数增长，22 万点那种会把 40G A100 打爆。
    抽样在 CPU 上做，且发生在缓存之前，所以每个文件只抽一次（同一 epoch 内
    同一份点云对所有 rollout 保持一致，不会给 GRPO 的组内比较引入噪声）。
    """
    pcd = load_o3d_pcd(pcd_path)
    grid_size = Layout.get_grid_size(num_bins)
    if cleanup:
        pcd = cleanup_pcd(pcd, voxel_size=grid_size)
    points, colors = get_points_and_colors(pcd)
    feat = preprocess_point_cloud(points, colors, grid_size, num_bins)
    if max_points and feat.shape[0] > max_points:
        idx = torch.randperm(feat.shape[0])[:max_points]
        feat = feat[idx.sort().values]  # 保序，别打乱 z-order 序列化的局部性
    return feat
