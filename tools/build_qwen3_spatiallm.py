#!/usr/bin/env python
# Copyright (c) Manycore Tech Inc. and affiliates.
# All rights reserved.
#
# Assemble an initial SpatialLM checkpoint with a Qwen3 language backbone:
#   Qwen3 (pretrained LLM) + Sonata (pretrained point encoder) + Projector (random).
#
# This mirrors spatiallm/tuner/initialize_weight.py, but instead of cloning an
# existing SpatialLM config it *derives* a fresh SpatialLMQwen3Config from the
# plain Qwen3 LLM config and grafts on the Sonata point tower + point tokens.
#
# Usage:
#   python tools/build_qwen3_spatiallm.py \
#       --llm_weight /root/lnj/models/Qwen3-1.7B \
#       --encoder_weight /root/lnj/models/sonata/sonata.pth \
#       --output_dir /root/lnj/models/SpatialLM-Qwen3-1.7B-init

import argparse

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Importing spatiallm registers SpatialLMQwen3Config / SpatialLMQwen3ForCausalLM
# with the transformers Auto* factories.
import spatiallm  # noqa: F401
from spatiallm.model.spatiallm_qwen3 import (
    SpatialLMQwen3Config,
    SpatialLMQwen3ForCausalLM,
)

# Point special tokens (must match the literals used in
# spatiallm/tuner/data/mm_plugin.py: POINT_S_TOKEN / point_token / POINT_E_TOKEN).
POINT_START_TOKEN = "<|point_start|>"
POINT_PAD_TOKEN = "<|point_pad|>"
POINT_END_TOKEN = "<|point_end|>"

# Sonata point encoder configuration (SpatialLM1.1 settings). in_channels=6 means
# the encoder consumes [xyz(3), rgb(3)] per point (grid_coord is passed separately).
SONATA_POINT_CONFIG = {
    "in_channels": 6,
    "order": ["z", "z-trans", "hilbert", "hilbert-trans"],
    "stride": [2, 2, 2, 2],
    "enc_depths": [3, 3, 3, 12, 3],
    "enc_channels": [48, 96, 192, 384, 512],
    "enc_num_head": [3, 6, 12, 24, 32],
    "enc_patch_size": [1024, 1024, 1024, 1024, 1024],
    "mlp_ratio": 4,
    "mask_token": True,
    "enc_mode": True,
    "num_bins": 1280,
}


def main():
    parser = argparse.ArgumentParser("Build a SpatialLM-Qwen3 init checkpoint")
    parser.add_argument(
        "--llm_weight",
        type=str,
        default="/root/lnj/models/Qwen3-1.7B",
        help="Path to the pretrained Qwen3 LLM weights.",
    )
    parser.add_argument(
        "--encoder_weight",
        type=str,
        default="/root/lnj/models/sonata/sonata.pth",
        help="Path to the pretrained Sonata encoder .pth (expects a 'state_dict' key).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Where to save the assembled SpatialLM-Qwen3 checkpoint.",
    )
    parser.add_argument(
        "--projector",
        type=str,
        default="mlp",
        choices=["linear", "mlp"],
        help="Projector type mapping point features into the LLM embedding space.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Derive a SpatialLMQwen3Config from the plain Qwen3 LLM config.
    # ------------------------------------------------------------------
    print(f"Loading Qwen3 LLM config from {args.llm_weight} ...")
    llm_config = AutoConfig.from_pretrained(args.llm_weight)
    config_dict = llm_config.to_dict()
    # Drop fields that should be re-derived by the SpatialLM config class.
    for k in ("architectures", "model_type", "auto_map", "transformers_version"):
        config_dict.pop(k, None)

    config = SpatialLMQwen3Config(**config_dict)
    config.point_backbone = "sonata"
    config.projector = args.projector
    config.point_config = dict(SONATA_POINT_CONFIG)
    # Provisional ids; overwritten once the tokenizer assigns real ids below.
    config.point_start_token_id = -1
    config.point_end_token_id = -1
    config.point_token_id = -1
    config.use_cache = True

    # ------------------------------------------------------------------
    # 2. Instantiate the (randomly initialized) SpatialLM-Qwen3 model.
    # ------------------------------------------------------------------
    print("Instantiating SpatialLMQwen3ForCausalLM (random init) ...")
    model = SpatialLMQwen3ForCausalLM(config)

    # ------------------------------------------------------------------
    # 3. Load the pretrained Qwen3 LLM weights into .model / .lm_head.
    #    A tied-embeddings state_dict materializes both embed_tokens and lm_head.
    # ------------------------------------------------------------------
    print("Loading pretrained Qwen3 LLM weights ...")
    llm_model = AutoModelForCausalLM.from_pretrained(args.llm_weight, torch_dtype="auto")
    llm_state_dict = llm_model.state_dict()
    missing, unexpected = model.load_state_dict(llm_state_dict, strict=False)
    llm_loaded = len(llm_state_dict) - len(
        [k for k in unexpected if k in llm_state_dict]
    )
    print(
        f"  LLM tensors provided: {len(llm_state_dict)} | "
        f"unexpected: {len(unexpected)} | "
        f"model still-missing (point tower + projector expected): {len(missing)}"
    )
    del llm_model, llm_state_dict

    # ------------------------------------------------------------------
    # 4. Load the pretrained Sonata encoder weights into .point_backbone.
    #    The Sonata ckpt was trained with in_channels=9; SpatialLM uses 6, so
    #    the first-layer weight is sliced to the first 6 input columns (same as
    #    the official initialize_weight.py).
    # ------------------------------------------------------------------
    print(f"Loading pretrained Sonata encoder from {args.encoder_weight} ...")
    ckpt = torch.load(args.encoder_weight, weights_only=False)
    encoder_state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    for name, param in model.point_backbone.named_parameters():
        if name in encoder_state_dict and param.shape != encoder_state_dict[name].shape:
            src = encoder_state_dict[name]
            if src.dim() == 2 and src.shape[0] == param.shape[0]:
                encoder_state_dict[name] = src[:, : param.shape[1]]
                print(
                    f"  sliced {name}: {tuple(src.shape)} -> "
                    f"{tuple(encoder_state_dict[name].shape)}"
                )
    enc_missing, enc_unexpected = model.point_backbone.load_state_dict(
        encoder_state_dict, strict=False
    )
    print(
        f"  Sonata loaded | encoder-side missing: {len(enc_missing)} | "
        f"unexpected: {len(enc_unexpected)}"
    )
    del ckpt, encoder_state_dict

    # ------------------------------------------------------------------
    # 5. Add the 3 point special tokens, resize embeddings, set config ids.
    #    The projector (point_proj) stays randomly initialized by design.
    # ------------------------------------------------------------------
    print("Adding point special tokens to the tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(args.llm_weight)
    num_added = tokenizer.add_special_tokens(
        {
            "additional_special_tokens": [
                POINT_START_TOKEN,
                POINT_PAD_TOKEN,
                POINT_END_TOKEN,
            ]
        }
    )
    print(f"  added {num_added} special tokens; new vocab size: {len(tokenizer)}")
    model.resize_token_embeddings(len(tokenizer))

    start_id = tokenizer.convert_tokens_to_ids(POINT_START_TOKEN)
    pad_id = tokenizer.convert_tokens_to_ids(POINT_PAD_TOKEN)
    end_id = tokenizer.convert_tokens_to_ids(POINT_END_TOKEN)
    model.config.point_start_token_id = start_id
    model.config.point_end_token_id = end_id
    model.config.point_token_id = pad_id
    model.config.vocab_size = len(tokenizer)
    model.vocab_size = len(tokenizer)
    # Keep the runtime attributes used by forward() in sync with the config.
    model.point_start_token_id = start_id
    model.point_end_token_id = end_id
    model.point_token_id = pad_id
    print(
        f"  point_start_token_id={start_id}, "
        f"point_end_token_id={end_id}, point_token_id(pad)={pad_id}"
    )

    # ------------------------------------------------------------------
    # 6. Save the assembled checkpoint + tokenizer.
    # ------------------------------------------------------------------
    print(f"Saving assembled checkpoint to {args.output_dir} ...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
