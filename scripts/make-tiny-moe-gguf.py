#!/usr/bin/env python3
"""Generate a tiny random qwen2moe GGUF for expert-streaming tests.

The model is deliberately small but uses Q4_K-compatible dimensions (multiples
of 256) so the F16 output can be requantized with llama-quantize. Per-expert
.scale tensors are included with distinct values per expert, so any confusion
between real expert ids and pool slot ids changes the logits.

Usage: make-tiny-moe-gguf.py [-o tiny-moe.gguf] [--seed 42]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gguf-py"))
import gguf

N_VOCAB_BYTES = 256
N_EMBD = 256
N_LAYER = 4
N_HEAD = 4
N_HEAD_KV = 2
N_FF_EXP = 256
N_FF_SHEXP = 256
N_EXPERT = 8
N_EXPERT_USED = 2
N_CTX = 512


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", default="tiny-moe.gguf")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    def w(*shape):
        return (rng.standard_normal(shape) * 0.02).astype(np.float16)

    tokens = ["<unk>", "<s>", "</s>"] + [f"<0x{i:02X}>" for i in range(N_VOCAB_BYTES)]
    types = [gguf.TokenType.UNKNOWN, gguf.TokenType.CONTROL, gguf.TokenType.CONTROL]
    types += [gguf.TokenType.BYTE] * N_VOCAB_BYTES
    n_vocab = len(tokens)

    writer = gguf.GGUFWriter(args.output, "qwen2moe")
    writer.add_name("tiny-moe-test")
    writer.add_context_length(N_CTX)
    writer.add_embedding_length(N_EMBD)
    writer.add_block_count(N_LAYER)
    writer.add_feed_forward_length(N_FF_SHEXP)
    writer.add_head_count(N_HEAD)
    writer.add_head_count_kv(N_HEAD_KV)
    writer.add_layer_norm_rms_eps(1e-6)
    writer.add_rope_freq_base(10000.0)
    writer.add_expert_count(N_EXPERT)
    writer.add_expert_used_count(N_EXPERT_USED)
    writer.add_expert_feed_forward_length(N_FF_EXP)
    writer.add_expert_shared_feed_forward_length(N_FF_SHEXP)

    writer.add_tokenizer_model("llama")
    writer.add_token_list(tokens)
    writer.add_token_scores([0.0] * n_vocab)
    writer.add_token_types(types)
    writer.add_unk_token_id(0)
    writer.add_bos_token_id(1)
    writer.add_eos_token_id(2)

    writer.add_tensor("token_embd.weight", w(n_vocab, N_EMBD))
    writer.add_tensor("output_norm.weight", np.ones(N_EMBD, dtype=np.float32))
    writer.add_tensor("output.weight", w(n_vocab, N_EMBD))

    n_embd_gqa = N_EMBD // N_HEAD * N_HEAD_KV
    for i in range(N_LAYER):
        p = f"blk.{i}."
        writer.add_tensor(p + "attn_norm.weight", np.ones(N_EMBD, dtype=np.float32))
        writer.add_tensor(p + "attn_q.weight", w(N_EMBD, N_EMBD))
        writer.add_tensor(p + "attn_k.weight", w(n_embd_gqa, N_EMBD))
        writer.add_tensor(p + "attn_v.weight", w(n_embd_gqa, N_EMBD))
        writer.add_tensor(p + "attn_output.weight", w(N_EMBD, N_EMBD))
        writer.add_tensor(p + "ffn_norm.weight", np.ones(N_EMBD, dtype=np.float32))

        writer.add_tensor(p + "ffn_gate_inp.weight", w(N_EXPERT, N_EMBD))
        writer.add_tensor(p + "ffn_gate_exps.weight", w(N_EXPERT, N_FF_EXP, N_EMBD))
        writer.add_tensor(p + "ffn_up_exps.weight", w(N_EXPERT, N_FF_EXP, N_EMBD))
        writer.add_tensor(p + "ffn_down_exps.weight", w(N_EXPERT, N_EMBD, N_FF_EXP))

        for kind in ("gate", "up", "down"):
            scales = (0.5 + 0.125 * np.arange(N_EXPERT) + 0.01 * i).astype(np.float32)
            writer.add_tensor(p + f"ffn_{kind}_exps.scale", scales)

        writer.add_tensor(p + "ffn_gate_inp_shexp.weight", w(N_EMBD))
        writer.add_tensor(p + "ffn_gate_shexp.weight", w(N_FF_SHEXP, N_EMBD))
        writer.add_tensor(p + "ffn_up_shexp.weight", w(N_FF_SHEXP, N_EMBD))
        writer.add_tensor(p + "ffn_down_shexp.weight", w(N_EMBD, N_FF_SHEXP))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(
        f"wrote {args.output} (n_vocab={n_vocab}, {N_LAYER} layers, {N_EXPERT} experts)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
