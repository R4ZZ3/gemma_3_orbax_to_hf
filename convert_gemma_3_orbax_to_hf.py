#!/usr/bin/env python
"""
Copyright 2024 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

This script converts a maxtext Gemma3 checkpoint (in Flax/Orbax format)
into a Hugging Face–compatible PyTorch checkpoint.
It assumes that the maxtext checkpoint was produced by the conversion
script from Keras → maxtext and that the maxtext parameter tree has the following structure:

  {
    "params": {
      "token_embedder": {
         "embedding": <np.ndarray>  # shape (vocab_size, embed_dim), already scaled by sqrt(embed_dim)
      },
      "decoder": {
         "decoder_norm": { "scale": <np.ndarray> },  # final norm; stored as (original_scale + 1)
         "layers": {
             "self_attention_local": {
                "query": { "kernel": <np.ndarray> },   # shape (num_local, num_heads, hidden_size, head_dim)
                "key":   { "kernel": <np.ndarray> },   # shape (num_local, num_kv_heads, hidden_size, head_dim)
                "value": { "kernel": <np.ndarray> },   # shape (num_local, num_kv_heads, hidden_size, head_dim)
                "out":   { "kernel": <np.ndarray> }    # shape (num_local, num_heads, head_dim, hidden_size)
             },
             "self_attention_global": {
                "query": { "kernel": <np.ndarray> },   # shape (num_global, num_heads, hidden_size, head_dim)
                "key":   { "kernel": <np.ndarray> },   # shape (num_global, num_kv_heads, hidden_size, head_dim)
                "value": { "kernel": <np.ndarray> },   # shape (num_global, num_kv_heads, hidden_size, head_dim)
                "out":   { "kernel": <np.ndarray> }    # shape (num_global, num_heads, head_dim, hidden_size)
             },
             "mlp_local": {
                "wi_0": { "kernel": <np.ndarray> },    # gate_proj weight, shape (num_local, intermediate_dim, hidden_size)
                "wi_1": { "kernel": <np.ndarray> },    # up_proj weight, shape (num_local, intermediate_dim, hidden_size)
                "wo":   { "kernel": <np.ndarray> }     # down_proj weight, shape (num_local, hidden_size, intermediate_dim)
             },
             "mlp_global": {
                "wi_0": { "kernel": <np.ndarray> },
                "wi_1": { "kernel": <np.ndarray> },
                "wo":   { "kernel": <np.ndarray> }
             },
             "pre_self_attention_norm_local": { "scale": <np.ndarray> },   # shape (num_local, hidden_size) (stored as original+1)
             "post_self_attention_norm_local": { "scale": <np.ndarray> },
             "pre_ffw_norm_local": { "scale": <np.ndarray> },
             "post_ffw_norm_local": { "scale": <np.ndarray> },
             "pre_self_attention_norm_global": { "scale": <np.ndarray> },  # same for global parts
             "post_self_attention_norm_global": { "scale": <np.ndarray> },
             "pre_ffw_norm_global": { "scale": <np.ndarray> },
             "post_ffw_norm_global": { "scale": <np.ndarray> },
         }
      }
    }
  }

The resulting Hugging Face checkpoint will be a flat state dict whose keys follow
the HF Gemma3 model naming (for example, "model.embed_tokens.weight", "model.layers.0.self_attn.q_proj.weight", etc.).

##### Created by R4ZZ3 (Rasmus Toivanen) and Gemini 2.5 Pro #####
##### Inspiration from https://github.com/AI-Hypercomputer/maxtext/blob/f6ebc1662cb944bd7748fb350bba164b13479b68/MaxText/gemma2_orbax_to_hf.py #####

Usage
JAX_PLATFORMS=cpu python MaxText/convert_gemma_3_orbax_to_hf.py \
    MaxText/configs/base.yml \
    run_name=convert_gemma3_step3500 \
    base_output_directory=/tmp/convert_gemma3 \
    load_parameters_path=gs/checkpoints/3500/ \
    hf_model_path=gemma3_4b_pt_continue_fi_3500 \
    hf_model_name=google/gemma-3-4b-pt
"""



import math
import numpy as np
import torch
import jax
from absl import app
import pyconfig
import maxengine
from transformers import AutoConfig, Gemma2ForCausalLM, Gemma3ForCausalLM
from generate_param_only_checkpoint import _read_train_checkpoint
from tqdm import tqdm
from typing import Sequence
jax.config.update("jax_platform_name", "cpu")

GEMMA_VOCAB_SIZE = 256000


def load_hf_model(model_size):
  """
  Load the model that we are interested in from HuggingFace

  """
  if model_size == "4b":
    config = AutoConfig.from_pretrained("google/gemma-3-4b-pt")
    model = Gemma3ForCausalLM(config.text_config)
  else:
    raise NotImplementedError

  return model


def load_maxtext_params(config):
    """
    Loads model parameters from the given checkpoint_path using load_model_state.
    This function temporarily sets config.load_parameters_path and config.load_full_state_path,
    calls load_model_state, and then returns the inner parameters (discarding optimizer state).
    """
    engine = maxengine.MaxEngine(config)
    rng = jax.random.PRNGKey(1234)
    rng, rng_load_params = jax.random.split(rng)
    params = engine.load_params(rng_load_params)

    return params['params']


def unpermute_from_match_maxtext_rope(arr):
  """
  Function to get the RoPE values in correct ordering
  """
  split_size = arr.shape[-1] // 2  # Assuming half for evens, half for odds
  evens = arr[..., :split_size]
  odds = arr[..., split_size:]
  return jax.numpy.stack([evens, odds], axis=len(arr.shape)).reshape(arr.shape)

def reverse_scale(arr,scale):
  """
  MaxText has the scaling factor included into the weights,
  we reverse it when writing out the HuggingFace checkpoint
  """
  return arr * np.sqrt(scale)


def convert_maxtext_to_hf(config, model_size):
    params = load_maxtext_params(config)
    hf_state = {}

    # Load HF config to get target layer count and vocab size
    hf_model_identifier = "google/gemma-3-4b-pt" 
    print(f"Loading HF config for: {hf_model_identifier}")
    hf_config = AutoConfig.from_pretrained(hf_model_identifier) 
    if hasattr(hf_config, 'text_config'):
        hf_config = hf_config.text_config
    target_num_layers = hf_config.num_hidden_layers
    target_vocab_size = hf_config.vocab_size
    target_hidden_size = hf_config.hidden_size
    target_num_heads = hf_config.num_attention_heads
    target_num_kv_heads = hf_config.num_key_value_heads
    target_head_dim = hf_config.head_dim

    print(f"Target HF model layers: {target_num_layers}, vocab size: {target_vocab_size}")

    # Verify expected keys for Gemma3 structure
    if "decoder" not in params or "token_embedder" not in params:
        raise KeyError("Checkpoint structure mismatch. Expected 'decoder' and 'token_embedder' keys based on loaded structure.")

    # --- Token embedding ---
    print("Converting token embeddings...")
    embed = np.array(params["token_embedder"]["embedding"]) # Shape: (maxtext_vocab, embed_dim)
    maxtext_vocab_size, embed_dim = embed.shape
    if embed_dim != target_hidden_size:
         print(f"Warning: Embedding dimension mismatch! MaxText={embed_dim}, HF Target={target_hidden_size}")

    # Scale and slice vocab
    hf_embed = embed / math.sqrt(embed_dim)
    hf_state["model.embed_tokens.weight"] = torch.tensor(
        np.array(hf_embed[:target_vocab_size], dtype=np.float32), dtype=torch.bfloat16 # Slice to target vocab
    )

    # --- Final (decoder) norm ---
    print("Converting final norm...")
    final_norm = np.array(params["decoder"]["decoder_norm"]["scale"])
    hf_state["model.norm.weight"] = torch.tensor(
        np.array(final_norm - 1, dtype=np.float32), dtype=torch.bfloat16 # Subtract 1
    )

    # --- Layers conversion ---
    print("Converting layers...")
    maxtext_layers = params["decoder"]["layers"]

    # Determine number of layers from the norm scale shapes
    try:
        # MaxText shapes have layer index as the second dimension (index 1)
        num_layers = maxtext_layers["pre_self_attention_norm"]["scale"].shape[1] # Use known norm key
        print(f"Found {num_layers} layers in MaxText checkpoint from norm shapes.")
    except KeyError as e:
         raise KeyError(f"Could not determine layer count from MaxText norm structure (e.g., pre_self_attention_norm): {e}")
    except IndexError as e:
         raise ValueError(f"Could not determine layer count from shape of norm scale. Checkpoint structure might be different. Error: {e}")


    if num_layers != target_num_layers:
        print(f"Warning: Layer count mismatch! MaxText={num_layers}, HF Target={target_num_layers}. Will convert up to {min(num_layers, target_num_layers)} layers.")
        num_layers = min(num_layers, target_num_layers) # Adjust target based on available layers

    query_pre_attn_scalar = target_head_dim # From HF Config (used for reverse_scale)

    # Iterate through layers
    for layer_idx in tqdm(range(num_layers), desc='Converting layers'):
        prefix = f"model.layers.{layer_idx}."
        try:
            # Attention weights (Q, K, V, O)
            # MaxText shapes often have layer index as the second dimension [:, layer_idx]
            # HF Target shape needs transposing and reshaping

            # Q: MaxText (Hidden, Layers, Heads, HeadDim) -> HF (Heads*HeadDim, Hidden) [Transposed]
            q_mt = np.array(maxtext_layers["self_attention"]["query"]["kernel"][:, layer_idx]) # Shape (Hidden, Heads, HeadDim)
            q_mt_scaled = reverse_scale(q_mt, query_pre_attn_scalar)
            q_hf = q_mt_scaled.reshape(target_hidden_size, -1).T # Reshape to (Hidden, Heads*HeadDim) then Transpose
            hf_state[prefix + "self_attn.q_proj.weight"] = torch.tensor(q_hf.astype(np.float32), dtype=torch.bfloat16)

            # K: MaxText (Hidden, Layers, KVHeads, HeadDim) -> HF (KVHeads*HeadDim, Hidden) [Transposed]
            k_mt = np.array(maxtext_layers["self_attention"]["key"]["kernel"][:, layer_idx]) # Shape (Hidden, KVHeads, HeadDim)
            k_hf = k_mt.reshape(target_hidden_size, -1).T # Reshape to (Hidden, KVHeads*HeadDim) then Transpose
            hf_state[prefix + "self_attn.k_proj.weight"] = torch.tensor(k_hf.astype(np.float32), dtype=torch.bfloat16)

            # V: MaxText (Hidden, Layers, KVHeads, HeadDim) -> HF (KVHeads*HeadDim, Hidden) [Transposed]
            v_mt = np.array(maxtext_layers["self_attention"]["value"]["kernel"][:, layer_idx]) # Shape (Hidden, KVHeads, HeadDim)
            v_hf = v_mt.reshape(target_hidden_size, -1).T # Reshape to (Hidden, KVHeads*HeadDim) then Transpose
            hf_state[prefix + "self_attn.v_proj.weight"] = torch.tensor(v_hf.astype(np.float32), dtype=torch.bfloat16)

            # O: MaxText (Heads, Layers, HeadDim, Hidden) -> HF (Hidden, Heads*HeadDim) [Transposed]
            o_mt = np.array(maxtext_layers["self_attention"]["out"]["kernel"][:, layer_idx]) # Shape (Heads, HeadDim, Hidden)
            o_hf = o_mt.reshape(-1, target_hidden_size).T # Reshape to (Heads*HeadDim, Hidden) then Transpose
            hf_state[prefix + "self_attn.o_proj.weight"] = torch.tensor(o_hf.astype(np.float32), dtype=torch.bfloat16)


            # MLP block (gate, up, down)
            # Gate: MaxText (Hidden, Layers, Intermediate) -> HF (Intermediate, Hidden) [Transposed]
            gate_mt = np.array(maxtext_layers["mlp"]["wi_0"]["kernel"][:, layer_idx]) # Shape (Hidden, Intermediate)
            hf_state[prefix + "mlp.gate_proj.weight"] = torch.tensor(gate_mt.T.astype(np.float32), dtype=torch.bfloat16)

            # Up: MaxText (Hidden, Layers, Intermediate) -> HF (Intermediate, Hidden) [Transposed]
            up_mt = np.array(maxtext_layers["mlp"]["wi_1"]["kernel"][:, layer_idx]) # Shape (Hidden, Intermediate)
            hf_state[prefix + "mlp.up_proj.weight"] = torch.tensor(up_mt.T.astype(np.float32), dtype=torch.bfloat16)

            # Down: MaxText (Intermediate, Layers, Hidden) -> HF (Hidden, Intermediate) [Transposed]
            down_mt = np.array(maxtext_layers["mlp"]["wo"]["kernel"][:, layer_idx]) # Shape (Intermediate, Hidden)
            hf_state[prefix + "mlp.down_proj.weight"] = torch.tensor(down_mt.T.astype(np.float32), dtype=torch.bfloat16)

            # Norm layers
            # MaxText shapes: (HiddenDim, NumLayers) or (HeadDim, NumLayers) -> index [:, layer_idx] -> (Dim,)
            inp_norm = np.array(maxtext_layers["pre_self_attention_norm"]["scale"][:, layer_idx]) - 1
            hf_state[prefix + "input_layernorm.weight"] = torch.tensor(inp_norm.astype(np.float32), dtype=torch.bfloat16)

            post_attn_norm = np.array(maxtext_layers["post_self_attention_norm"]["scale"][:, layer_idx]) - 1
            hf_state[prefix + "post_attention_layernorm.weight"] = torch.tensor(post_attn_norm.astype(np.float32), dtype=torch.bfloat16)

            pre_ffw_norm = np.array(maxtext_layers["pre_ffw_norm"]["scale"][:, layer_idx]) - 1
            hf_state[prefix + "pre_feedforward_layernorm.weight"] = torch.tensor(pre_ffw_norm.astype(np.float32), dtype=torch.bfloat16)

            post_ffw_norm = np.array(maxtext_layers["post_ffw_norm"]["scale"][:, layer_idx]) - 1
            hf_state[prefix + "post_feedforward_layernorm.weight"] = torch.tensor(post_ffw_norm.astype(np.float32), dtype=torch.bfloat16)

            # QK Norms (Gemma 3 specific)
            # MaxText: (HeadDim, Layers) -> HF (HeadDim,)
            if "query_norm" in maxtext_layers["self_attention"]:
                q_norm = np.array(maxtext_layers["self_attention"]["query_norm"]["scale"][:, layer_idx]) - 1
                hf_state[prefix + "self_attn.q_norm.weight"] = torch.tensor(
                    q_norm.astype(np.float32), dtype=torch.bfloat16
                )
            else:
                 print(f"Warning: query_norm not found in MaxText self_attention for layer {layer_idx}")


            if "key_norm" in maxtext_layers["self_attention"]:
                k_norm = np.array(maxtext_layers["self_attention"]["key_norm"]["scale"][:, layer_idx]) - 1
                hf_state[prefix + "self_attn.k_norm.weight"] = torch.tensor(
                    k_norm.astype(np.float32), dtype=torch.bfloat16
                )
            else:
                print(f"Warning: key_norm not found in MaxText self_attention for layer {layer_idx}")


        except KeyError as e:
            print(f"KeyError encountered processing layer {layer_idx}: {e}. Check MaxText param name.")
            raise e
        except Exception as e:
            print(f"An unexpected error occurred processing layer {layer_idx}: {e}")
            raise e

    # --- LM head ---
    print("Converting LM head...")
    # Assume tied weights as lm_head key wasn't observed in the structure dump
    print("LM head key not found, assuming tied weights.")
    hf_state["lm_head.weight"] = hf_state["model.embed_tokens.weight"].clone() # Already sliced to target vocab

    return hf_state


def main(argv: Sequence[str]):
    config = pyconfig.initialize(argv[:-2])
    hf_model_path = argv[-2].split("=")[1]
    model_size = argv[-1].split("=")[1]
    print(f"Will save converted HuggingFace checkpoint to path = {hf_model_path}")

    print("Load hf checkpoint")
    hf_model = load_hf_model(model_size)

    print("Checkpoint loaded; converting parameters...")
    hf_state_dict = convert_maxtext_to_hf(config, model_size)

    print("Conversion complete; saving Hugging Face checkpoint to", hf_model_path)
    hf_model.save_pretrained(hf_model_path, state_dict=hf_state_dict)
    print("Done.")


if __name__ == "__main__":
    app.run(main)
