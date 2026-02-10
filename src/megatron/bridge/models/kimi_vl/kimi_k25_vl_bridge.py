import math

import torch
from transformers import Gemma3ForConditionalGeneration

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge

from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    QKVMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.hf_pretrained.vlm import PreTrainedVLM
from megatron.bridge.models.kimi_vl.kimi_k25_vl_provider import KimiK25VLModelProvider
from megatron.bridge.models.kimi_vl.modeling_kimi_k25_vl import KimiK25VLModel
from megatron.bridge.models.deepseek.common import get_common_configs, get_common_mapping_list

@MegatronModelBridge.register_bridge(source="KimiK25ForConditionalGeneration", target=KimiK25VLModel)
class KimiK25VLBridge(MegatronModelBridge):
    """
    Megatron Bridge for Kimi K2.5 VL.
    """

    def provider_bridge(self, hf_pretrained: PreTrainedVLM) -> KimiK25VLModelProvider:
        hf_config = hf_pretrained.config
        text_config = hf_config.text_config
        vision_config = hf_config.vision_config

        # get_common_configs expects TextConfig
        hf_pretrained.config = text_config
        configs = get_common_configs(hf_pretrained)

        configs["make_vocab_size_divisible_by"] = 1280
        configs["moe_router_score_function"] = "sigmoid"
        configs["moe_router_enable_expert_bias"] = True
        # aux_loss_alpha is not set in all DSv3 HF configs
        if hasattr(hf_config, "aux_loss_alpha"):
            configs["moe_aux_loss_coeff"] = hf_config.aux_loss_alpha

        provider = KimiK25VLModelProvider(
            # Text configuration
            **configs,
            # Vision configuration
            vision_config=vision_config,
            # VL-specific token IDs
            bos_token_id=text_config.bos_token_id,
            eos_token_id=text_config.eos_token_id,
            media_placeholder_token_id=hf_config.media_placeholder_token_id,
            # Precision configuration
            fp16=(self.dtype_from_hf(hf_config, default=torch.float32) == torch.float16),
            bf16=(self.dtype_from_hf(hf_config, default=torch.float32) == torch.bfloat16),
            params_dtype=self.dtype_from_hf(hf_config, default=torch.float32),
            # misc
            hf_model_path=hf_pretrained._model_name_or_path,
        )

        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        # Return MegatronMappingRegistry containing parameter mappings from Megatron to HF format
        # First create simple 1:1 parameter mappings using a dictionary for readability
        """
        Return MegatronMappingRegistry containing parameter mappings for MoE models.
        The MoE mappings include:
        1. Standard language model mappings (embeddings, layer norms, output)
        2. Vision model mappings (same as dense model)
        3. QKV mappings with QK layernorm
        4. MoE-specific mappings:
           - Router weights for expert selection
           - Expert MLPs (multiple experts per layer)
           - Pre-MLP layernorm
        5. Deepstack visual merger mappings
        Returns:
            MegatronMappingRegistry with all MoE parameter mappings
        """

        hf_prefix = "language_model."

        # Language model direct mappings (DeepSeek-style MLA/MoE)
        param_mappings = {
            # Embeddings and output layers
            "language_model.embedding.word_embeddings.weight": f"{hf_prefix}model.embed_tokens.weight",
            "language_model.output_layer.weight": f"{hf_prefix}lm_head.weight",
            "language_model.decoder.final_layernorm.weight": f"{hf_prefix}model.norm.weight",
            # Layer normalization for attention
            "language_model.decoder.layers.*.input_layernorm.weight": f"{hf_prefix}model.layers.*.input_layernorm.weight",
            # MoE-specific: pre-MLP layernorm
            "language_model.decoder.layers.*.pre_mlp_layernorm.weight": f"{hf_prefix}model.layers.*.post_attention_layernorm.weight",
            "language_model.decoder.layers.*.mlp.linear_fc1.layer_norm_weight": f"{hf_prefix}model.layers.*.post_attention_layernorm.weight",
            # Attention output projection
            "language_model.decoder.layers.*.self_attention.linear_proj.weight": f"{hf_prefix}model.layers.*.self_attn.o_proj.weight",
            # MLA Q/KV projections
            "language_model.decoder.layers.*.self_attention.linear_q_proj.weight": f"{hf_prefix}model.layers.*.self_attn.q_proj.weight",
            "language_model.decoder.layers.*.self_attention.linear_kv_down_proj.weight": f"{hf_prefix}model.layers.*.self_attn.kv_a_proj_with_mqa.weight",
            "language_model.decoder.layers.*.self_attention.linear_kv_up_proj.weight": f"{hf_prefix}model.layers.*.self_attn.kv_b_proj.weight",
            "language_model.decoder.layers.*.self_attention.linear_kv_up_proj.layer_norm_weight": f"{hf_prefix}model.layers.*.self_attn.kv_a_layernorm.weight",
            "language_model.decoder.layers.*.self_attention.kv_layernorm.weight": f"{hf_prefix}model.layers.*.self_attn.kv_a_layernorm.weight",
            # MoE router weights
            "language_model.decoder.layers.*.mlp.router.weight": f"{hf_prefix}model.layers.*.mlp.gate.weight",
            "language_model.decoder.layers.*.mlp.router.expert_bias": f"{hf_prefix}model.layers.*.mlp.gate.e_score_correction_bias",
            # Dense/Shared experts down proj
            "language_model.decoder.layers.*.mlp.linear_fc2.weight": f"{hf_prefix}model.layers.*.mlp.down_proj.weight",
            "language_model.decoder.layers.*.mlp.shared_experts.linear_fc2.weight": f"{hf_prefix}model.layers.*.mlp.shared_experts.down_proj.weight",
        }

        mapping_list = []
        for megatron_param, hf_param in param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        mapping_list.extend(
            [
                # Vision tower
                ReplicatedMapping(
                    megatron_param="vision_tower.**",
                    hf_param="vision_tower.**",
                ),
                ReplicatedMapping(
                    megatron_param="mm_projector.**",
                    hf_param="mm_projector.**",
                ),
                # Dense MLP mappings (for non-MoE layers)
                GatedMLPMapping(
                    megatron_param="language_model.decoder.layers.*.mlp.linear_fc1.weight",
                    gate=f"{hf_prefix}model.layers.*.mlp.gate_proj.weight",
                    up=f"{hf_prefix}model.layers.*.mlp.up_proj.weight",
                ),
                # Expert MLP mappings (gate/up are separate in HF)
                GatedMLPMapping(
                    megatron_param="language_model.decoder.layers.*.mlp.experts.linear_fc1.weight*",
                    gate=f"{hf_prefix}model.layers.*.mlp.experts.*.gate_proj.weight",
                    up=f"{hf_prefix}model.layers.*.mlp.experts.*.up_proj.weight",
                ),
                AutoMapping(
                    megatron_param="language_model.decoder.layers.*.mlp.experts.linear_fc2.weight*",
                    hf_param=f"{hf_prefix}model.layers.*.mlp.experts.*.down_proj.weight",
                ),
                # Shared experts gate+up projections
                GatedMLPMapping(
                    megatron_param="language_model.decoder.layers.*.mlp.shared_experts.linear_fc1.weight",
                    gate=f"{hf_prefix}model.layers.*.mlp.shared_experts.gate_proj.weight",
                    up=f"{hf_prefix}model.layers.*.mlp.shared_experts.up_proj.weight",
                ),
            ]
        )
        return MegatronMappingRegistry(*mapping_list)