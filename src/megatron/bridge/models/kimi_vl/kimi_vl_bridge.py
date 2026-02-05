import logging
from typing import Dict, Mapping, Union

import torch
import torch.nn as nn
from megatron.core import parallel_state
from transformers.generation.utils import GenerationMixin
from transformers.modeling_utils import PreTrainedModel

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge, WeightConversionTask
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    QKVMapping,
    ReplicatedMapping,
    RowParallelMapping,
    ColumnParallelMapping
)
from megatron.bridge.models.hf_pretrained.vlm import PreTrainedVLM
from megatron.bridge.models.kimi_vl.kimi_vl_provider import KimiVLMoEModelProvider
from megatron.bridge.models.kimi_vl.modelling_kimi_vl.model import KimiVLModel
from megatron.bridge.models.kimi_vl.modelling_kimi_vl.transfomer_config import KimiVLConfig
from megatron.bridge.utils.common_utils import extract_expert_number_from_param

logger = logging.getLogger(__name__)


class KimiVLPreTrainedModel(PreTrainedModel, GenerationMixin):
    config_class = KimiVLConfig
    base_model_prefix = "model"
    _no_split_modules = ["MoonVitEncoderLayer", "DeepseekV3DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = False

    def _init_weights(self, module):
        # important: this ported version of Llava isn't meant for training from scratch - only
        # inference and fine-tuning - so the proper init weights code has been removed - the original codebase
        # https://github.com/haotian-liu/LLaVA/tree/main/llava should serve for that purpose
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class KimiVLForConditionalGeneration(KimiVLPreTrainedModel, GenerationMixin):
    def __init__(self, config: KimiVLConfig):
        super().__init__(config)


@MegatronModelBridge.register_bridge(source=KimiVLForConditionalGeneration, target=KimiVLModel)
class KimiVLMoEBridge(MegatronModelBridge):

    def __init__(self):
        super().__init__()
        # Cache expert shards during HF export until all ranks contribute.
        self.hf_weights_cache: Dict[str, Dict[int, torch.Tensor]] = {}

    def provider_bridge(self, hf_pretrained: PreTrainedVLM) -> KimiVLMoEModelProvider:       
        hf_config = hf_pretrained.config
        text_config = hf_config.text_config

        model_dtype = self.dtype_from_hf(hf_config, default=torch.float32)
        
        vision_config = hf_config.vision_config
        vision_config.torch_dtype = model_dtype
        
        # breakpoint()
        

        provider = KimiVLMoEModelProvider(
            # Language model configuration from text_config
            num_layers=text_config.num_hidden_layers,
            hidden_size=text_config.hidden_size,
            ffn_hidden_size=text_config.intermediate_size,  # Dense FFN size (for non-MoE layers if any)
            moe_ffn_hidden_size=text_config.moe_intermediate_size,  # Expert FFN size
            num_attention_heads=text_config.num_attention_heads,
            init_method_std=text_config.initializer_range,
            layernorm_epsilon=text_config.rms_norm_eps,
            gated_linear_unit=True,
            make_vocab_size_divisible_by=self.make_vocab_size_divisible_by(text_config.vocab_size),
            rotary_base=text_config.rope_theta,
            share_embeddings_and_output_weights=getattr(text_config, "tie_word_embeddings", False),
            vocab_size=text_config.vocab_size,
            seq_length=text_config.max_position_embeddings,
            fp16=(model_dtype == torch.float16),
            bf16=(model_dtype == torch.bfloat16),
            params_dtype=model_dtype,
            generation_config=hf_pretrained.generation_config,
            vision_config=vision_config,
            hf_text_config=text_config,
        )

        return provider

    def get_hf_tokenizer_kwargs(self) -> dict:
        # Kimi-VL tokenizer relies on custom code.
        return {"trust_remote_code": True}

    def mapping_registry(self) -> MegatronMappingRegistry:
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
                    megatron_param="vision_model.**",
                    hf_param="vision_tower.**",
                ),
                # projector weights
                ReplicatedMapping(
                    megatron_param="multi_modal_projector.pre_norm.**",
                    hf_param="multi_modal_projector.pre_norm.**",
                ),
                ColumnParallelMapping(
                    megatron_param="multi_modal_projector.linear_1.**",
                    hf_param="multi_modal_projector.linear_1.**",
                ),
                RowParallelMapping(
                    megatron_param="multi_modal_projector.linear_2.**",
                    hf_param="multi_modal_projector.linear_2.**",
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

    def maybe_modify_converted_hf_weight(
        self,
        task: WeightConversionTask,
        converted_weights_dict: Dict[str, torch.Tensor],
        hf_state_dict: Mapping[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        # Add rotary inv_freq if expected but missing (export path)
        global_name = task.global_param_name
        if global_name.startswith("language_model.decoder.layers.") and global_name.endswith(".input_layernorm.weight"):
            parts = global_name.split(".")
            if len(parts) >= 4 and parts[3].isdigit():
                layer_idx = int(parts[3])
                inv_freq_key = f"language_model.model.layers.{layer_idx}.self_attn.rotary_emb.inv_freq"
                if inv_freq_key not in converted_weights_dict and inv_freq_key in hf_state_dict:
                    inv_freq = getattr(self, "_kimi_inv_freq", None)
                    if inv_freq is None:
                        text_config = self.hf_config.text_config
                        rotary_dim = text_config.qk_rope_head_dim
                        rotary_base = text_config.rope_theta
                        inv_freq = 1.0 / (rotary_base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
                        self._kimi_inv_freq = inv_freq
                    if converted_weights_dict:
                        reference_tensor = next(iter(converted_weights_dict.values()))
                        if inv_freq.device != reference_tensor.device:
                            inv_freq = inv_freq.to(device=reference_tensor.device)
                            self._kimi_inv_freq = inv_freq
                    # Avoid shared storage across layers for safetensors.
                    converted_weights_dict[inv_freq_key] = inv_freq.clone()

        # If we're exporting per-expert tensors on a single EP rank, just return them.
        if parallel_state.get_expert_model_parallel_world_size() == 1 and any(
            ".mlp.experts." in key or ".mlp.shared_experts." in key for key in converted_weights_dict
        ):
            return converted_weights_dict

        text_config = self.hf_config.text_config
        num_experts = getattr(text_config, "num_experts", None)
        if num_experts is None:
            num_experts = getattr(text_config, "n_routed_experts", None)
        if num_experts is None:
            return converted_weights_dict
        ep_size = parallel_state.get_expert_model_parallel_world_size()
        experts_per_rank = num_experts // ep_size

        try:
            local_expert_number = extract_expert_number_from_param(task.param_name) % experts_per_rank
        except ValueError:
            # not an expert weight
            return converted_weights_dict

        result: Dict[str, torch.Tensor] = {}
        for key, value in converted_weights_dict.items():
            if key not in self.hf_weights_cache:
                self.hf_weights_cache[key] = {}

            # we end up with ep_size many weights to add to the cache
            # unpack the weights and re-index
            if ep_size == 1:
                self.hf_weights_cache[key][local_expert_number] = value
            else:
                assert value.shape[0] == ep_size
                for i, exp_val in enumerate(value):
                    global_expert_number = local_expert_number + (i * experts_per_rank)
                    self.hf_weights_cache[key][global_expert_number] = exp_val
            if len(self.hf_weights_cache[key]) == num_experts:
                logger.debug("All experts are loaded for %s", key)
                # all experts are loaded
                if self.hf_weights_cache[key][0].ndim == 3:  # expert 0
                    # gate up
                    merged_hf_gate_weights = torch.cat(
                        [self.hf_weights_cache[key][i][0].unsqueeze(0) for i in range(num_experts)], dim=0
                    )
                    merged_hf_up_weights = torch.cat(
                        [self.hf_weights_cache[key][i][1].unsqueeze(0) for i in range(num_experts)], dim=0
                    )
                    del self.hf_weights_cache[key]
                    result[key] = torch.cat([merged_hf_gate_weights, merged_hf_up_weights], dim=-1)
                elif self.hf_weights_cache[key][0].ndim == 2:  # expert 0
                    # down
                    merged_hf_down_weights = torch.cat(
                        [self.hf_weights_cache[key][i].unsqueeze(0) for i in range(num_experts)], dim=0
                    )
                    del self.hf_weights_cache[key]
                    result[key] = merged_hf_down_weights
                else:
                    raise ValueError(
                        f"Incorrect shape of self.hf_weights_cache[key]: {key} {self.hf_weights_cache[key].shape}"
                    )
            else:
                # not all experts are loaded yet, return empty dict
                logger.debug("%s/%s experts are loaded for %s", len(self.hf_weights_cache[key]), num_experts, key)
                continue

        if result:
            return result
        return {}


class ExpertMLPDownProjMapping(AutoMapping):
    """Mapping for expert MLP down projection weights between HF and Megatron formats."""

    def hf_to_megatron(self, hf_weights: torch.Tensor, megatron_module: nn.Module) -> torch.Tensor:
        global_expert_number = extract_expert_number_from_param(self.megatron_param)
        # hf_weights: [num_experts, down_in, mlp_out]
        expert_weight = hf_weights[global_expert_number].transpose(0, 1).contiguous()
        return super().hf_to_megatron(expert_weight, megatron_module)

    def megatron_to_hf(self, megatron_weights: torch.Tensor, megatron_module: nn.Module) -> Dict[str, torch.Tensor]:
        # [ep_size, down_in, mlp_out]
        # experts need subsequently merged by maybe_modify_converted_hf_weight
        converted_weights_dict = super().megatron_to_hf(megatron_weights, megatron_module)
        for key in converted_weights_dict:
            converted_weights_dict[key] = converted_weights_dict[key].transpose(-1, -2).contiguous()
        return converted_weights_dict

    def _validate_patterns(self, *args, **kwargs):
        # allow number of wildcards to mismatch in this mapping
        pass


class ExpertMLPGateUpProjMapping(AutoMapping):
    """Mapping for expert MLP gate+up projection using shared GatedMLPMapping logic."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Qwen3-VL MoE expert shards use mismatched wildcard counts; relax validation globally.
        GatedMLPMapping._validate_patterns = lambda *args, **kwargs: None

        # Reuse the generic TP-aware split/gather, but we still handle expert selection
        # and HF<->Megatron transpose at this wrapper layer.
        self._gated_mapping = GatedMLPMapping(
            megatron_param=self.megatron_param,
            gate=f"{self.hf_param}.gate",
            up=f"{self.hf_param}.up",
        )

    def hf_to_megatron(self, hf_weights: Union[torch.Tensor, Dict], megatron_module: nn.Module) -> torch.Tensor:
        global_expert_number = extract_expert_number_from_param(self.megatron_param)
        # hf_weights: [num_experts, mlp_in, fused_gate_up_out]
        expert_weight = hf_weights[global_expert_number].transpose(0, 1).contiguous()

        # HF gate_up_proj is [2 * hidden, hidden]; Megatron expects transposed.
        gate, up = torch.chunk(expert_weight, 2, dim=0)
        return self._gated_mapping.hf_to_megatron({"gate": gate, "up": up}, megatron_module)

    def megatron_to_hf(self, megatron_weights: torch.Tensor, megatron_module: nn.Module) -> Dict[str, torch.Tensor]:
        # Let the shared mapping handle TP/PP/EP gather.
        # We only split gate+up at the end for HF format.
        converted_weights_dict = self._gated_mapping.megatron_to_hf(megatron_weights, megatron_module)
        for key in converted_weights_dict:
            gate, up = torch.chunk(converted_weights_dict[key], 2, dim=-2)
            converted_weights_dict[key] = torch.stack([gate, up], dim=0)
        return converted_weights_dict

    def _validate_patterns(self, *args, **kwargs):
        # allow number of wildcards to mismatch in this mapping
        pass