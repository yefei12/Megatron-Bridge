# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import types
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from megatron.core.tensor_parallel.layers import ColumnParallelLinear
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.module import MegatronModule
from torch import Tensor
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.utils.common_utils import hook_hf_module_setattr_for_tp_grad_sync
from megatron.bridge.utils.import_utils import safe_import_from
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core import InferenceParams, tensor_parallel


TENorm, _ = safe_import_from("megatron.core.extensions.transformer_engine", "TENorm")


class KimiK25VLModel(MegatronModule):
    """
    Kimi K2.5 Vision-Language (VL) model wrapper for Megatron.
    Args:
        config (GPTModelProvider): Model provider containing configuration for language and vision modules.
        pre_process (bool, optional): Whether to construct the vision tower and projector. Default: True.
        post_process (bool, optional): Whether to apply post-processing. Default: True.
        vp_stage (Optional[int], optional): Pipeline stage for model parallelism. Default: None.

    Attributes:
        pre_process (bool): If True, enables vision and multimodal components.
        post_process (bool): If True, enables post-processing.
        vp_stage (Optional[int]): Pipeline stage for model parallelism.
        vision_tower (nn.Module): Vision encoder (MoonViT3d vision backbone).
        mm_projector (nn.Module): PatchMergerMLP that projects vision features to language model space.
        language_model (nn.Module): The underlying DeepSeek V3 language model.
        get_image_features (callable): Method to extract and project image features.

    Forward Inputs:
        input_ids (torch.LongTensor, optional): Tokenized input ids for the language model.
        attention_mask (torch.Tensor, optional): Attention mask for the language model.
        position_ids (torch.LongTensor, optional): Position ids for the language model.
        inputs_embeds (torch.FloatTensor, optional): Precomputed input embeddings.
        pixel_values (torch.Tensor, optional): Image tensor(s) for the vision tower.
        labels (torch.Tensor, optional): Target labels for supervised training.
        runtime_gather_output (bool, optional): If True, gather outputs across pipeline stages.
        loss_mask (Tensor, optional): Mask for loss computation.

    Returns:
        Tensor: Model output (e.g., logits or loss, depending on mode).

    Note:
        - If `pre_process` is False, only the language model is constructed.
        - The vision tower and projector are only active if `pre_process` is True.
        - This class is intended for use within the Megatron-LM framework.
    """

    def __init__(
        self,
        config: GPTModelProvider,
        pre_process: bool = True,
        post_process: bool = True,
        vp_stage: Optional[int] = None,
    ) -> None:
        super().__init__(config=config)

        self.pre_process = pre_process
        self.post_process = post_process
        self.vp_stage = vp_stage

        if config.hf_model_path is None:
            raise ValueError("hf_model_path must be set.")

        self.config.image_token_index: int = getattr(config, "image_token_index", 163605)
        self.config.pad_token_id: int = getattr(config, "pad_token_id", 163839)
        self.config.ignore_index: int = getattr(config, "ignore_index", -100)
        
        KimiK25ForConditionalGeneration = get_class_from_dynamic_module(
            "modeling_kimi_k25.KimiK25ForConditionalGeneration",
            config.hf_model_path,
        )

        if pre_process:
            # Load vision tower and projector classes from the custom HuggingFace model code
            MoonViT3dPretrainedModel = get_class_from_dynamic_module(
                "modeling_kimi_k25.MoonViT3dPretrainedModel",
                config.hf_model_path,
            )
            PatchMergerMLP = get_class_from_dynamic_module(
                "modeling_kimi_k25.PatchMergerMLP",
                config.hf_model_path,
            )
            ProjectorConfig = get_class_from_dynamic_module(
                "modeling_kimi_k25.ProjectorConfig",
                config.hf_model_path,
            )
            VisionTowerConfig = get_class_from_dynamic_module(
                "modeling_kimi_k25.VisionTowerConfig",
                config.hf_model_path,
            )
            self.vision_tower_config = VisionTowerConfig(config.vision_config)
            self.projector_config = ProjectorConfig(config.vision_config)
            self.vision_tower = MoonViT3dPretrainedModel(self.vision_tower_config)
            self.mm_projector = PatchMergerMLP(self.projector_config) # TODO: support different types of mm projector
            # Ensure HF visual tower params are marked for TP grad sync and future assignments are hooked.
            hook_hf_module_setattr_for_tp_grad_sync(self.vision_tower)
            hook_hf_module_setattr_for_tp_grad_sync(self.mm_projector)

        self.language_model = self.config.provide_language_model(
            pre_process=pre_process, post_process=post_process, vp_stage=vp_stage
        )

        # Finalize grad requires these to be bound with module
        self.share_embeddings_and_output_weights = config.share_embeddings_and_output_weights
        self.shared_embedding_or_output_weight = self.language_model.shared_embedding_or_output_weight

        self._extract_image_features = types.MethodType(KimiK25ForConditionalGeneration._extract_image_features, self)
        self._merge_input_ids_with_image_features = types.MethodType(KimiK25ForConditionalGeneration._merge_input_ids_with_image_features, self)

    def set_input_tensor(self, input_tensor) -> None:
        """Set model chunk input tensor."""
        self.language_model.set_input_tensor(input_tensor)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        grid_thws: torch.Tensor = None,
        image_input_mask: torch.Tensor = None,
        labels: Optional[torch.Tensor] = None,
        runtime_gather_output: Optional[bool] = None,
        loss_mask: torch.Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
    ) -> Tensor:
        r"""
        Args:
            input_ids: Tokenized input ids for the language model.
            attention_mask: Attention mask for the language model.
            position_ids: Position ids for the language model.
            inputs_embeds: Precomputed input embeddings.
            pixel_values: Image tensor for the vision tower.
            grid_thws: Tensor of shape (num_images, 3) containing [temporal, height, width]
                for each image's grid dimensions in the LLM.
            labels: Target labels for supervised training.
            runtime_gather_output: If True, gather outputs across pipeline stages.
            loss_mask: Mask for loss computation.
        """
        
        if position_ids is None:
            seq_len = input_ids.size(1)
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand_as(input_ids)

        
        if self.pre_process:
            if inputs_embeds is None:
                inputs_embeds = self.language_model.embedding(
                    input_ids=input_ids, position_ids=None
                ).clone()  # [decoder_seq_len, b, h_language]

                inputs_embeds = inputs_embeds.transpose(1, 0).contiguous()  # [b, decoder_seq_len, h_language]

            if pixel_values is not None:
                image_features = self._extract_image_features(pixel_values.to(self.vision_tower.dtype), grid_thws)
                image_features = self.mm_projector(image_features)
                inputs_embeds = inputs_embeds.to(image_features[0].dtype)

                # Create backup of original inputs
                original_inputs_embeds = inputs_embeds
                original_attention_mask = attention_mask
                original_labels = labels
                original_position_ids = position_ids

                # Check if image token count matches image feature count
                if input_ids is not None:
                    total_image_tokens = (input_ids == self.config.image_token_index).sum().item()

                    if isinstance(image_features, list):
                        num_image_features = len(image_features)
                    elif isinstance(image_features, torch.Tensor):
                        num_image_features = image_features.shape[0]
                    else:
                        num_image_features = 0

                    if total_image_tokens != num_image_features:
                        # Skip image processing due to mismatch
                        inputs_embeds = original_inputs_embeds
                        attention_mask = original_attention_mask
                        labels = original_labels
                        position_ids = original_position_ids
                        skip_image_merge = True
                    else:
                        skip_image_merge = False
                else:
                    skip_image_merge = False

                # Attempt to merge image features if counts match
                if not skip_image_merge:
                    try:
                        inputs_embeds, attention_mask, labels, position_ids = (
                            self._merge_input_ids_with_image_features(
                                image_features,
                                inputs_embeds,
                                input_ids,
                                attention_mask,
                                labels,
                            ))
                    except Exception:
                        # Restore original inputs on failure
                        inputs_embeds = original_inputs_embeds
                        attention_mask = original_attention_mask
                        labels = original_labels
                        position_ids = original_position_ids

                # Disable packed_seq_params if image processing actually happened
                if packed_seq_params is not None and inputs_embeds is not original_inputs_embeds:
                    packed_seq_params = None

            
            attention_mask = self._compute_attention_mask(input_ids)

        # Ensure inputs_embeds is in [T, B, D] format (seq_len, batch, hidden)
        if inputs_embeds.dim() == 3:
            # Force [T, B, D] format for Megatron language model
            # If current shape is [B, T, D], transpose to [T, B, D]
            # If already [T, B, D], transposing twice returns to original
            # This ensures consistent format regardless of input
            # sys.stderr.write(f"[DEBUG Kimi K25 VL] Before transpose: inputs_embeds.shape={inputs_embeds.shape}\n")
            # sys.stderr.flush()
            inputs_embeds = inputs_embeds.transpose(1, 0).contiguous()
            # sys.stderr.write(f"[DEBUG Kimi K25 VL] After transpose: inputs_embeds.shape={inputs_embeds.shape}\n")
            # sys.stderr.flush()

        if self.config.sequence_parallel:
            # Import tensor parallel utilities to get tp size
            try:
                from megatron.core import mpu
                tp_size = mpu.get_tensor_model_parallel_world_size()
                tp_rank = mpu.get_tensor_model_parallel_rank()
            except ImportError:
                tp_size = dist.get_world_size()
                tp_rank = dist.get_rank()

            # sys.stderr.write(f"[DEBUG Kimi K25 VL] Rank {dist.get_rank()} (TP rank {tp_rank}): scatter_to_sequence_parallel_region - inputs_embeds.shape={inputs_embeds.shape}, tp_size={tp_size}, divisible? {inputs_embeds.shape[0] % tp_size == 0}\n")
            # sys.stderr.flush()
            # sys.stderr.write(f"[DEBUG Kimi K25 VL] Full shape info: dims={inputs_embeds.shape}, sequence_parallel={self.config.sequence_parallel}\n")
            # sys.stderr.flush()

            # Check and pad if needed before scattering
            seq_len = inputs_embeds.shape[0]
            if seq_len % tp_size != 0:
                pad_needed = tp_size - (seq_len % tp_size)
                # sys.stderr.write(f"[WARNING Kimi K25 VL] First dimension {seq_len} not divisible by tp_size {tp_size}. Padding with {pad_needed} zeros.\n")
                # sys.stderr.flush()

                # Pad inputs_embeds along sequence dimension (first dimension)
                # inputs_embeds shape: [T, B, D]
                inputs_embeds = torch.nn.functional.pad(
                    inputs_embeds,
                    (0, 0, 0, 0, 0, pad_needed),  # pad last dim (D), then second (B), then first (T)
                    mode='constant',
                    value=0
                )
                # sys.stderr.write(f"[DEBUG Kimi K25 VL] After padding: inputs_embeds.shape={inputs_embeds.shape}\n")
                # sys.stderr.flush()

                # IMPORTANT: If we have labels, attention_mask, position_ids, they also need padding
                # However, in Kimi VL model, these might be handled by the language model internally
                # We'll rely on the language model to handle padded inputs

            inputs_embeds = tensor_parallel.scatter_to_sequence_parallel_region(inputs_embeds)
            inputs_embeds = inputs_embeds.contiguous()
            # sys.stderr.write(f"[DEBUG Kimi K25 VL] After scatter: shape={inputs_embeds.shape}\n")
            # sys.stderr.flush()

        outputs = self.language_model.forward(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,  # (B, 1, T, T)
            decoder_input=inputs_embeds,  # (T, B, D)
            labels=labels,  # (B, T)
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
            loss_mask=loss_mask,
            **(extra_block_kwargs or {}),
        )
        return outputs

    def freeze(self, freeze_language_model: bool, freeze_vision_model: bool, freeze_vision_projection: bool):
        """Freeze model modules.

        Make specific modules non-trainable by setting requires_grad to False.

        Args:
            freeze_language_model (bool): Freeze the language model module.
            freeze_vision_model (bool): Freeze the vision model module (patch_embed and blocks).
            freeze_vision_projection (bool): Freeze the vision projection module (merger).
        """
        modules = []

        if freeze_language_model and hasattr(self, "language_model") and self.language_model is not None:
            modules.append(self.language_model)

        if freeze_vision_model and hasattr(self, "vision_tower") and self.vision_tower is not None:
            # Vision model consists of patch_embed and blocks
            modules.append(self.vision_tower)

        if (
            freeze_vision_projection
            and hasattr(self, "mm_projector")
            and self.mm_projector is not None
        ):
            # Vision projection is the merger module
            modules.append(self.mm_projector)

        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

    def _compute_attention_mask(
        self,
        input_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.pre_process:
            return None
        batch_size, seq_len = input_ids.shape
        causal_mask = torch.tril(torch.ones((batch_size, 1, seq_len, seq_len))).to(input_ids.device)

        image_mask = input_ids == self.config.image_token_index
        padded_mask = F.pad(image_mask, (1, 0), value=0)
        boundary = padded_mask[:, 1:] > padded_mask[:, :-1]
        numbered_boundary = torch.cumsum(boundary, dim=-1)
        q_block_indices = image_mask * numbered_boundary
        kv_block_indices = q_block_indices
        bidirectional_mask = torch.logical_and(
            kv_block_indices[:, None, :] == q_block_indices.unsqueeze(-1),
            q_block_indices.unsqueeze(-1) > 0,
        )
        # See te.DotProductAttention for the requirement of custom mask
        attention_mask = ~torch.logical_or(causal_mask, bidirectional_mask.unsqueeze(1))
        return attention_mask