from typing import Optional

import torch
import transformers
from megatron.core import InferenceParams, tensor_parallel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from packaging.version import Version as PkgVersion
from torch import nn

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.kimi_vl.modelling_kimi_vl.moonvit import MoonVitPretrainedModel
from megatron.bridge.models.kimi_vl.modelling_kimi_vl.transfomer_config import (
    KimiVLMultimodalProjectorConfig,
    MoonViTConfig,
)
from megatron.bridge.utils.common_utils import hook_hf_module_setattr_for_tp_grad_sync


def is_transformers_min_version(version):
    """Check if minimum version of transformers is installed."""
    try:
        transformers_version = PkgVersion(transformers.__version__)
        return transformers_version >= PkgVersion(version)
    except Exception:
        # If version parsing fails, assume false for safety
        return False


class KimiVLModel(MegatronModule):
    def __init__(
        self,
        config: GPTModelProvider,
        vision_transformer_config: MoonViTConfig,
        multi_modal_projector_config: KimiVLMultimodalProjectorConfig,
        pre_process: bool = True,
        post_process: bool = True,
        vp_stage: Optional[int] = None,
    ) -> None:
        super().__init__(config=config)

        self.pre_process = pre_process
        self.post_process = post_process
        self.vp_stage = vp_stage

        self.vision_model = None
        self.image_token_id = getattr(config, "image_token_id", 163605)
        self.video_token_id = getattr(config, "video_token_id", 151656)
        self.vision_start_token_id = getattr(config, "vision_start_token_id", 151652)

        # This attribute is needed to check if an all-reduce is required
        # on the word embeddings inside `finalize_model_grads._allreduce_word_embedding_grads`.
        self.share_embeddings_and_output_weights = False

        if self.pre_process:
            # Initialize vision model with random weights from config
            self.vision_model = MoonVitPretrainedModel._from_config(vision_transformer_config)
            self.multi_modal_projector = KimiVLMultiModalProjector(multi_modal_projector_config)

            # Ensure HF visual tower params are marked for TP grad sync and future assignments are hooked.
            hook_hf_module_setattr_for_tp_grad_sync(self.vision_model)

        self.language_model = self.config.provide_language_model(
            pre_process=pre_process, post_process=post_process, vp_stage=vp_stage
        )

    def shared_embedding_or_output_weight(self):
        """This is a convenience method to surface the language model's word embeddings, which is
        necessary for `finalize_model_grads._allreduce_word_embedding_grads`."""
        if self.add_decoder:
            return self.language_model.shared_embedding_or_output_weight()
        return None

    def set_input_tensor(self, input_tensor) -> None:
        """Set model chunk input tensor for pipeline parallelism."""
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1, "input_tensor should only be length 1 for KimiVL"

        if self.pre_process:
            self.encoder_hidden_state = input_tensor[0]
        else:
            self.language_model.set_input_tensor(input_tensor[0])

    def freeze(
        self,
        freeze_language_model: bool,
        freeze_vision_model: bool,
        freeze_vision_projection: bool,
    ):
        modules = []

        if freeze_language_model and self.language_model is not None:
            modules.append(self.language_model)

        if freeze_vision_model and self.vision_model is not None:
            # Freeze vision encoder components (patch_embed, blocks, pos_embed, rotary_pos_emb)
            if hasattr(self.vision_model, "patch_embed"):
                modules.append(self.vision_model.patch_embed)
            if hasattr(self.vision_model, "encoder"):
                modules.append(self.vision_model.encoder)

        if freeze_vision_projection and self.vision_model is not None:
            # Freeze vision projection components (merger and deepstack_merger_list)
            if hasattr(self.multi_modal_projector, "pre_norm"):
                modules.append(self.multi_modal_projector.pre_norm)
            if hasattr(self.multi_modal_projector, "linear_1"):
                modules.append(self.multi_modal_projector.linear_1)
            if hasattr(self.multi_modal_projector, "act"):
                modules.append(self.multi_modal_projector.act)
            if hasattr(self.multi_modal_projector, "linear_2"):
                modules.append(self.multi_modal_projector.linear_2)

        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        loss_mask: torch.Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        pixel_values: torch.Tensor = None,
        image_grid_hws: torch.Tensor = None,
        image_input_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        if position_ids is None:
            seq_len = input_ids.size(1)
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand_as(input_ids)

        if self.pre_process:
            if image_grid_hws is not None:
                image_mask = image_input_mask
                if image_mask is None:
                    image_mask = (input_ids == self.image_token_id).contiguous()
                vision_grid_hws = image_grid_hws
                vision_data = pixel_values
            vision_embeds = None

            if vision_grid_hws is not None and vision_grid_hws.shape[0] > 0:
                vision_embeds = self.vision_model(vision_data.to(self.vision_model.dtype), image_grid_hws)
                vision_embeds = self.multi_modal_projector(vision_embeds)

            # Get text embeddings
            combined_embeddings = self.language_model.embedding(input_ids, position_ids=None).clone()

            if vision_embeds is not None:
                combined_embeddings = combined_embeddings.transpose(0, 1).contiguous()
                combined_embeddings[image_mask] = vision_embeds
                combined_embeddings = combined_embeddings.transpose(0, 1).contiguous()

            if self.config.sequence_parallel:
                # Check and pad if needed before scattering
                try:
                    from megatron.core import mpu
                    tp_size = mpu.get_tensor_model_parallel_world_size()
                except ImportError:
                    # Fallback
                    import torch.distributed as dist
                    tp_size = dist.get_world_size()

                seq_len = combined_embeddings.shape[0]
                if seq_len % tp_size != 0:
                    pad_needed = tp_size - (seq_len % tp_size)
                    # Debug output
                    import sys
                    sys.stderr.write(f"[WARNING Kimi VL Model] Sequence length {seq_len} not divisible by tp_size {tp_size}. Padding with {pad_needed} zeros.\n")
                    sys.stderr.flush()

                    # Pad combined_embeddings along sequence dimension (first dimension)
                    # combined_embeddings shape: [T, B, D]
                    combined_embeddings = torch.nn.functional.pad(
                        combined_embeddings,
                        (0, 0, 0, 0, 0, pad_needed),  # pad last dim (D), then second (B), then first (T)
                        mode='constant',
                        value=0
                    )
                    sys.stderr.write(f"[DEBUG Kimi VL Model] After padding: combined_embeddings.shape={combined_embeddings.shape}\n")
                    sys.stderr.flush()

                combined_embeddings = tensor_parallel.scatter_to_sequence_parallel_region(combined_embeddings)
                combined_embeddings = combined_embeddings.contiguous()

        else:
            combined_embeddings = None

        return self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels,
            loss_mask=loss_mask,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
            decoder_input=combined_embeddings,
            **(extra_block_kwargs or {}),
        )


class KimiVLMultiModalProjector(MegatronModule):
    """Megatron-style MultiModal Projector for Vision→Text alignment"""

    def __init__(self, config: TransformerConfig):
        super().__init__(config=config)

        vision_hidden_size = config.input_size
        merge_k = config.merge_kernel_size
        self.hidden_size = vision_hidden_size * merge_k[0] * merge_k[1]

        # LayerNorm before projection
        self.pre_norm = nn.LayerNorm(vision_hidden_size, eps=config.layernorm_epsilon)

        # Linear 1 — Column Parallel (split output dimension across GPUs)
        self.linear_1 = ColumnParallelLinear(
            self.hidden_size,
            self.hidden_size,
            init_method=config.init_method,
            bias=True,
            gather_output=False,
            skip_bias_add=False,
            config=config,
        )

        # Activation
        self.act = nn.GELU()

        # Linear 2 — Row Parallel (split input dimension)
        self.linear_2 = RowParallelLinear(
            self.hidden_size,
            config.hidden_size,
            init_method=config.init_method,
            bias=True,
            input_is_parallel=True,
            skip_bias_add=False,
            config=config,
        )

    def forward(self, image_features: list[torch.Tensor]) -> torch.Tensor:
        # Gather image features
        image_features = torch.cat(image_features, dim=0)

        # LayerNorm + reshape
        hidden_states = self.pre_norm(image_features).view(-1, self.hidden_size)

        # First Linear projection
        hidden_states, _ = self.linear_1(hidden_states)

        # Activation
        hidden_states = self.act(hidden_states)

        # Second projection to text hidden size
        hidden_states, _ = self.linear_2(hidden_states)

        return hidden_states
