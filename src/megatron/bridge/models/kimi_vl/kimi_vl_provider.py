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

"""
Qwen3 VL MoE Model Provider configurations for Megatron-Core.

This module provides configuration classes for Qwen3-VL MoE (Mixture of Experts) multimodal models,
compatible with HuggingFace's Qwen3-VL-MoE model configurations.
Reference: https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct
"""


from dataclasses import dataclass, field
from typing import Optional

from megatron.core.models.gpt import GPTModel

from megatron.bridge.models.deepseek.deepseek_provider import MoonlightModelProvider16B
from megatron.bridge.models.kimi_vl.modelling_kimi_vl.model import KimiVLModel
from megatron.bridge.models.kimi_vl.modelling_kimi_vl.transfomer_config import (
    DeepseekV3Config,
    KimiVLConfig,
    KimiVLMultimodalProjectorConfig,
    MoonViTConfig,
)


@dataclass
class KimiVLMoEModelProvider(MoonlightModelProvider16B):

    vision_config: MoonViTConfig = field(default_factory=lambda: MoonViTConfig())
    vl_config: KimiVLConfig = field(default_factory=lambda: KimiVLConfig())
    multi_modal_projector_config: KimiVLMultimodalProjectorConfig = field(default_factory=lambda: KimiVLMultimodalProjectorConfig())

    hf_text_config: Optional[DeepseekV3Config] = None
    pretrained_model_name: str = "moonshotai/Kimi-VL-A3B-Instruct"

    freeze_language_model: bool = True
    # Whether to freeze vision encoder weights
    freeze_vision_model: bool = True
    # Whether to freeze vision-to-language projection weights
    freeze_vision_projection: bool = False
    scatter_embedding_sequence_parallel: bool = False

    def finalize(self) -> None:
        if self.tensor_model_parallel_size > 1:
            self.sequence_parallel = True

        super().finalize()

    def provide(self, pre_process=None, post_process=None, vp_stage=None):
        
        hf_config = self.vision_config


        model = KimiVLModel(
            self,
            vision_transformer_config=hf_config,
            multi_modal_projector_config=self.multi_modal_projector_config,
            pre_process=pre_process,
            post_process=post_process,
            vp_stage=vp_stage
        )

        # Apply freeze options if any are enabled for fine-tuning
        if self.freeze_language_model or self.freeze_vision_model or self.freeze_vision_projection:
            model.freeze(
                freeze_language_model=self.freeze_language_model,
                freeze_vision_model=self.freeze_vision_model,
                freeze_vision_projection=self.freeze_vision_projection,
            )

        return model

    def provide_language_model(self, pre_process=None, post_process=None, vp_stage=None) -> GPTModel:
        """
        Provide just the language model component without vision.

        Args:
            pre_process: Whether this is the first stage in pipeline parallelism
            post_process: Whether this is the last stage in pipeline parallelism
            vp_stage: Virtual pipeline stage number

        Returns:
            GPTModel instance (language model only)
        """
        # Use parent class to create standard language model
        return super().provide(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)


