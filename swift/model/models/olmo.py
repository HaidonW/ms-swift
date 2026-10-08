# Copyright (c) ModelScope Contributors. All rights reserved.
from swift.template import TemplateType
from ..constant import LLMModelType
from ..model_arch import ModelArch
from ..model_meta import Model, ModelGroup, ModelMeta
from ..register import register_model

register_model(
    ModelMeta(
        LLMModelType.olmo3,
        [ModelGroup([
            Model(None, 'allenai/Olmo-3-7B-Think'),
        ])],
        template=TemplateType.olmo3_thinking,
        # The projection/module paths match Llama; loading uses native Olmo3ForCausalLM.
        model_arch=ModelArch.llama,
        architectures=['Olmo3ForCausalLM'],
        requires=['transformers>=4.57.0'],
    ))
