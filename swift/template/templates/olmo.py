# Copyright (c) ModelScope Contributors. All rights reserved.
from ..base import Template
from ..constant import LLMTemplateType
from ..register import register_template
from ..template_meta import TemplateMeta


class Olmo3Template(Template):

    def _get_system(self, inputs):
        system = super()._get_system(inputs)
        # Match the official chat_template.jinja for conversations without functions,
        # including an explicitly empty system message.
        return system + ' You do not currently have access to any functions. <functions></functions>'


register_template(
    TemplateMeta(
        LLMTemplateType.olmo3_thinking,
        template_cls=Olmo3Template,
        prefix=[],
        system_prefix=['<|im_start|>system\n{{SYSTEM}}<|im_end|>\n'],
        prompt=['<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n'],
        chat_sep=['<|im_end|>\n'],
        suffix=['<|endoftext|>'],
        default_system=('You are OLMo, a helpful function-calling AI assistant built by Ai2. '
                        'Your date cutoff is November 2024, and your model weights are available at '
                        'https://huggingface.co/allenai.'),
        stop_words=['<|im_end|>'],
        is_thinking=True,
        thinking_prefix='<think>',
        preserve_thinking=True,
    ))
