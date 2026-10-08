# Copyright (c) ModelScope Contributors. All rights reserved.
"""Offline OLMo 3 Think regression tests; no model weights or hub access are needed."""
import pytest
import torch
from copy import deepcopy
from peft import LoraConfig, get_peft_model
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PretrainedConfig, PreTrainedTokenizerFast

from swift.model import LLMModelType, ModelInfo, get_matched_model_meta, get_model_info_meta, get_model_processor
from swift.model.model_meta import get_matched_model_types
from swift.template import TemplateType, get_template
from swift.utils import find_all_linears

_DEFAULT_SYSTEM = (
    'You are OLMo, a helpful function-calling AI assistant built by Ai2. Your date cutoff is November 2024, '
    'and your model weights are available at https://huggingface.co/allenai.')
_NO_TOOLS = ' You do not currently have access to any functions. <functions></functions>'


@pytest.fixture
def tokenizer(tmp_path):
    # Byte-level BPE without merges round-trips arbitrary text and whitespace exactly.
    tokens = ['<unk>', '<pad>', '<|endoftext|>', '<|im_start|>', '<|im_end|>']
    tokens += sorted(pre_tokenizers.ByteLevel.alphabet())
    backend = Tokenizer(models.BPE({token: index for index, token in enumerate(tokens)}, [], unk_token='<unk>'))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token='<unk>',
        pad_token='<pad>',
        eos_token='<|endoftext|>',
        additional_special_tokens=['<|im_start|>', '<|im_end|>'],
        clean_up_tokenization_spaces=False,
    )
    tokenizer.model_info = ModelInfo(
        model_type=LLMModelType.olmo3,
        model_dir=str(tmp_path),
        torch_dtype=torch.float32,
        max_model_len=4096,
        quant_method=None,
        quant_bits=None,
        config=PretrainedConfig(),
        task_type='causal_lm',
    )
    tokenizer.model_meta = get_matched_model_meta('allenai/Olmo-3-7B-Think')
    return tokenizer


def _tiny_config(tokenizer):
    Olmo3Config = pytest.importorskip('transformers.models.olmo3.configuration_olmo3').Olmo3Config
    return Olmo3Config(
        vocab_size=len(tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=2048,
        layer_types=['sliding_attention', 'full_attention'],
        sliding_window=16,
        rope_theta=500000,
        rope_scaling={
            'rope_type': 'yarn',
            'factor': 8.0,
            'original_max_position_embeddings': 256,
        },
        bos_token_id=None,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=False,
        architectures=['Olmo3ForCausalLM'],
    )


@pytest.mark.parametrize('model_id', [
    'allenai/Olmo-3-7B-Think',
    '/models/Olmo-3-7B-Think',
])
def test_olmo3_registered_model_id(model_id):
    meta = get_matched_model_meta(model_id)
    assert meta.model_type == LLMModelType.olmo3
    assert meta.template == TemplateType.olmo3_thinking
    assert get_matched_model_types(['Olmo3ForCausalLM']) == [LLMModelType.olmo3]


def test_olmo3_local_config_auto_detection(tokenizer, tmp_path):
    # The arbitrary directory name forces architecture-based detection.
    config = _tiny_config(tokenizer)
    config.save_pretrained(tmp_path)
    info, meta = get_model_info_meta(str(tmp_path))
    assert info.model_type == LLMModelType.olmo3
    assert info.max_model_len == config.max_position_embeddings
    assert meta.template == TemplateType.olmo3_thinking


@pytest.mark.parametrize('system', [None, '', 'Custom system.'])
def test_olmo3_single_round_sft_and_loss_mask(tokenizer, system):
    template = get_template(tokenizer)
    template.set_mode('train')
    expected_system = _DEFAULT_SYSTEM if system is None else system
    prompt = ('<|im_start|>system\n' + expected_system + _NO_TOOLS + '<|im_end|>\n'
              '<|im_start|>user\n Question? \n<|im_end|>\n<|im_start|>assistant\n')
    answer = '<think>\nReason.</think>\nAnswer. <|endoftext|>'
    messages = [
        {
            'role': 'user',
            'content': ' Question? \n'
        },
        {
            'role': 'assistant',
            'content': '<think>\nReason.</think>\nAnswer. '
        },
    ]
    if system is not None:
        messages.insert(0, {'role': 'system', 'content': system})
    encoded = template.encode({'messages': messages})
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    answer_ids = tokenizer.encode(answer, add_special_tokens=False)
    assert encoded['input_ids'] == prompt_ids + answer_ids
    assert encoded['labels'] == [-100] * len(prompt_ids) + answer_ids


def test_olmo3_multiround_sft_preserves_thinking_and_masks_prompts(tokenizer):
    template = get_template(tokenizer)
    template.set_mode('train')
    chunks = [
        ('<|im_start|>system\nSystem.' + _NO_TOOLS + '<|im_end|>\n'
         '<|im_start|>user\nFirst?<|im_end|>\n<|im_start|>assistant\n', False),
        ('<think>First reasoning.</think>First answer.', True),
        ('<|im_end|>\n<|im_start|>user\nSecond?<|im_end|>\n<|im_start|>assistant\n', False),
        ('<think>Second reasoning.</think>Second answer.<|endoftext|>', True),
    ]
    encoded = template.encode({
        'messages': [
            {
                'role': 'system',
                'content': 'System.'
            },
            {
                'role': 'user',
                'content': 'First?'
            },
            {
                'role': 'assistant',
                'content': '<think>First reasoning.</think>First answer.'
            },
            {
                'role': 'user',
                'content': 'Second?'
            },
            {
                'role': 'assistant',
                'content': '<think>Second reasoning.</think>Second answer.'
            },
        ]
    })
    expected_ids, expected_labels = [], []
    for text, supervised in chunks:
        ids = tokenizer.encode(text, add_special_tokens=False)
        expected_ids.extend(ids)
        expected_labels.extend(ids if supervised else [-100] * len(ids))
    assert encoded['input_ids'] == expected_ids
    assert encoded['labels'] == expected_labels


def test_olmo3_inference_preserves_history_and_has_exact_thinking_prefix(tokenizer):
    template = get_template(tokenizer)
    messages = [
        {
            'role': 'system',
            'content': 'System.'
        },
        {
            'role': 'user',
            'content': 'First?'
        },
        {
            'role': 'assistant',
            'content': '<think>Keep this reasoning.</think>First answer.'
        },
        {
            'role': 'user',
            'content': 'Second?'
        },
    ]
    original = deepcopy(messages)
    encoded = template.encode({'messages': messages})
    expected = ('<|im_start|>system\nSystem.' + _NO_TOOLS + '<|im_end|>\n'
                '<|im_start|>user\nFirst?<|im_end|>\n'
                '<|im_start|>assistant\n<think>Keep this reasoning.</think>First answer.<|im_end|>\n'
                '<|im_start|>user\nSecond?<|im_end|>\n<|im_start|>assistant\n<think>')
    assert tokenizer.decode(encoded['input_ids']) == expected
    assert 'labels' not in encoded
    assert messages == original


def test_olmo3_default_system_inference(tokenizer):
    template = get_template(tokenizer)
    encoded = template.encode({'messages': [{'role': 'user', 'content': 'Question?'}]})
    expected = ('<|im_start|>system\n' + _DEFAULT_SYSTEM + _NO_TOOLS + '<|im_end|>\n'
                '<|im_start|>user\nQuestion?<|im_end|>\n<|im_start|>assistant\n<think>')
    assert tokenizer.decode(encoded['input_ids']) == expected


def test_olmo3_plain_sft_response_is_not_rewritten(tokenizer):
    template = get_template(tokenizer)
    template.set_mode('train')
    encoded = template.encode(
        {'messages': [{
            'role': 'user',
            'content': 'Question?'
        }, {
            'role': 'assistant',
            'content': 'Answer.'
        }]})
    supervised_ids = [label for label in encoded['labels'] if label != -100]
    assert tokenizer.decode(supervised_ids) == 'Answer.<|endoftext|>'


def test_olmo3_local_sdpa_sft_and_lora(tokenizer, tmp_path):
    # Exercise Swift's actual loading and template selection using random local weights.
    config = _tiny_config(tokenizer)
    from transformers import Olmo3ForCausalLM

    Olmo3ForCausalLM(config).save_pretrained(tmp_path)
    tokenizer.save_pretrained(tmp_path)
    model, processor = get_model_processor(str(tmp_path), device_map='cpu', attn_impl='sdpa', torch_dtype=torch.float32)
    assert isinstance(model, Olmo3ForCausalLM)
    assert model.config._attn_implementation == 'sdpa'
    assert model.config.layer_types == ['sliding_attention', 'full_attention']
    template = get_template(processor)
    assert template.template_meta.template_type == TemplateType.olmo3_thinking
    template.set_mode('train')
    encoded = template.encode({
        'messages': [
            {
                'role': 'system',
                'content': 'System.'
            },
            {
                'role': 'user',
                'content': 'Question?'
            },
            {
                'role': 'assistant',
                'content': '<think>Reason.</think>Answer.'
            },
        ]
    })
    batch = template.data_collator([encoded])
    model.train()
    loss = model(**batch).loss
    assert torch.isfinite(loss)
    loss.backward()
    assert model.model.layers[0].self_attn.q_proj.weight.grad is not None
    model.zero_grad(set_to_none=True)

    targets = find_all_linears(model)
    assert set(targets) == {'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'}
    model = get_peft_model(model, LoraConfig(r=2, lora_alpha=4, target_modules=targets, task_type='CAUSAL_LM'))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    assert model.is_gradient_checkpointing
    loss = model(**batch).loss
    assert torch.isfinite(loss)
    loss.backward()
    lora_grads = [parameter.grad for name, parameter in model.named_parameters() if 'lora_B' in name]
    assert lora_grads and all(gradient is not None and torch.isfinite(gradient).all() for gradient in lora_grads)
    assert any(torch.count_nonzero(gradient) > 0 for gradient in lora_grads)
    assert model.base_model.model.model.layers[0].self_attn.q_proj.base_layer.weight.grad is None
