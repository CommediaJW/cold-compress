import json
import torch
from transformers import AutoConfig, LlamaForCausalLM, AutoTokenizer, AutoModelForCausalLM
from fastchat.model import get_conversation_template
from generation_utils import (
    generate,
    get_model_size,
    load_model,
    merge_cache_config,
    setup_caches,
    decode_one_token,
    prefill
)
import os
from pathlib import Path
import string
import regex as re


def is_punc_id(text):
    # Define a regex pattern that matches any character that is not whitespace or punctuation
    pattern = rf"^[\s{re.escape(string.punctuation)}]*$"
    return bool(re.match(pattern, text))


def get_punc_ids(tokenizer):
    vocab = tokenizer.vocab
    result = []
    for token in vocab:
        if is_punc_id(token):
            result.append(vocab[token])
    return result


def get_special_ids(tokenizer):
    result = []
    vocab = tokenizer.vocab
    special_map = tokenizer.special_tokens_map
    for key in special_map:
        result.append([vocab[special_map[key]]])
    return result


def longchat_appy_chat_template(prompt, tokenizer):
    conv = get_conversation_template("vicuna")
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    encoded = tokenizer(prompt)
    return encoded

def mistral_apply_chat_template(prompt, tokenizer):
    prompt = f"[INST] {prompt} [/INST]"
    encoded = tokenizer(prompt)
    return encoded


def llama2_apply_chat_template(prompt, tokenizer):
    prompt = f"[INST] {prompt} [/INST]"
    encoded = tokenizer(prompt)
    return encoded


def llama3_apply_chat_template(prompt, tokenizer):
    messages = [{"role": "user", "content": f"{prompt}"}]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    encoded = tokenizer(prompt)
    return encoded


def llama_load_model_and_tokenizer(method, model_name_or_path, **kwargs):
    model_config = AutoConfig.from_pretrained(model_name_or_path)
    generate_kwarg = {}

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, fast_tokenizer=True, use_fast=True
    )

    # select apply_chat_template
    if any([x in model_name_or_path.lower() for x in ["llama-2", "llama2", "llama_2"]]):
        print("run llama2 model")
        apply_chat_template = llama2_apply_chat_template
        tokenizer.pad_token = "[PAD]"
        tokenizer.padding_side = "left"

    elif any(
        [
            x in model_name_or_path.lower()
            for x in ["llama-3.1", "llama3.1", "llama_3.1", "llama-3", "llama3", "llama_3"]
        ]
    ):
        print("run llama3 model")
        apply_chat_template = llama3_apply_chat_template
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"

    elif any([
            x in model_name_or_path.lower()
            for x in ["mistral"]
    ]):
        print("run mistral model")
        apply_chat_template = mistral_apply_chat_template
        tokenizer.pad_token = "[PAD]"
        tokenizer.padding_side = "left"

    elif any([
        x in model_name_or_path.lower() for x in ["longchat"]
    ]):
        print("run longchat model")
        apply_chat_template = longchat_appy_chat_template

    else:
        raise ValueError("Unsupported model name")

    generate_kwarg["special_ids"] = get_special_ids(tokenizer)
    generate_kwarg["punc_ids"] = get_punc_ids(tokenizer)

    checkpoint_path = Path(os.path.join(model_name_or_path, "model.pth"))
    model = load_model(checkpoint_path, kwargs["device"], precision=torch.bfloat16, use_tp=False)

    return model, tokenizer, generate_kwarg, apply_chat_template


def fastgen_generate(x, generate_kwarg, model, tokenizer):
    input_length = x["input_ids"].shape[1]

    output, _, _ = generate(model, x["input_ids"][0, :], prefill, decode_one_token,
                            max_new_tokens=generate_kwarg["max_new_tokens"],
                            eos_token_id=generate_kwarg["eos_token_id"],
                            attn_top_k=1.0,
                            feed_long_prompts=False)
    output = output.unsqueeze(0)
    output = output[:, input_length:]
    preds = tokenizer.batch_decode(output, skip_special_tokens=True)

    return preds
