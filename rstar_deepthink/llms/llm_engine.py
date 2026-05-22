# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os, sys
import torch
from dataclasses import dataclass
from rstar_deepthink.llms.rm import *
from transformers import AutoConfig, AutoTokenizer
from vllm import LLM, SamplingParams


@dataclass
class ApiSamplingParams:
    temperature: float
    top_p: float
    max_tokens: int
    n: int
    stop: list | None


class OpenAIResponsesEngine:
    def __init__(self, model: str):
        from openai import OpenAI

        self.model = model
        self.client = OpenAI()

    def generate(self, prompt: str, sampling_params: ApiSamplingParams) -> str:
        kwargs = {
            "model": self.model,
            "input": prompt,
            "max_output_tokens": sampling_params.max_tokens,
        }
        if sampling_params.temperature is not None:
            kwargs["temperature"] = sampling_params.temperature
        if sampling_params.top_p is not None:
            kwargs["top_p"] = sampling_params.top_p
        response = self.client.responses.create(**kwargs)
        if getattr(response, "output_text", None):
            return response.output_text
        chunks = []
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                text = getattr(content, "text", None)
                if text:
                    chunks.append(text)
        return "".join(chunks)

def llm_init(config):
    llm = LLM(
        model=config.model_dir, 
        tensor_parallel_size=config.tp, 
        trust_remote_code=True,
        seed=config.seed if config.seed else 0,
        swap_space=config.swap_space,
        max_model_len=config.max_model_len,
        gpu_memory_utilization=config.llm_gpu_memory_utilization,
        enforce_eager=True,
        distributed_executor_backend='ray' if config.tp > 1 else None,
        dtype="bfloat16",
    )
    sampling_params = SamplingParams(
        temperature=config.temperature,
        top_k=config.top_k,
        top_p=config.top_p,
        best_of=config.best_of,
        max_tokens=config.max_tokens, 
        n=config.n_generate_sample,
        stop=config.stop,
        skip_special_tokens=False,
        seed=config.seed if config.temperature == 0 else None, # vllm0.6.6.post1 
    )
    return llm, sampling_params

def llm_engine(config):
    if getattr(config, "llm_backend", "vllm") == "openai_api":
        sampling_params = ApiSamplingParams(
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
            n=config.n_generate_sample,
            stop=config.stop,
        )
        return OpenAIResponsesEngine(config.api_model), sampling_params
    llm, sampling_params = llm_init(config)
    return llm, sampling_params

def rm_engine(config):
    if config.need_value_func:
        prm_model = LLM(
            model=config.reward_model_dir, 
            task="reward",
            tensor_parallel_size=1, 
            trust_remote_code=True,
            max_model_len=config.max_model_len,
            enforce_eager=True,
            swap_space=0,
            gpu_memory_utilization=0.98 - config.llm_gpu_memory_utilization, # for qwen 7b, rm need 15G memory
        )
        
        v_head_state = torch.load(os.path.join(config.reward_model_dir, "value_head.bin"), weights_only=True)
        v_state = {}
        for name, param in v_head_state.items():
            v_state[name.replace("v_head.", "")] = param
        model_config = AutoConfig.from_pretrained(config.reward_model_dir, trust_remote_code=True, use_cache = False)
        v_head = ValueHead(model_config)
        v_head.load_state_dict(v_state)
        v_head.eval()
        tokenizer = AutoTokenizer.from_pretrained(config.reward_model_dir, trust_remote_code=True, use_cache = False, split_special_tokens=False,)
        return prm_model, v_head, tokenizer
    else:
        return None, None, None
