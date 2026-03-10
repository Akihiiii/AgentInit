import aiohttp
from typing import List, Union, Optional
from tenacity import retry, wait_random_exponential, stop_after_attempt, wait_fixed
from typing import Dict, Any
from dotenv import load_dotenv
import os
import logging
import random
import re
from openai import AsyncOpenAI
import async_timeout
from transformers import AutoTokenizer
import itertools
from AgentInit.llm.format import Message
from AgentInit.llm.price import cost_count, cost_count_llama3, cost_count_deepseek, cost_count_qwen
from AgentInit.llm.llm import LLM
from AgentInit.llm.llm_registry import LLMRegistry


load_dotenv()

MINE_BASE_URL = "http://10.1.1.158:11438/v1"
MINE_API_KEYS =  "123"


@retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(3))
async def achat(model: str, msg: List[Dict],):
    print("Achat")
    api_kwargs = dict(api_key = MINE_API_KEYS, base_url = MINE_BASE_URL)
    aclient = AsyncOpenAI(**api_kwargs)
    try:
        async with async_timeout.timeout(1000):
            completion = await aclient.chat.completions.create(model=model,messages=msg)
        response_message = completion.choices[0].message.content
        
        if isinstance(response_message, str):
            prompt = "".join([item['content'] for item in msg])
            cost_count(prompt, response_message, model)
            return response_message

    except Exception as e:
        raise RuntimeError(f"Failed to complete the async chat request: {e}")

# @retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(6))
@retry(wait=wait_random_exponential(max=30), stop=stop_after_attempt(2))
async def achat_deepseek(model: str, msg: List[Dict],):
    api_kwargs = dict(api_key=MINE_API_KEYS, base_url=MINE_BASE_URL)
    aclient = AsyncOpenAI(**api_kwargs)
    try:
        async with async_timeout.timeout(1000):
            completion = await aclient.chat.completions.create(model=model, messages=msg)
        response_message = completion.choices[0].message.content
        if isinstance(response_message, str):
            prompt_tokens = completion.usage.prompt_tokens
            completion_tokens = completion.usage.completion_tokens
            cost_count_qwen(prompt_tokens, completion_tokens, model)
            return response_message

    except Exception as e:
        raise RuntimeError(f"Failed to complete the async chat request: {e}")

@retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(3))
# @retry(wait=wait_random_exponential(max=10), stop=stop_after_attempt(10000))
async def achat_llama(model: str, msg: List[Dict]):
    print("Achat Llama")
    api_kwargs = dict(api_key = MINE_API_KEYS, base_url = MINE_BASE_URL)
    aclient = AsyncOpenAI(**api_kwargs)
    try:
        async with async_timeout.timeout(1000):
            # completion = await aclient.chat.completions.create(model=model,messages=msg)
            completion = await aclient.chat.completions.create(model=model,messages=msg,temperature=1.0,top_p=1.0)
        response_message = completion.choices[0].message.content
        # print(msg)
        
        if isinstance(response_message, str):
            prompt = "".join([item['content'] for item in msg])
            cost_count_llama3(prompt, response_message, model)
            return response_message

    except Exception as e:
        raise RuntimeError(f"Error in achat_llama, Failed to complete the async chat request: {e}")
        # logging.error(f"Error in achat_llama: {e}", exc_info=True)
        # raise


# 新增 achat_qwen
@retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(2))
async def achat_qwen(model: str, msg: List[Dict]):
    api_kwargs = dict(api_key=MINE_API_KEYS, base_url=MINE_BASE_URL)
    aclient = AsyncOpenAI(**api_kwargs)
    enable_thinking = False
    try:
        async with async_timeout.timeout(1000):
            completion = await aclient.chat.completions.create(
                model=model,
                messages=msg,
                extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}}
            )
        response_message = completion.choices[0].message.content

        if isinstance(response_message, str):
            if enable_thinking:
                # 1. 移除成对的 think 标签及其内容
                # 2. 移除可能存在的未闭合开标签及其后所有内容（防止截断导致的残留）
                response_message = re.sub(r'<think>.*?</think>', '', response_message, flags=re.DOTALL)
                response_message = re.sub(r'<think>.*', '', response_message, flags=re.DOTALL)
                response_message = response_message.strip()

            # Qwen 直接从 completion.usage 获取 token 数量计算 cost
            prompt_tokens = completion.usage.prompt_tokens
            completion_tokens = completion.usage.completion_tokens
            cost_count_qwen(prompt_tokens, completion_tokens, model)
            return response_message

    except Exception as e:
        raise RuntimeError(f"Error in achat_qwen, Failed to complete the async chat request: {e}")

@LLMRegistry.register('GPTChat')
class GPTChat(LLM):

    def __init__(self, model_name: str):
        self.model_name = model_name

    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        ) -> Union[List[str], str]:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS
        
        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]
        return await achat(self.model_name,messages)
    
    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        pass

@LLMRegistry.register('deepseek')
class DeepseekChat(LLM):

    def __init__(self, model_name: str):
        self.model_name = model_name

    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        ) -> Union[List[str], str]:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS
        
        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]
        return await achat_deepseek(self.model_name,messages)
    
    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        pass

@LLMRegistry.register('llama')
class LlamaChat(LLM):

    def __init__(self, model_name: str):
        self.model_name = model_name

    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        ) -> Union[List[str], str]:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS
        
        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]
        return await achat_llama(self.model_name,messages)
    
    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        pass


# 新增 QwenChat 的注册类
@LLMRegistry.register('qwen')
class QwenChat(LLM):

    def __init__(self, model_name: str):
        self.model_name = model_name

    async def agen(
            self,
            messages: List[Message],
            max_tokens: Optional[int] = None,
            temperature: Optional[float] = None,
            num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS

        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]
        return await achat_qwen(self.model_name, messages)

    def gen(
            self,
            messages: List[Message],
            max_tokens: Optional[int] = None,
            temperature: Optional[float] = None,
            num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        pass