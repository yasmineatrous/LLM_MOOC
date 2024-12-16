from termcolor import colored  
import os  
from openai import AsyncOpenAI, OpenAI
import time
import json
import asyncio

from anthropic import AsyncAnthropic

class UnifiedApis:
    def __init__(self,
                 name="Unified Apis",
                 api_key=None,
                 max_history_words=10000,
                 max_words_per_message=None,
                 json_mode=False,
                 stream=True,
                 use_async=False,
                 max_retry=10,
                 provider="anthropic",
                 model=None,
                 should_print_init=True,
                 print_color="green"):
        self.provider = provider.lower()
        if self.provider == "openai":
            self.model = model or "gpt-4o"
        elif self.provider == "anthropic":
            self.model = model or "claude-3-5-sonnet-20240620"
        elif self.provider == "openrouter":
            self.model = model or "google/gemini-pro-1.5"

        self.name = name
        self.api_key = api_key or self._get_api_key()
        self.history = []

        self.max_history_words = max_history_words
        self.max_words_per_message = max_words_per_message

        self.json_mode = json_mode
        self.stream = stream

        self.use_async = use_async
        self.max_retry = max_retry

        self.print_color = print_color

        self.system_message = "You are a helpful assistant."
        if self.provider == "openai" and self.json_mode:
            self.system_message += " Please return your response in JSON unless user has specified a system message."

        self._initialize_client()

        if should_print_init:
            print(colored(f"({self.name}) initialized with provider={self.provider}, model={self.model}, json_mode={self.json_mode}, stream={self.stream}, "
                          f"use_async={self.use_async}, max_history_words={self.max_history_words}, max_words_per_message={self.max_words_per_message}", self.print_color))

    def _get_api_key(self):
        if self.provider == "openai":
            return os.getenv("OPENAI_API_KEY") or "YOUR_OPENAI_KEY_HERE"
        elif self.provider == "anthropic":
            return os.getenv("ANTHROPIC_API_KEY") or "YOUR_ANTHROPIC_KEY_HERE"
        elif self.provider == "openrouter":
            return os.getenv("OPENROUTER_API_KEY") or "YOUR_OPENROUTER_KEY_HERE"
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _initialize_client(self):
        if self.provider == "openai" and self.use_async:
            self.client = AsyncOpenAI(api_key=self.api_key)
        elif self.provider == "anthropic" and self.use_async:
            self.client = AsyncAnthropic(api_key=self.api_key)
        elif self.provider == "openrouter" and self.use_async:
            self.client = AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.api_key
            )
        elif self.provider == "openai" and not self.use_async:
            self.client = OpenAI(api_key=self.api_key)
        #elif self.provider == "anthropic" and not self.use_async:
            #self.client = Anthropic(api_key=self.api_key)
        elif self.provider == "openrouter" and not self.use_async:
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.api_key
            )

    def set_system_message(self, message=None):
        self.system_message = message or "You are a helpful assistant."
        if self.provider == "openai" and self.json_mode and "json" not in message.lower():
            self.system_message += " Please return your response in JSON unless user has specified a system message."

    async def set_system_message_async(self, message=None):
        self.set_system_message(message)

    def add_message(self, role, content):
        if role == "user" and self.max_words_per_message:
            content += f" please use {self.max_words_per_message} words or less"
        self.history.append({"role": role, "content": str(content)})

    async def add_message_async(self, role, content):
        self.add_message(role, content)

    def print_history_length(self):
        history_length = sum(len(str(message["content"]).split()) for message in self.history)
        print(f"Current history length is {history_length} words")

    async def print_history_length_async(self):
        self.print_history_length()

    def clear_history(self):
        self.history.clear()

    def chat(self, user_input, **kwargs):
        self.add_message("user", user_input)
        return self.get_response(**kwargs)
    
    async def chat_async(self, user_input, **kwargs):
        await self.add_message_async("user", user_input)
        return await self.get_response_async(**kwargs)
    def trim_history(self):
        words_count = sum(len(str(message["content"]).split()) for message in self.history if message["role"] != "system")
        while words_count > self.max_history_words and len(self.history) > 1:
            words_count -= len(self.history[0]["content"].split())
            self.history.pop(0)

    async def trim_history_async(self):
        self.trim_history()
    
    def get_response(self, color=None, should_print=True, **kwargs):
        if color is None:
            color = self.print_color

        max_tokens = kwargs.pop('max_tokens', 4000)
        anthropic_max_tokens = kwargs.pop('max_tokens', 8192)

        retries = 0
        while retries < self.max_retry:
            try:
                if self.provider == "openai":
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "system", "content": self.system_message}] + self.history,
                        stream=self.stream,
                        max_tokens=max_tokens,
                        response_format={"type": "json_object"}if self.json_mode else None,
                        **kwargs
                    )
                elif self.provider == "anthropic":
                    response = self.client.completions.create(
                        model=self.model,
                        system=self.system_message,
                        messages=self.history,
                        stream=self.stream,
                        max_tokens=anthropic_max_tokens,
                        extra_headers={"anthropic-beta": "max-tokens-3-5-sonnet-2024-07-15"},
                        **kwargs
                    )
                elif self.provider == "openrouter":
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "system", "content": self.system_message}] + self.history,
                        stream=self.stream,
                        max_tokens=max_tokens,
                        **kwargs
                )

                if self.stream:
                    assistant_response = ""
                    for chunk in response:
                        if self.provider == "openai" or self.provider == "openrouter":
                            if chunk.choices[0].delta.content:
                                content = chunk.choices[0].delta.content
                            else:
                                content = None
                        elif self.provider == "anthropic":
                            content = chunk.delta.text if chunk.type == "content_block_delta" else None

                        if content:
                            if should_print:
                                    print(colored(content, color), end="", flush=True)
                            assistant_response += content
                    print()

                else:
                    if self.provider == "openai" or self.provider == "openrouter":
                        assistant_response = response.choices[0].message.content
                    elif self.provider == "anthropic":
                        assistant_response = response.content[0].text

                if self.json_mode and self.provider == "openai":
                    assistant_response = json.loads(assistant_response)

                self.add_message("assistant", str(assistant_response))

                self.trim_history()
                return assistant_response

            except Exception as e:
                    print("Error:", e)
                    retries += 1
                    time.sleep(1)

        raise Exception("Max retries reached")
    async def get_response_async(self, color=None, should_print=True, **kwargs):
        if color is None:
            color = self.print_color

        max_tokens = kwargs.pop('max_tokens', 4000)
        anthropic_max_tokens = kwargs.pop('max_tokens', 8192)

        retries = 0
        while retries < self.max_retry:
            try:
                if self.provider == "openai":
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "system", "content": self.system_message}] + self.history,
                        stream=self.stream,
                        max_tokens=max_tokens,
                        response_format={"type": "json_object"} if self.json_mode else None,
                        **kwargs
                    )
                elif self.provider == "anthropic":
                    response = await self.client.completions.create(
                        model=self.model,
                        system=self.system_message,
                        messages=self.history,
                        stream=self.stream,
                        max_tokens=anthropic_max_tokens,
                        extra_headers={"anthropic-beta": "max-tokens-3-5-sonnet-2024-07-15"},
                        **kwargs
                    )
                elif self.provider == "openrouter":
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "system", "content": self.system_message}] + self.history,
                        stream=self.stream,
                        max_tokens=max_tokens,
                        **kwargs
                    )

                if self.stream:
                    assistant_response = ""
                    async for chunk in response:
                        if self.provider == "openai" or self.provider == "openrouter":
                            if chunk.choices[0].delta.content:
                                content = chunk.choices[0].delta.content
                            else:
                                content = None
                        elif self.provider == "anthropic":
                            content = chunk.delta.text if chunk.type == "content_block_delta" else None

                        if content:
                            if should_print:
                                print(colored(content, color), end="", flush=True)
                            assistant_response += content
                    print()

                else:
                    if self.provider == "openai" or self.provider == "openrouter":
                        assistant_response = response.choices[0].message.content
                    elif self.provider == "anthropic":
                        assistant_response = response.content[0].text

                if self.json_mode and self.provider == "openai":
                    assistant_response = json.loads(assistant_response)

                self.add_message("assistant", str(assistant_response))

                self.trim_history()
                return assistant_response

            except Exception as e:
                print("Error:", e)
                retries += 1
                await asyncio.sleep(1)

        raise Exception("Max retries reached")