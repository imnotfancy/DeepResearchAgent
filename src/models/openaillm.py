import warnings
from typing import Dict, List, Optional, Any, Tuple # Added Tuple
from copy import deepcopy

import time # New Import
from asyncache import cached, LRUCache

from src.config import config
from src.logger import logger # New Import
from src.models.base import (ApiModel,
                             ChatMessage,
                             tool_role_conversions,
                             MessageRole)
from src.models.message_manager import MessageManager

class OpenAIServerModel(ApiModel):
    """This model connects to an OpenAI-compatible API server.

    Parameters:
        model_id (`str`):
            The model identifier to use on the server (e.g. "gpt-3.5-turbo").
        api_base (`str`, *optional*):
            The base URL of the OpenAI-compatible API server.
        api_key (`str`, *optional*):
            The API key to use for authentication.
        organization (`str`, *optional*):
            The organization to use for the API request.
        project (`str`, *optional*):
            The project to use for the API request.
        client_kwargs (`dict[str, Any]`, *optional*):
            Additional keyword arguments to pass to the OpenAI client (like organization, project, max_retries etc.).
        custom_role_conversions (`dict[str, str]`, *optional*):
            Custom role conversion mapping to convert message roles in others.
            Useful for specific models that do not support specific message roles like "system".
        flatten_messages_as_text (`bool`, default `False`):
            Whether to flatten messages as text.
        **kwargs:
            Additional keyword arguments to pass to the OpenAI API.
    """

    def __init__(
        self,
        model_id: str,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        custom_role_conversions: dict[str, str] | None = None,
        flatten_messages_as_text: bool = False,
        http_client=None,
        **kwargs,
    ):
        self.model_id = model_id
        self.api_base = api_base
        self.api_key = api_key
        flatten_messages_as_text = (
            flatten_messages_as_text
            if flatten_messages_as_text is not None
            else model_id.startswith(("ollama", "groq", "cerebras"))
        )

        self.http_client = http_client

        self.message_manager = MessageManager(model_id=model_id)

        # Initialize cache based on global config
        self._llm_cache_enabled = config.llm_caching_enabled
        cache_maxsize = config.llm_cache_maxsize if self._llm_cache_enabled else 0
        self._llm_cache = LRUCache(maxsize=cache_maxsize)

        super().__init__(
            model_id=model_id,
            custom_role_conversions=custom_role_conversions,
            flatten_messages_as_text=flatten_messages_as_text,
            **kwargs,
        )

    def _generate_llm_cache_key(self, messages: List[Dict[str, Any]], **kwargs_call: Any) -> Tuple[Any, ...]:
        key_parts = []
        # From self attributes
        key_parts.append(self.model_id)
        key_parts.append(self.api_base)
        key_parts.append(self.flatten_messages_as_text)
        key_parts.append(tuple(sorted(self.custom_role_conversions.items())) if self.custom_role_conversions else None)

        # Include relevant items from self.kwargs (init-time kwargs)
        relevant_self_kwargs = {}
        for k, v in self.kwargs.items():
            if k in ['temperature', 'max_tokens', 'top_p', 'frequency_penalty', 'presence_penalty',
                       'organization', 'project']: # Added OpenAI specific init args
                if isinstance(v, list): v = tuple(v)
                elif isinstance(v, dict): v = tuple(sorted(v.items()))
                relevant_self_kwargs[k] = v
        key_parts.append(tuple(sorted(relevant_self_kwargs.items())))

        # Process messages (same logic as LiteLLMModel)
        processed_messages = []
        for msg in messages:
            role = msg.get('role')
            content = msg.get('content')
            processed_content_item = None
            if isinstance(content, list):
                processed_list_content = []
                for item in content:
                    if isinstance(item, dict):
                        processed_list_content.append(tuple(sorted(item.items())))
                    else:
                        processed_list_content.append(item)
                processed_content_item = tuple(processed_list_content)
            elif isinstance(content, dict):
                processed_content_item = tuple(sorted(content.items()))
            else:
                processed_content_item = content

            tool_calls = msg.get('tool_calls')
            processed_tool_calls = None
            if isinstance(tool_calls, list):
                processed_tc_list = []
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        func = tc.get('function', {})
                        args = func.get('arguments')
                        if isinstance(args, dict):
                            args = tuple(sorted(args.items()))
                        elif isinstance(args, str): # Arguments can be a string
                            pass # Keep as string
                        else: # If not dict or str, convert to str for safety
                            args = str(args)

                        processed_tc_list.append(
                            (tc.get('id'), tc.get('type'),
                             (func.get('name'), args))
                        )
                    else:
                        processed_tc_list.append(tc)
                processed_tool_calls = tuple(processed_tc_list)

            processed_messages.append((role, processed_content_item, processed_tool_calls))
        key_parts.append(tuple(processed_messages))

        # From __call__ kwargs (similar to LiteLLMModel)
        relevant_call_kwargs = {}
        for k, v in kwargs_call.items():
            if k in ['stop_sequences', 'grammar', 'tools_to_call_from',
                       'temperature', 'max_tokens', 'top_p', 'frequency_penalty', 'presence_penalty', # if overridable
                       'model', 'timeout']: # 'model' from kwargs_call could override self.model_id
                if isinstance(v, list):
                    if k == 'tools_to_call_from' and v:
                        v_processed = []
                        for tool in v: # Assuming get_tool_json_schema makes it hashable enough
                            if hasattr(tool, 'name') and hasattr(tool, 'description') and hasattr(tool, 'inputs'):
                                v_processed.append((tool.name, tool.description, tuple(sorted(tool.inputs.items()))))
                            else:
                                v_processed.append(str(tool))
                        v = tuple(v_processed)
                    else:
                        v = tuple(v)
                elif isinstance(v, dict):
                    v = tuple(sorted(v.items()))
                relevant_call_kwargs[k] = v
        key_parts.append(tuple(sorted(relevant_call_kwargs.items())))

        return tuple(key_parts)

    def create_client(self):

        if self.http_client:
            return self.http_client
        else:
            try:
                import openai
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError(
                    "Please install 'openai' extra to use OpenAIServerModel: `pip install 'smolagents[openai]'`"
                ) from e

            return openai.OpenAI(
                base_url=self.api_base,
                api_key=self.api_key
            )

    def _prepare_completion_kwargs(
            self,
            messages: List[Dict[str, str]],
            stop_sequences: Optional[List[str]] = None,
            grammar: Optional[str] = None,
            tools_to_call_from: Optional[List[Any]] = None,
            custom_role_conversions: dict[str, str] | None = None,
            convert_images_to_image_urls: bool = False,
            timeout: Optional[int] = 300,
            **kwargs,
    ) -> Dict:
        """
        Prepare parameters required for model invocation, handling parameter priorities.

        Parameter priority from high to low:
        1. Explicitly passed kwargs
        2. Specific parameters (stop_sequences, grammar, etc.)
        3. Default values in self.kwargs
        """
        # Clean and standardize the message list
        messages = self.message_manager.get_clean_message_list(
            messages,
            role_conversions=custom_role_conversions or tool_role_conversions,
            convert_images_to_image_urls=convert_images_to_image_urls,
            flatten_messages_as_text=self.flatten_messages_as_text,
        )

        # Use self.kwargs as the base configuration
        completion_kwargs = {
            **self.kwargs,
            "messages": messages,
        }
        # Handle timeout
        if timeout is not None:
            completion_kwargs["timeout"] = timeout

        # Handle specific parameters
        if stop_sequences is not None:
            completion_kwargs["stop"] = stop_sequences
        if grammar is not None:
            completion_kwargs["grammar"] = grammar

        # Handle tools parameter
        if tools_to_call_from:
            completion_kwargs.update(
                {
                    "tools": [self.message_manager.get_tool_json_schema(tool,
                                   model_id=self.model_id) for tool in tools_to_call_from],
                    "tool_choice": "required",
                }
            )

        # Finally, use the passed-in kwargs to override all settings
        completion_kwargs.update(kwargs)

        completion_kwargs = self.message_manager.get_clean_completion_kwargs(completion_kwargs)

        return completion_kwargs

    async def __call__(
        self,
        messages: List[Dict[str, str]],
        stop_sequences: Optional[List[str]] = None,
        grammar: Optional[str] = None,
        tools_to_call_from: Optional[List[Any]] = None,
        **kwargs,
    ) -> ChatMessage:

        # The lambda for OpenAIServerModel.__call__ (in openaillm.py) is:
        # key=lambda self, messages, **kwargs_call: self._generate_llm_cache_key(messages, **kwargs_call)
        # So, we need to pass all of __call__'s kwargs (excluding messages) into kwargs_call for _generate_llm_cache_key.
        kwargs_for_key_func = {"stop_sequences": stop_sequences, "grammar": grammar, "tools_to_call_from": tools_to_call_from, **kwargs}

        was_miss = False
        if self._llm_cache_enabled:
            log_cache_key = self._generate_llm_cache_key(messages, **kwargs_for_key_func)
            if self._llm_cache.get(log_cache_key, "CACHE_MISS_SENTINEL") != "CACHE_MISS_SENTINEL":
                logger.debug(f"LLM call CACHE HIT for {self.model_id} (key: {str(log_cache_key)[:100]}...)")
            else:
                logger.debug(f"LLM call CACHE MISS for {self.model_id} (key: {str(log_cache_key)[:100]}...)")
                was_miss = True
        else:
            was_miss = True # Always a 'miss' if caching is disabled

        start_time = time.time()

        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            grammar=grammar,
            tools_to_call_from=tools_to_call_from,
            model=self.model_id,
            custom_role_conversions=self.custom_role_conversions,
            convert_images_to_image_urls=True,
            **kwargs,
        )

        try:
            response = self.client.chat.completions.create(**completion_kwargs)
        finally:
            if was_miss:
                duration = time.time() - start_time
                logger.debug(f"LLM API call for {self.model_id} took {duration:.2f}s")

        self.last_input_token_count = response.usage.prompt_tokens # Ensure these are set even on cache hit if they were part of cached object
        self.last_output_token_count = response.usage.completion_tokens # Or handle this in the caching of ChatMessage itself

        first_message = ChatMessage.from_dict(
            response.choices[0].message.model_dump(include={"role", "content", "tool_calls"}),
            raw=response,
        )
        return self.postprocess_message(first_message, tools_to_call_from)