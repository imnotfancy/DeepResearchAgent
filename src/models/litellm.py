import warnings
from typing import Dict, List, Optional, Any, Tuple

import time # New Import
from asyncache import cached, LRUCache

from src.config import config
from src.logger import logger # New Import
from src.models.base import (ApiModel,
                             ChatMessage,
                             tool_role_conversions,
                             )
from src.models.message_manager import (
    MessageManager
)

class LiteLLMModel(ApiModel):
    """Model to use [LiteLLM Python SDK](https://docs.litellm.ai/docs/#litellm-python-sdk) to access hundreds of LLMs.

    Parameters:
        model_id (`str`):
            The model identifier to use on the server (e.g. "gpt-3.5-turbo").
        api_base (`str`, *optional*):
            The base URL of the provider API to call the model.
        api_key (`str`, *optional*):
            The API key to use for authentication.
        custom_role_conversions (`dict[str, str]`, *optional*):
            Custom role conversion mapping to convert message roles in others.
            Useful for specific models that do not support specific message roles like "system".
        flatten_messages_as_text (`bool`, *optional*): Whether to flatten messages as text.
            Defaults to `True` for models that start with "ollama", "groq", "cerebras".
        **kwargs:
            Additional keyword arguments to pass to the OpenAI API.
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        api_base=None,
        api_key=None,
        custom_role_conversions: dict[str, str] | None = None,
        flatten_messages_as_text: bool | None = None,
        http_client=None,
        **kwargs,
    ):
        if not model_id:
            warnings.warn(
                "The 'model_id' parameter will be required in version 2.0.0. "
                "Please update your code to pass this parameter to avoid future errors. "
                "For now, it defaults to 'anthropic/claude-3-5-sonnet-20240620'.",
                FutureWarning,
            )
            model_id = "anthropic/claude-3-5-sonnet-20240620"
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
        # These are settings that define the model's behavior fixed at init.
        # Example: temperature, max_tokens if not passed in __call__ typically
        relevant_self_kwargs = {}
        for k, v in self.kwargs.items():
            if k in ['temperature', 'max_tokens', 'top_p', 'frequency_penalty', 'presence_penalty']: # Add others as needed
                if isinstance(v, list): v = tuple(v)
                elif isinstance(v, dict): v = tuple(sorted(v.items()))
                relevant_self_kwargs[k] = v
        key_parts.append(tuple(sorted(relevant_self_kwargs.items())))

        # Process messages
        processed_messages = []
        for msg in messages:
            role = msg.get('role')
            content = msg.get('content')
            processed_content_item = None
            if isinstance(content, list):  # Handle list content (e.g., multimodal)
                processed_list_content = []
                for item in content:
                    if isinstance(item, dict):
                        # Sort dict items for stable hashing
                        processed_list_content.append(tuple(sorted(item.items())))
                    else:
                        processed_list_content.append(item) # Assumed hashable
                processed_content_item = tuple(processed_list_content)
            elif isinstance(content, dict):
                processed_content_item = tuple(sorted(content.items()))
            else: # content is str or None
                processed_content_item = content

            tool_calls = msg.get('tool_calls')
            processed_tool_calls = None
            if isinstance(tool_calls, list):
                processed_tc_list = []
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        # Assuming tool_call dicts have 'id', 'type', and 'function'
                        # And 'function' has 'name' and 'arguments' (which could be str or dict)
                        func = tc.get('function', {})
                        args = func.get('arguments')
                        if isinstance(args, dict):
                            args = tuple(sorted(args.items()))

                        processed_tc_list.append(
                            (tc.get('id'), tc.get('type'),
                             (func.get('name'), args)
                            )
                        )
                    else: # Should not happen based on ChatMessage structure
                        processed_tc_list.append(tc)
                processed_tool_calls = tuple(processed_tc_list)

            processed_messages.append((role, processed_content_item, processed_tool_calls))
        key_parts.append(tuple(processed_messages))

        # From __call__ kwargs
        relevant_call_kwargs = {}
        for k, v in kwargs_call.items():
            # Include kwargs that affect LLM response and are not already covered by self.kwargs
            # model, api_base, api_key, custom_role_conversions are passed to _prepare_completion_kwargs
            # but usually derive from self or are handled.
            # We care about what _actually_ changes the call to the LLM.
            if k in ['stop_sequences', 'grammar', 'tools_to_call_from',
                       'temperature', 'max_tokens', 'top_p', 'frequency_penalty', 'presence_penalty', # if overridable in call
                       'model', 'timeout']: # 'model' from kwargs_call could override self.model_id for the call
                if isinstance(v, list):
                    # Special handling for tools_to_call_from if it contains unhashable tool objects
                    if k == 'tools_to_call_from' and v:
                        v_processed = []
                        for tool in v:
                            if hasattr(tool, 'name') and hasattr(tool, 'description') and hasattr(tool, 'inputs'):
                                # Assuming inputs is a dict that can be sorted
                                v_processed.append((tool.name, tool.description, tuple(sorted(tool.inputs.items()))))
                            else: # Fallback for unhandled tool types
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
        """Create the LiteLLM client."""
        try:
            import litellm
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Please install 'litellm' extra to use LiteLLMModel: `pip install 'smolagents[litellm]'`"
            ) from e

        return litellm

    def _prepare_completion_kwargs(
            self,
            messages: List[Dict[str, str]],
            stop_sequences: Optional[List[str]] = None,
            grammar: Optional[str] = None,
            tools_to_call_from: Optional[List[Any]] = None,
            custom_role_conversions: Optional[Dict[str, str]] = None,
            convert_images_to_image_urls: bool = False,
            http_client=None,
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
        
        if http_client:
            completion_kwargs['client'] = http_client

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

        # Reconstruct kwargs for logging key generation, matching the @cached key=lambda
        # The lambda for LiteLLMModel.__call__ is:
        # key=lambda self, messages, **kwargs_call: self._generate_llm_cache_key(messages, **kwargs_call)
        # So, we need to pass all of __call__'s kwargs (excluding messages, which is a named arg)
        # into kwargs_call for _generate_llm_cache_key.

        kwargs_for_key_func = {"stop_sequences": stop_sequences, "grammar": grammar, "tools_to_call_from": tools_to_call_from, **kwargs}

        if self._llm_cache_enabled:
            log_cache_key = self._generate_llm_cache_key(messages, **kwargs_for_key_func)
            # Check self._llm_cache directly, not log_cache_key in self._llm_cache.cache for asyncache
            if self._llm_cache.get(log_cache_key, "CACHE_MISS_SENTINEL") != "CACHE_MISS_SENTINEL":
                 logger.debug(f"LLM call CACHE HIT for {self.model_id} (key: {str(log_cache_key)[:100]}...)")
            else:
                logger.debug(f"LLM call CACHE MISS for {self.model_id} (key: {str(log_cache_key)[:100]}...)")

        start_time = time.time()

        # The original __call__ method is now decorated
        completion_kwargs = self._prepare_completion_kwargs( # Note: self.http_client is passed here
            messages=messages, # This is fine, _prepare_completion_kwargs needs the original messages
            stop_sequences=stop_sequences, # Pass along to be included if not None
            grammar=grammar,
            tools_to_call_from=tools_to_call_from,
            model=self.model_id,
            api_base=self.api_base, # Already part of key via self.api_base
            api_key=self.api_key, # Not part of key
            http_client=self.http_client, # http_client might affect the call, how to make it hashable? For now, assume not part of key or self.http_client is stable.
            convert_images_to_image_urls=True, # Hardcoded, so not varying
            custom_role_conversions=self.custom_role_conversions,
            **kwargs, # Pass all other original kwargs from __call__
        )

        try:
            response = self.client.completion(**completion_kwargs)
        finally:
            # Log duration if it was a cache miss path (actual call was made)
            # This logging will occur regardless of success or failure of the API call itself.
            if not self._llm_cache_enabled or \
               (self._llm_cache_enabled and self._llm_cache.get(self._generate_llm_cache_key(messages, **kwargs_for_key_func), "CACHE_MISS_SENTINEL") == "CACHE_MISS_SENTINEL"): # Re-check, or assume miss if no hit log
                duration = time.time() - start_time
                logger.debug(f"LLM API call for {self.model_id} took {duration:.2f}s")


        self.last_input_token_count = response.usage.prompt_tokens
        self.last_output_token_count = response.usage.completion_tokens
        first_message = ChatMessage.from_dict(
            response.choices[0].message.model_dump(include={"role", "content", "tool_calls"}),
            raw=response,
        )
        return self.postprocess_message(first_message, tools_to_call_from)