import pytest
import asyncio
from unittest.mock import AsyncMock, patch

# Assuming models are importable this way. Adjust if necessary.
from src.models.litellm import LiteLLMModel
# If other models are tested, import them here e.g.:
# from src.models.base import InferenceClientModel, OpenAIServerModel as BaseOpenAIServerModel, AmazonBedrockServerModel
# from src.models.openaillm import OpenAIServerModel as OpenAILLMOpenAIServerModel

from src.config import config
from src.models.base import ChatMessage # For creating dummy responses

# Store original config values to reset after tests
original_llm_caching_enabled = config.llm_caching_enabled
original_llm_cache_maxsize = config.llm_cache_maxsize

@pytest.fixture(autouse=True)
def reset_config_after_test():
    """Ensures config is reset after each test."""
    global original_llm_caching_enabled, original_llm_cache_maxsize
    # Store fresh original values before each test if they might change
    current_original_llm_caching_enabled = config.llm_caching_enabled
    current_original_llm_cache_maxsize = config.llm_cache_maxsize
    yield
    # Reset to the values that were present just before this specific test ran
    config.llm_caching_enabled = current_original_llm_caching_enabled
    config.llm_cache_maxsize = current_original_llm_cache_maxsize

# A dummy ChatMessage response for mocking
dummy_response = ChatMessage(role="assistant", content="Test response")

@pytest.mark.asyncio
async def test_cache_hit():
    """Test that a second identical call hits the cache."""
    config.llm_caching_enabled = True
    config.llm_cache_maxsize = 128

    # Path to the method that makes the actual external LLM call for LiteLLMModel
    # This is 'self.client.completion' where self.client is 'litellm' module itself
    mock_target_path = "litellm.completion" # For LiteLLMModel

    with patch(mock_target_path, new_callable=AsyncMock) as mock_llm_call:
        mock_llm_call.return_value = AsyncMock( # Mocking the response structure LiteLLM expects
            usage=AsyncMock(prompt_tokens=10, completion_tokens=10),
            choices=[AsyncMock(message=AsyncMock(role="assistant", content="Test response", tool_calls=None))]
        )

        # Instantiate the model. It will use the mocked 'litellm' module if client is created at call time
        # or if self.client is patched. For LiteLLMModel, self.client is 'litellm'.
        # So the patch on 'litellm.completion' should work.
        model = LiteLLMModel(model_id="test-cache-hit-model")

        call_params = {
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.7
        }

        # First call - should call the mocked LLM
        await model(**call_params)

        # Second identical call - should hit the cache
        await model(**call_params)

        mock_llm_call.assert_called_once()

@pytest.mark.asyncio
async def test_cache_miss_different_parameters():
    """Test that calls with different parameters miss the cache."""
    config.llm_caching_enabled = True
    config.llm_cache_maxsize = 128
    mock_target_path = "litellm.completion"

    with patch(mock_target_path, new_callable=AsyncMock) as mock_llm_call:
        mock_llm_call.return_value = AsyncMock(
            usage=AsyncMock(prompt_tokens=10, completion_tokens=10),
            choices=[AsyncMock(message=AsyncMock(role="assistant", content="Test response", tool_calls=None))]
        )
        model = LiteLLMModel(model_id="test-cache-miss-model")

        call_params1 = {"messages": [{"role": "user", "content": "Hello"}], "temperature": 0.7}
        call_params2 = {"messages": [{"role": "user", "content": "Hi"}], "temperature": 0.7} # Different message

        await model(**call_params1)
        await model(**call_params2)

        assert mock_llm_call.call_count == 2

@pytest.mark.asyncio
async def test_cache_disabled():
    """Test that caching is bypassed when disabled."""
    config.llm_caching_enabled = False
    config.llm_cache_maxsize = 128 # Maxsize is non-zero, but disabled by flag
    mock_target_path = "litellm.completion"

    with patch(mock_target_path, new_callable=AsyncMock) as mock_llm_call:
        mock_llm_call.return_value = AsyncMock(
            usage=AsyncMock(prompt_tokens=10, completion_tokens=10),
            choices=[AsyncMock(message=AsyncMock(role="assistant", content="Test response", tool_calls=None))]
        )
        model = LiteLLMModel(model_id="test-cache-disabled-model")

        call_params = {"messages": [{"role": "user", "content": "Hello"}], "temperature": 0.7}

        await model(**call_params)
        await model(**call_params) # Identical call

        assert mock_llm_call.call_count == 2

@pytest.mark.asyncio
async def test_cache_maxsize_eviction():
    """Test cache eviction when maxsize is reached."""
    config.llm_caching_enabled = True
    config.llm_cache_maxsize = 1 # Cache can only hold one item
    mock_target_path = "litellm.completion"

    with patch(mock_target_path, new_callable=AsyncMock) as mock_llm_call:
        mock_llm_call.return_value = AsyncMock(
            usage=AsyncMock(prompt_tokens=10, completion_tokens=10),
            choices=[AsyncMock(message=AsyncMock(role="assistant", content="Test response", tool_calls=None))]
        )
        model = LiteLLMModel(model_id="test-cache-maxsize-model")

        call_A_params = {"messages": [{"role": "user", "content": "Call A"}], "temperature": 0.1}
        call_B_params = {"messages": [{"role": "user", "content": "Call B"}], "temperature": 0.2}

        # Call A (1st time) - should call LLM, A is cached
        await model(**call_A_params)
        mock_llm_call.assert_called_once()

        # Call B - should call LLM, B is cached, A is evicted
        await model(**call_B_params)
        assert mock_llm_call.call_count == 2

        # Call A (2nd time) - should call LLM again because A was evicted by B
        await model(**call_A_params)
        assert mock_llm_call.call_count == 3

# To run these tests: pytest tests/test_llm_caching.py
# Ensure that the environment is set up for async tests with pytest (e.g., pytest-asyncio).
# If LiteLLMModel's __init__ or other parts require more setup (e.g. specific kwargs not part of config),
# the model instantiation in tests might need adjustment.
# The mock_target_path might need to be adjusted based on actual client usage in other models.
# For example, for OpenAI models, it might be 'openai.ChatCompletion.acreate' or similar if using the raw client,
# or the path to the specific client method if it's wrapped.
# Since LiteLLMModel directly uses `self.client.completion` where `self.client` is the `litellm` module,
# patching `litellm.completion` is correct for it.

# A note on mocking the LiteLLMModel:
# The LiteLLMModel's __init__ sets self.client = litellm.
# The __call__ method then calls self.client.completion(...).
# So, @patch('litellm.completion', ...) correctly intercepts the call.
# For other models, like OpenAIServerModel which might do `self.client = openai.OpenAI(...)`
# and then `self.client.chat.completions.create(...)`, the mock path would be
# different, e.g. @patch('openai.OpenAI().chat.completions.create', ...). This can get tricky
# if the client is instantiated within the method or per instance.
# A common pattern is to allow injecting a client instance for easier mocking,
# or to patch the specific method on the class if the client is an instance variable.
# For this test suite, focusing on LiteLLMModel with 'litellm.completion' demonstrates the principle.
# To test other models, the mock_target_path and potentially the mock_llm_call.return_value
# structure would need to be adapted.

# Example of how to adapt for another model, e.g., OpenAIServerModel from base.py
# assuming it's imported as BaseOpenAIServerModel
"""
@pytest.mark.asyncio
async def test_cache_hit_openai_base():
    config.llm_caching_enabled = True
    config.llm_cache_maxsize = 128

    # For OpenAIServerModel in base.py, client is OpenAI instance, call is client.chat.completions.create
    # We need to mock the 'create' method on the AsyncCompletions object, which is part of an AsyncOpenAI client.
    # This requires ensuring the client used by the model is the one we are patching.
    # One way is to patch at the module level where 'openai.AsyncOpenAI' is instantiated or used.
    # Or, if the client is an instance variable, patch it on the instance.

    # Let's assume OpenAIServerModel (base) uses a client like: self.client.chat.completions.create
    # We would patch 'openai.resources.chat.completions.AsyncCompletions.create' if that's the path
    # or more reliably, patch it on the instance if possible after creation.

    # This example assumes we can patch the specific instance's client method after model init.
    # However, it's often easier to patch at the module/class level where the client is obtained.

    model = BaseOpenAIServerModel(model_id="test-openai-cache", api_key="fake") # api_key often needed for client init

    # Mock the actual method that makes the API call on the model's client instance
    # This is more robust if client is per-instance.
    mock_llm_client_method = AsyncMock()
    mock_llm_client_method.return_value = AsyncMock( # Structure for OpenAI response
        id="chatcmpl-test",
        object="chat.completion",
        created=12345,
        model="gpt-test",
        choices=[AsyncMock(
            finish_reason="stop",
            index=0,
            message=AsyncMock(role="assistant", content="OpenAI Test response", tool_calls=None)
        )],
        usage=AsyncMock(prompt_tokens=10, completion_tokens=10)
    )

    # Patching the instance's client method directly.
    # This is tricky because self.client might be initialized in OpenAIServerModel's __init__
    # So we might need to patch 'openai.AsyncOpenAI' if it's created there.

    # A common way for patchable clients:
    # with patch.object(model.client.chat.completions, 'create', new=mock_llm_client_method) as patched_create:
    # This requires model.client to be already initialized and be the actual client obj.

    # For now, this test is commented out as it requires careful setup of OpenAI client mocking.
    # The LiteLLMModel tests serve as the primary examples.
    pass
"""

# Final global reset for safety, though fixture should handle per-test.
def teardown_module(module):
    global original_llm_caching_enabled, original_llm_cache_maxsize
    config.llm_caching_enabled = original_llm_caching_enabled
    config.llm_cache_maxsize = original_llm_cache_maxsize
