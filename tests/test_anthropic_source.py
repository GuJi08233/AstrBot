import pytest
from anthropic.types import MessageDeltaUsage, Usage

from astrbot.core.provider.entities import LLMResponse, TokenUsage
from astrbot.core.provider.sources.anthropic_source import ProviderAnthropic


def _provider() -> ProviderAnthropic:
    return ProviderAnthropic.__new__(ProviderAnthropic)


def test_anthropic_extract_usage_counts_cache_creation_input():
    provider = _provider()

    usage = provider._extract_usage(
        Usage(
            input_tokens=10,
            cache_read_input_tokens=100,
            cache_creation_input_tokens=50,
            output_tokens=20,
        )
    )

    # Anthropic's input_tokens excludes cache writes, so cache_creation
    # must be folded into input_other to keep total input accurate.
    assert usage.input_other == 60
    assert usage.input_cached == 100
    assert usage.input == 160
    assert usage.output == 20


def test_anthropic_extract_usage_without_cache_breakpoints():
    provider = _provider()

    usage = provider._extract_usage(Usage(input_tokens=30, output_tokens=10))

    assert usage.input_other == 30
    assert usage.input_cached == 0
    assert usage.input == 30
    assert usage.output == 10


def test_anthropic_extract_usage_none_returns_empty():
    provider = _provider()

    assert provider._extract_usage(None) == TokenUsage()


def test_anthropic_update_usage_counts_cache_creation_input():
    provider = _provider()
    token_usage = TokenUsage(input_other=5, input_cached=0, output=0)

    provider._update_usage(
        token_usage,
        MessageDeltaUsage(
            input_tokens=10,
            cache_read_input_tokens=100,
            cache_creation_input_tokens=50,
            output_tokens=20,
        ),
    )

    assert token_usage.input_other == 60
    assert token_usage.input_cached == 100
    assert token_usage.input == 160
    assert token_usage.output == 20


def test_anthropic_update_usage_omitted_fields_are_preserved():
    provider = _provider()
    token_usage = TokenUsage(input_other=5, input_cached=0, output=0)

    # message_delta usage only carries output tokens in practice.
    provider._update_usage(token_usage, MessageDeltaUsage(output_tokens=7))

    assert token_usage.input_other == 5
    assert token_usage.input_cached == 0
    assert token_usage.output == 7


@pytest.mark.asyncio
async def test_anthropic_query_uses_stream_when_force_stream_enabled():
    provider = _provider()
    provider.provider_config = {"force_stream": True}
    captured = {}

    async def fake_query_stream(
        payloads, tools, *, request_max_retries=None, conversation_id=None
    ):
        captured["request_max_retries"] = request_max_retries
        captured["conversation_id"] = conversation_id
        yield LLMResponse("assistant", completion_text="Hel", is_chunk=True)
        yield LLMResponse("assistant", completion_text="Hello")

    provider._query_stream = fake_query_stream

    response = await provider._query(
        {"model": "claude-test", "messages": []},
        None,
        request_max_retries=2,
        conversation_id="conversation-1",
    )

    assert response.is_chunk is False
    assert response.completion_text == "Hello"
    assert captured == {
        "request_max_retries": 2,
        "conversation_id": "conversation-1",
    }
