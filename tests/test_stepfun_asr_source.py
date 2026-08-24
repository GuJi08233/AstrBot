import json

import pytest

from astrbot.core.provider.sources.stepfun_asr_source import ProviderStepFunASR


def _make_provider(overrides: dict | None = None) -> ProviderStepFunASR:
    provider_config = {
        "id": "test-stepfun-asr",
        "type": "stepfun_asr_sse",
        "model": "stepaudio-2.5-asr",
        "api_key": "test-key",
    }
    if overrides:
        provider_config.update(overrides)
    return ProviderStepFunASR(provider_config=provider_config, provider_settings={})


async def _sse_lines(*events: object):
    for event in events:
        if isinstance(event, bytes):
            yield event
        else:
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n".encode()


def test_defaults_target_step_plan_endpoint():
    provider = _make_provider()
    assert provider.api_base == "https://api.stepfun.com/step_plan/v1"
    assert provider.model_name == "stepaudio-2.5-asr"
    assert provider.language == "zh"
    assert provider.enable_itn is True


def test_open_platform_base_url_is_normalized():
    provider = _make_provider({"api_base": "https://api.stepfun.com/v1/"})
    assert provider.api_base == "https://api.stepfun.com/v1"


def test_payload_carries_transcription_and_wav_format():
    provider = _make_provider({"language": "en", "enable_itn": False})
    payload = provider._build_payload("QUJD")
    assert payload["audio"]["data"] == "QUJD"
    transcription = payload["audio"]["input"]["transcription"]
    assert transcription == {
        "model": "stepaudio-2.5-asr",
        "enable_itn": False,
        "language": "en",
    }
    assert payload["audio"]["input"]["format"] == {"type": "wav"}


@pytest.mark.asyncio
async def test_done_event_text_wins_over_deltas():
    provider = _make_provider()
    text = await provider._collect_sse_text(
        _sse_lines(
            {"type": "transcript.text.delta", "delta": "你好"},
            {"type": "transcript.text.delta", "delta": "世界"},
            {"type": "transcript.text.done", "text": "你好，世界。"},
        )
    )
    assert text == "你好，世界。"


@pytest.mark.asyncio
async def test_deltas_are_joined_when_done_missing():
    provider = _make_provider()
    text = await provider._collect_sse_text(
        _sse_lines(
            {"type": "transcript.text.delta", "delta": "你好"},
            {"type": "transcript.text.delta", "delta": "世界"},
        )
    )
    assert text == "你好世界"


@pytest.mark.asyncio
async def test_non_data_lines_and_done_marker_are_ignored():
    provider = _make_provider()
    text = await provider._collect_sse_text(
        _sse_lines(
            b": keep-alive comment\n",
            b"event: message\n",
            b"\n",
            {"type": "transcript.text.done", "text": "ok"},
            b"data: [DONE]\n",
        )
    )
    assert text == "ok"


@pytest.mark.asyncio
async def test_error_event_raises():
    provider = _make_provider()
    with pytest.raises(RuntimeError, match="StepFun ASR stream error"):
        await provider._collect_sse_text(
            _sse_lines({"type": "error", "error": {"message": "invalid audio"}})
        )
