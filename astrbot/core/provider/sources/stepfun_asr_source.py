"""StepFun (阶跃星辰) speech-to-text provider using the HTTP + SSE ASR endpoint.

Works with both the open platform (``https://api.stepfun.com/v1``) and the
Step Plan subscription (``https://api.stepfun.com/step_plan/v1``) — the two
share identical request parameters, only the base path differs.
"""

import base64
import json
from collections.abc import AsyncIterable

import aiohttp

from astrbot.core.utils.media_utils import MediaResolver

from ..entities import ProviderType
from ..provider import STTProvider
from ..register import register_provider_adapter

DEFAULT_STEPFUN_API_BASE = "https://api.stepfun.com/step_plan/v1"
DEFAULT_STEPFUN_ASR_MODEL = "stepaudio-2.5-asr"
SSE_READ_BUFSIZE = 2**20


@register_provider_adapter(
    "stepfun_asr_sse",
    "StepFun ASR (SSE)",
    provider_type=ProviderType.SPEECH_TO_TEXT,
)
class ProviderStepFunASR(STTProvider):
    def __init__(
        self,
        provider_config: dict,
        provider_settings: dict,
    ) -> None:
        super().__init__(provider_config, provider_settings)
        self.chosen_api_key = provider_config.get("api_key", "")
        self.api_base = (
            provider_config.get("api_base") or DEFAULT_STEPFUN_API_BASE
        ).rstrip("/")
        self.language = str(provider_config.get("language") or "zh").strip()
        self.enable_itn = bool(provider_config.get("enable_itn", True))
        try:
            self.timeout = float(provider_config.get("timeout") or 60)
        except (TypeError, ValueError):
            self.timeout = 60.0
        self.proxy = provider_config.get("proxy", "") or None
        self.set_model(provider_config.get("model") or DEFAULT_STEPFUN_ASR_MODEL)

    def _build_payload(self, audio_b64: str) -> dict:
        transcription: dict = {
            "model": self.model_name,
            "enable_itn": self.enable_itn,
        }
        if self.language:
            transcription["language"] = self.language
        return {
            "audio": {
                "data": audio_b64,
                "input": {
                    "transcription": transcription,
                    "format": {"type": "wav"},
                },
            }
        }

    async def get_text(self, audio_url: str) -> str:
        async with MediaResolver(
            audio_url,
            media_type="audio",
            default_suffix=".wav",
        ).as_path(target_format="wav") as audio:
            audio_b64 = base64.b64encode(audio.read_bytes()).decode("ascii")

        url = f"{self.api_base}/audio/asr/sse"
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "Authorization": f"Bearer {self.chosen_api_key}",
        }
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(
            timeout=timeout, read_bufsize=SSE_READ_BUFSIZE
        ) as session:
            async with session.post(
                url,
                json=self._build_payload(audio_b64),
                headers=headers,
                proxy=self.proxy,
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise RuntimeError(
                        f"StepFun ASR request failed: HTTP {resp.status}, "
                        f"response: {body[:512]}"
                    )
                return await self._collect_sse_text(resp.content)

    async def _collect_sse_text(self, line_iter: AsyncIterable[bytes]) -> str:
        """Collect the transcription from a StepFun ASR SSE stream.

        The server emits ``transcript.text.delta`` events followed by a final
        ``transcript.text.done`` event carrying the full text; the done text
        wins when present, deltas are the fallback.
        """
        done_text: str | None = None
        delta_parts: list[str] = []
        async for raw_line in line_iter:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data_text = line[len("data:") :].strip()
            if not data_text or data_text == "[DONE]":
                continue
            try:
                event = json.loads(data_text)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            event_type = str(event.get("type") or "")
            if event_type == "transcript.text.delta":
                delta = event.get("delta")
                if not isinstance(delta, str):
                    text = event.get("text")
                    delta = text if isinstance(text, str) else ""
                delta_parts.append(delta)
            elif event_type == "transcript.text.done":
                text = event.get("text")
                if isinstance(text, str):
                    done_text = text
            elif "error" in event_type.lower() or event.get("error"):
                raise RuntimeError(
                    "StepFun ASR stream error: "
                    + json.dumps(event, ensure_ascii=False)[:512]
                )
        if done_text is not None:
            return done_text.strip()
        return "".join(delta_parts).strip()
