import base64
import re
from pathlib import Path

from astrbot.core.utils.datetime_utils import generate_timestamp_id

from ..entities import ProviderType
from ..provider import TTSProvider
from ..register import register_provider_adapter
from .mimo_api_common import (
    DEFAULT_MIMO_API_BASE,
    DEFAULT_MIMO_TTS_MODEL,
    DEFAULT_MIMO_TTS_SEED_TEXT,
    DEFAULT_MIMO_TTS_VOICE,
    MiMoAPIError,
    build_api_url,
    build_headers,
    create_http_client,
    get_temp_dir,
    normalize_timeout,
)

# Voice clone samples must be mp3/wav and stay within 10 MB after base64.
VOICE_CLONE_MIME_TYPES = {
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
}
VOICE_CLONE_MAX_BASE64_BYTES = 10 * 1024 * 1024


@register_provider_adapter(
    "mimo_tts_api",
    "MiMo TTS API",
    provider_type=ProviderType.TEXT_TO_SPEECH,
)
class ProviderMiMoTTSAPI(TTSProvider):
    def __init__(
        self,
        provider_config: dict,
        provider_settings: dict,
    ) -> None:
        super().__init__(provider_config, provider_settings)
        self.chosen_api_key = provider_config.get("api_key", "")
        self.api_base = provider_config.get("api_base", DEFAULT_MIMO_API_BASE)
        self.proxy = provider_config.get("proxy", "")
        self.timeout = normalize_timeout(provider_config.get("timeout", 20))
        self.voice = provider_config.get("mimo-tts-voice", DEFAULT_MIMO_TTS_VOICE)
        self.audio_format = provider_config.get("mimo-tts-format", "wav")
        self.style_prompt = provider_config.get("mimo-tts-style-prompt", "")
        self.dialect = provider_config.get("mimo-tts-dialect", "")
        self.seed_text = provider_config.get(
            "mimo-tts-seed-text", DEFAULT_MIMO_TTS_SEED_TEXT
        )
        self.voice_clone_audio = provider_config.get("mimo-tts-voice-clone-audio", "")
        self._voice_clone_data_url: str | None = None
        self.set_model(provider_config.get("model", DEFAULT_MIMO_TTS_MODEL))
        self.client = create_http_client(self.timeout, self.proxy)

    def _build_user_prompt(self) -> str | None:
        seed_text = self.seed_text.strip()
        return seed_text or None

    def _build_style_prefix(self) -> str:
        style_parts: list[str] = []

        if self.style_prompt.strip():
            style_parts.append(self.style_prompt.strip())
        if self.dialect.strip():
            style_parts.append(self.dialect.strip())

        style_content = " ".join(style_parts).strip()
        if not style_content:
            return ""

        # MiMo V2.5 uses a leading "(style)" tag; singing must be the only tag
        # and accepts 唱歌 / sing / singing equivalently.
        style_words = set(re.split(r"\s+", style_content.lower()))
        if "唱歌" in style_content or style_words & {"sing", "singing"}:
            return "(唱歌)"

        return f"({style_content})"

    def _build_assistant_content(self, text: str) -> str:
        return f"{self._build_style_prefix()}{text}"

    def _is_voice_clone_model(self) -> bool:
        return "voiceclone" in (self.model_name or "").lower()

    def _is_voice_design_model(self) -> bool:
        return "voicedesign" in (self.model_name or "").lower()

    def _build_voice_clone_data_url(self) -> str:
        if self._voice_clone_data_url is not None:
            return self._voice_clone_data_url

        audio_path_text = str(self.voice_clone_audio or "").strip()
        if not audio_path_text:
            raise MiMoAPIError(
                "MiMo TTS voice clone 模型需要配置参考音频文件路径 "
                "(mimo-tts-voice-clone-audio)"
            )
        audio_path = Path(audio_path_text)
        if not audio_path.is_file():
            raise MiMoAPIError(f"MiMo TTS 参考音频文件不存在: {audio_path}")
        mime_type = VOICE_CLONE_MIME_TYPES.get(audio_path.suffix.lower())
        if not mime_type:
            raise MiMoAPIError(
                "MiMo TTS 参考音频仅支持 mp3 和 wav 格式: " + audio_path.name
            )
        encoded = base64.b64encode(audio_path.read_bytes()).decode("ascii")
        if len(encoded) > VOICE_CLONE_MAX_BASE64_BYTES:
            raise MiMoAPIError(
                "MiMo TTS 参考音频 Base64 编码后超过 10 MB 限制: " + audio_path.name
            )
        self._voice_clone_data_url = f"data:{mime_type};base64,{encoded}"
        return self._voice_clone_data_url

    def _resolve_voice(self) -> str | None:
        """Return the audio.voice value for the current model.

        Preset voices only apply to mimo-v2.5-tts; voicedesign derives the
        voice from the user prompt, voiceclone takes a base64 audio sample.
        """
        if self._is_voice_design_model():
            return None
        if self._is_voice_clone_model():
            return self._build_voice_clone_data_url()
        return self.voice

    def _build_payload(self, text: str) -> dict:
        messages: list[dict[str, str]] = []

        user_prompt = self._build_user_prompt()
        if user_prompt:
            messages.append(
                {
                    "role": "user",
                    "content": user_prompt,
                }
            )

        messages.append(
            {
                "role": "assistant",
                "content": self._build_assistant_content(text),
            }
        )

        audio_params = {"format": self.audio_format}
        voice = self._resolve_voice()
        if voice:
            audio_params["voice"] = voice

        return {
            "model": self.model_name,
            "messages": messages,
            "audio": audio_params,
        }

    async def get_audio(self, text: str) -> str:
        response = await self.client.post(
            build_api_url(self.api_base),
            headers=build_headers(self.chosen_api_key),
            json=self._build_payload(text),
        )

        try:
            response.raise_for_status()
        except Exception as exc:
            error_text = response.text[:1024]
            raise MiMoAPIError(
                f"MiMo TTS API request failed: HTTP {response.status_code}, response: {error_text}"
            ) from exc

        data = response.json()
        choices = data.get("choices") or []
        first_choice = choices[0] if choices else {}
        message = first_choice.get("message", {})
        audio_data = message.get("audio", {}).get("data")
        if not audio_data:
            raise MiMoAPIError(f"MiMo TTS API returned no audio payload: {data}")

        output_path = (
            get_temp_dir()
            / f"mimo_tts_api_{generate_timestamp_id()}.{self.audio_format}"
        )
        output_path.write_bytes(base64.b64decode(audio_data))
        return str(output_path)

    async def terminate(self):
        if self.client:
            await self.client.aclose()
