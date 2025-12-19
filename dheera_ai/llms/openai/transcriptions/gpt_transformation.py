from typing import List

from dheera_ai.llms.base_llm.audio_transcription.transformation import (
    AudioTranscriptionRequestData,
)
from dheera_ai.types.llms.openai import OpenAIAudioTranscriptionOptionalParams
from dheera_ai.types.utils import FileTypes

from .whisper_transformation import OpenAIWhisperAudioTranscriptionConfig


class OpenAIGPTAudioTranscriptionConfig(OpenAIWhisperAudioTranscriptionConfig):
    def get_supported_openai_params(
        self, model: str
    ) -> List[OpenAIAudioTranscriptionOptionalParams]:
        """
        Get the supported OpenAI params for the `gpt-4o-transcribe` models
        """
        return [
            "language",
            "prompt",
            "response_format",
            "temperature",
            "include",
        ]

    def transform_audio_transcription_request(
        self,
        model: str,
        audio_file: FileTypes,
        optional_params: dict,
        dheera_ai_params: dict,
    ) -> AudioTranscriptionRequestData:
        """
        Transform the audio transcription request
        """
        data = {"model": model, "file": audio_file, **optional_params}

        return AudioTranscriptionRequestData(
            data=data,
        )
