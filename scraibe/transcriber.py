"""
Transcriber Module
------------------

This module provides the Transcriber class and the
OpenAIWhisperIPEXLLMTranscriber implementation which
leverages Intel IPEX-LLM on XPU with chunked audio
processing and segment-level timestamps.

Main Features:
  - Chunk audio into 30s segments for long-form transcription.
  - Process each chunk on XPU in bf16 or fp32 as configured.
  - Return segment list with start/end times for diarization merging.
  - Save transcripts to text files.

Constants:
  SAMPLE_RATE = 16000
  CHUNK_LENGTH = 30  # seconds
"""

import warnings
from abc import ABC, abstractmethod
from typing import TypeVar, Union, Optional, Dict, Any, List

import numpy as np
import torch
from torch import Tensor
from whisper.tokenizer import TO_LANGUAGE_CODE
from transformers import WhisperProcessor
from ipex_llm.transformers import AutoModelForSpeechSeq2Seq

SAMPLE_RATE = 16000
CHUNK_LENGTH = 30

ModelType = TypeVar("ModelType")


class Transcriber(ABC):
    """
    Base Transcriber interface.
    """
    def __init__(self, model_name: str, model_instance: ModelType, processor: WhisperProcessor):
        self.model_name = model_name
        self.model = model_instance
        self.processor = processor

    @abstractmethod
    def transcribe(self, audio: Union[str, np.ndarray, Tensor], **kwargs) -> Dict[str, Any]:
        ...

    @staticmethod
    def save_transcript(transcript: Dict[str, Any], save_path: str) -> None:
        text = transcript.get("text", "")
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Transcript saved to {save_path}")


class OpenAIWhisperIPEXLLMTranscriber(Transcriber):
    """
    Whisper with IPEX-LLM optimization on XPU.
    """

    @classmethod
    def load_model(
        cls,
        model_name: str = "openai/whisper-medium",
        device: str = "cpu",
        low_bit: str = "bf16",
        **kwargs
    ) -> "OpenAIWhisperIPEXLLMTranscriber":
        torch_device = torch.device(device)
        processor = WhisperProcessor.from_pretrained(model_name)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            load_in_low_bit=low_bit,
            optimize_model=True,
            trust_remote_code=True,
            **kwargs
        ).eval().to(torch_device)
        return cls(model_name, model, processor)

    def transcribe(
        self,
        audio: Union[str, np.ndarray, Tensor],
        language: str = "en",
        batch_size: int = 1,
        **generate_kwargs
    ) -> Dict[str, Any]:
        # load raw audio array
        if isinstance(audio, str):
            import soundfile as sf
            data, sr = sf.read(audio)
            if sr != SAMPLE_RATE:
                import librosa
                data = librosa.resample(data, orig_sr=sr, target_sr=SAMPLE_RATE)
            audio_arr = data.astype(np.float32)
        elif isinstance(audio, Tensor):
            audio_arr = audio.cpu().numpy().astype(np.float32)
        else:
            audio_arr = audio.astype(np.float32)

        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype

        # forced decoder prompt
        self.model.config.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=language, task="transcribe"
        )

        segments: List[Dict[str, Any]] = []
        texts: List[str] = []
        chunk_samples = SAMPLE_RATE * CHUNK_LENGTH
        total = len(audio_arr)
        n_chunks = int(np.ceil(total / chunk_samples))

        for idx in range(n_chunks):
            s = idx * chunk_samples
            e = min(s + chunk_samples, total)
            chunk = audio_arr[s:e]

            inputs = self.processor(
                chunk,
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
                truncation=False,
                padding="longest"
            )
            feats = inputs.input_features.to(device, dtype=dtype)

            with torch.no_grad():
                ids = self.model.generate(feats, **generate_kwargs)
                if device.type == "xpu":
                    torch.xpu.synchronize()
            text = self.processor.decode(ids[0], skip_special_tokens=True)

            segments.append({
                "id": idx, "start": s / SAMPLE_RATE, "end": e / SAMPLE_RATE, "text": text
            })
            texts.append(text)

        return {"text": " ".join(texts), "segments": segments, "language": language}


def load_transcriber(
    model_name: str = "openai/whisper-medium",
    device: str = "cpu",
    low_bit: str = "bf16",
    **kwargs
) -> Transcriber:
    return OpenAIWhisperIPEXLLMTranscriber.load_model(
        model_name=model_name, device=device, low_bit=low_bit, **kwargs
)
