"""
Transcriber Module
------------------

This module provides the Transcriber class, a comprehensive tool for working with Whisper models.
The Transcriber class offers functionalities such as loading different Whisper models, transcribing audio files,
and saving transcriptions to text files. It acts as an interface between various Whisper models and the user,
simplifying the process of audio transcription.

Main Features:
    - Loading different sizes and versions of Whisper models (OpenAI, FasterWhisper).
    - Support for IPEX-LLM optimization for OpenAI Whisper models on Intel XPUs.
    - Transcribing audio in various formats including str, Tensor, and ndarray.
    - Saving the transcriptions to the specified paths.
    - Adaptable to various language specifications.
    - Options to control the verbosity of the transcription process.
    
Constants:
    WHISPER_DEFAULT_PATH: Default path for downloading and loading Whisper models.

Usage:
    >>> from scraibe.transcriber import load_transcriber # Assuming part of scraibe package
    >>> # For IPEX-LLM optimized OpenAI Whisper on XPU
    >>> transcriber_ipex = load_transcriber(
    ...     model_name="tiny",
    ...     whisper_type="openai-ipex-llm",
    ...     device="xpu",
    ...     low_bit="bf16" # or "sym_int4" etc.
    ... )
    >>> result_ipex = transcriber_ipex.transcribe(audio="path/to/audio.wav")
    >>> print(result_ipex["text"])
    >>>
    >>> # For FasterWhisper
    >>> transcriber_faster = load_transcriber(
    ...     model_name="tiny",
    ...     whisper_type="faster-whisper",
    ...     device="cpu", # or "cuda"
    ...     compute_type="int8"
    ... )
    >>> result_faster = transcriber_faster.transcribe(audio="path/to/audio.wav")
    >>> print(result_faster["text"])
"""

from abc import ABC, abstractmethod
from inspect import signature
from typing import TypeVar, Union, Optional, Dict, Any

import torch
from torch import Tensor, device
from numpy import ndarray
import warnings
import numpy as np # For FasterWhisper audio conversion if needed

# OpenAI Whisper imports
from whisper import Whisper as OpenAIWhisperModel # Renamed for clarity
from whisper import load_model as openai_whisper_load_model # Renamed
from whisper.tokenizer import TO_LANGUAGE_CODE as OPENAI_WHISPER_TO_LANGUAGE_CODE # Renamed

# FasterWhisper imports
from faster_whisper import WhisperModel as FasterWhisperModel
from faster_whisper.tokenizer import _LANGUAGE_CODES as FASTER_WHISPER_LANGUAGE_CODES

# IPEX-LLM import (optional)
try:
    import ipex_llm
    print("INFO: IPEX-LLM library found.")
except ImportError:
    ipex_llm = None
    print("WARNING: IPEX-LLM library not found. IPEX-LLM specific optimizations will not be available for OpenAI Whisper.")

# Local project imports (assuming these exist in scraibe.misc)
try:
    from .misc import WHISPER_DEFAULT_PATH, SCRAIBE_TORCH_DEVICE, SCRAIBE_NUM_THREADS
except ImportError:
    print("WARNING: Could not import from .misc. Using placeholder constants.")
    WHISPER_DEFAULT_PATH = None
    SCRAIBE_TORCH_DEVICE = "cpu"
    SCRAIBE_NUM_THREADS = 4


# Using a generic TypeVar for the model instance in the ABC
ModelType = TypeVar('ModelType')


class Transcriber(ABC):
    """
    Abstract Base Class for Transcriber implementations.
    """

    def __init__(self, model_name: str, model_instance: ModelType, processor: Optional[Any] = None) -> None:
        """
        Initialize the Transcriber class.

        Args:
            model_name (str): The name of the model (e.g., "tiny", "medium").
            model_instance (ModelType): The loaded transcription model instance.
            processor (Optional[Any]): An optional processor object (e.g., for Hugging Face models).
        """
        self.model_name = model_name
        self.model = model_instance
        self.processor = processor

    @abstractmethod
    def transcribe(self, audio: Union[str, Tensor, ndarray], **kwargs) -> Dict[str, Any]:
        """
        Transcribe an audio file or audio data.

        Args:
            audio (Union[str, Tensor, ndarray]): Path to audio file, or audio data as Tensor/ndarray.
            **kwargs: Additional keyword arguments specific to the transcriber implementation.

        Returns:
            Dict[str, Any]: A dictionary containing at least a "text" key with the full transcript.
                            Optionally, it can include "segments" (list of dicts with start, end, text),
                            "language", and other metadata.
        """
        pass

    @staticmethod
    def save_transcript(transcript_data: Dict[str, Any], save_path: str) -> None:
        """
        Save the transcribed text to a file.

        Args:
            transcript_data (Dict[str, Any]): The dictionary returned by the transcribe method.
            save_path (str): The path to save the transcript text file.
        """
        text_content = transcript_data.get("text", "")
        if not text_content and "segments" in transcript_data: # Reconstruct if only segments given
            text_content = " ".join(seg.get("text", "") for seg in transcript_data["segments"]).strip()

        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(text_content)
        print(f'Transcript text saved to {save_path}')
        # Optionally, save segments to a separate file or a structured format like JSON if needed.

    @classmethod
    @abstractmethod
    def load_model(cls,
                   model_name: str,
                   download_root: Optional[str] = WHISPER_DEFAULT_PATH,
                   device_option: Optional[Union[str, device]] = SCRAIBE_TORCH_DEVICE,
                   **kwargs: Any
                   ) -> 'Transcriber':
        """
        Load the transcription model.

        Args:
            model_name (str): Name or path of the Whisper model.
            download_root (Optional[str]): Path to download/cache models.
            device_option (Optional[Union[str, device]]): Device to load the model on.
            **kwargs: Additional arguments specific to the model type or optimization.

        Returns:
            Transcriber: An instance of a Transcriber subclass.
        """
        pass

    @staticmethod
    @abstractmethod
    def _get_transcribe_kwargs(**kwargs: Any) -> Dict[str, Any]:
        """
        Filter and prepare keyword arguments for the specific model's transcribe method.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_name='{self.model_name}')"


class OpenAIWhisperIPEXLLMTranscriber(Transcriber):
    """
    Transcriber for OpenAI's Whisper model, with optional IPEX-LLM optimization.
    """
    def __init__(self, model_name: str, model_instance: OpenAIWhisperModel) -> None:
        super().__init__(model_name, model_instance)

    @classmethod
    def load_model(cls,
                   model_name: str = "medium",
                   download_root: Optional[str] = WHISPER_DEFAULT_PATH,
                   device_option: Optional[Union[str, torch.device]] = SCRAIBE_TORCH_DEVICE,
                   in_memory: bool = False, # Specific to openai-whisper
                   # IPEX-LLM specific arguments
                   use_ipex_llm: bool = True,
                   low_bit: str = 'bf16', # Default for XPU. Options: 'sym_int4', 'nf4', 'fp16', 'fp8' etc., or None
                   **kwargs: Any # Catches other potential args for whisper_load_model
                   ) -> 'OpenAIWhisperIPEXLLMTranscriber':

        target_device = torch.device(device_option if device_option else SCRAIBE_TORCH_DEVICE)
        
        # Determine the initial device for loading the model
        initial_load_device = target_device
        
        # Condition for performing IPEX-LLM XPU optimization
        perform_ipex_xpu_optimization = (use_ipex_llm and 
                                         ipex_llm is not None and 
                                         target_device.type == 'xpu')

        if perform_ipex_xpu_optimization:
            # For IPEX-LLM XPU optimization, we must load the model to CPU first
            initial_load_device = torch.device('cpu')
            print(f"Loading OpenAI Whisper model: '{model_name}' to CPU for IPEX-LLM XPU optimization.")
        else:
            print(f"Loading OpenAI Whisper model: '{model_name}' for device: '{target_device}'.")

        # Load the model using the determined initial_load_device
        _model = openai_whisper_load_model(name=model_name,
                                           download_root=download_root,
                                           device=initial_load_device, # Load to CPU if optimizing for XPU, else target_device
                                           in_memory=in_memory)
        
        print(f"Model '{model_name}' initially loaded on device: {_model.device}")

        if perform_ipex_xpu_optimization:
            print(f"Attempting IPEX-LLM optimization for OpenAI Whisper model (currently on {_model.device}) with low_bit='{low_bit}'...")
            try:
                # Ensure model is on CPU before optimization (should be if initial_load_device was 'cpu')
                if _model.device.type != 'cpu':
                    _model = _model.to('cpu')
                    print(f"Moved model to CPU for optimization. Current device: {_model.device}")

                _model = ipex_llm.optimize_model(_model,
                                                 low_bit=low_bit,
                                                 optimize_llm=True, # Default, good for Whisper
                                                 # cpu_embedding=True, # Optional: experiment if OOM issues on XPU
                                                 )
                print(f"IPEX-LLM optimization successful. Optimized model is on device: {_model.device}")
                
                # After successful optimization, move the model to the target XPU device
                print(f"Moving optimized model to target device: {target_device}...")
                _model = _model.to(target_device)
                print(f"Model is now on device: {_model.device}")

            except Exception as e:
                warnings.warn(f"IPEX-LLM optimization failed for OpenAI Whisper model: {e}. Model remains on {_model.device}.")
                # If optimization fails, the model is likely still the CPU version.
                # Move it to the target_device (XPU) anyway so the rest of the application can proceed.
                if str(_model.device) != str(target_device):
                    try:
                        _model = _model.to(target_device)
                        warnings.warn(f"Model (post-failed-optimization) moved to {target_device}. Current device: {_model.device}")
                    except Exception as move_e:
                        warnings.warn(f"Could not move model to {target_device} after failed optimization: {move_e}")
        
        elif use_ipex_llm: # IPEX-LLM was desired but conditions not met for XPU optimization
            if ipex_llm is None:
                warnings.warn("IPEX-LLM library not found. Skipping IPEX-LLM optimization.")
            elif target_device.type != 'xpu':
                warnings.warn(f"IPEX-LLM optimization skipped: Target device is '{target_device.type}', not 'xpu'.")
        
        # If no IPEX XPU optimization was performed, or if it failed and model was moved,
        # ensure model is on the final target_device if it's not already.
        if str(_model.device) != str(target_device):
            print(f"Ensuring model is on final target device {target_device}. Current: {_model.device}")
            _model = _model.to(target_device)
            print(f"Model finally on device: {_model.device}")
            
        return cls(model_name, _model)
        
    def transcribe(self, audio: Union[str, Tensor, ndarray], **kwargs: Any) -> Dict[str, Any]:
        """
        Transcribe using the OpenAI Whisper model (potentially IPEX-LLM optimized).
        """
        transcribe_kwargs = self._get_transcribe_kwargs(**kwargs)

        # OpenAI's model.transcribe already returns a dictionary.
        # No explicit torch.xpu.amp.autocast typically needed here if ipex-llm optimized the model
        # to operate in a specific precision (e.g., bf16 or low-bit).
        result_dict = self.model.transcribe(audio, **transcribe_kwargs)
        return result_dict

    @staticmethod
    def _get_transcribe_kwargs(**kwargs: Any) -> Dict[str, Any]:
        """
        Prepare keyword arguments for OpenAI Whisper's `transcribe` method.
        It accepts **decode_options.
        """
        # Known decode options for OpenAI Whisper (can be extended)
        # See whisper.decode.DecodingOptions
        known_options = [
            "language", "task", "fp16", "temperature", "sample_len", "best_of",
            "beam_size", "patience", "length_penalty", "prompt", "prefix", "suppress_blank",
            "suppress_tokens", "without_timestamps", "max_initial_timestamp",
            "word_timestamps", "prepend_punctuations", "append_punctuations", "verbose",
            "patience", "length_penalty", "condition_on_previous_text", "logprob_threshold",
            "no_speech_threshold", "compression_ratio_threshold"
        ]
        final_kwargs = {k: v for k, v in kwargs.items() if k in known_options}
        
        # Handle verbose explicitly as it's common and OpenAI Whisper uses None for default (no output)
        if "verbose" in final_kwargs and final_kwargs["verbose"] is None:
            pass
        elif "verbose" not in final_kwargs or not final_kwargs.get("verbose", False) :
            final_kwargs["verbose"] = None # Default to no detailed console output from whisper

        return final_kwargs


class FasterWhisperTranscriber(Transcriber):
    """
    Transcriber for FasterWhisper models.
    """
    def __init__(self, model_name: str, model_instance: FasterWhisperModel) -> None:
        super().__init__(model_name, model_instance)

    @classmethod
    def load_model(cls,
                   model_name: str = "medium",
                   download_root: Optional[str] = WHISPER_DEFAULT_PATH,
                   device_option: Optional[Union[str, torch.device]] = SCRAIBE_TORCH_DEVICE,
                   # FasterWhisper specific arguments
                   compute_type: str = "default", # Default for FasterWhisper e.g. "int8", "float16", "auto"
                   cpu_threads: int = SCRAIBE_NUM_THREADS,
                   num_workers: int = 1,
                   **kwargs: Any
                   ) -> 'FasterWhisperTranscriber':

        target_device_str = str(device_option if device_option else SCRAIBE_TORCH_DEVICE).lower()
        
        # Determine device and compute_type for FasterWhisper
        actual_fw_device = "cpu" # FasterWhisper's device argument ("cpu", "cuda", "auto")
        effective_compute_type = compute_type

        if "cuda" in target_device_str:
            actual_fw_device = "cuda"
            if effective_compute_type == "default": effective_compute_type = "float16"
        elif "xpu" in target_device_str:
            # FasterWhisper on XPU typically implies using OpenVINO backend via device="cpu"
            actual_fw_device = "cpu"
            if effective_compute_type == "default": effective_compute_type = "int8" # Good OpenVINO default
            warnings.warn(f"FasterWhisper on XPU: setting device to 'cpu' (for OpenVINO backend). Compute type: '{effective_compute_type}'.")
        else: # CPU
            actual_fw_device = "cpu"
            if effective_compute_type == "default": effective_compute_type = "int8"
        
        if actual_fw_device == 'cpu' and effective_compute_type == 'float16':
            warnings.warn(f"Compute type 'float16' with device 'cpu' for FasterWhisper may not be optimal. Consider 'int8' or 'auto'. Using '{effective_compute_type}'.")

        _model = FasterWhisperModel(model_size_or_path=model_name, # `model_size_or_path` is the arg
                                    download_root=download_root,
                                    device=actual_fw_device,
                                    compute_type=effective_compute_type,
                                    cpu_threads=cpu_threads,
                                    num_workers=num_workers)
        return cls(model_name, _model)

    def transcribe(self, audio: Union[str, Tensor, ndarray], **kwargs: Any) -> Dict[str, Any]:
        """
        Transcribe using the FasterWhisper model.
        """
        processed_audio = audio
        if isinstance(audio, Tensor):
            processed_audio = audio.cpu().numpy().astype(np.float32)
        elif isinstance(audio, np.ndarray):
            processed_audio = audio.astype(np.float32)
        # If str, FasterWhisper handles path directly

        transcribe_kwargs = self._get_transcribe_kwargs(**kwargs)
        
        segments_iterable, info = self.model.transcribe(processed_audio, **transcribe_kwargs)
        
        full_text_parts = []
        segments_data = []
        for i, seg in enumerate(segments_iterable):
            segment_text = seg.text.strip()
            full_text_parts.append(segment_text)
            segments_data.append({
                "id": i, 
                "start": round(seg.start, 3), 
                "end": round(seg.end, 3), 
                "text": segment_text,
                "tokens": seg.tokens,
                "temperature": seg.temperature,
                "avg_logprob": seg.avg_logprob,
                "compression_ratio": seg.compression_ratio,
                "no_speech_prob": seg.no_speech_prob,
                # Include seek if available, though not standard in segment object
            })
        
        full_text = " ".join(full_text_parts).strip()
        return {
            "text": full_text,
            "segments": segments_data,
            "language": info.language,
            "language_probability": info.language_probability,
            "duration_after_vad": getattr(info, "duration_after_vad", None), # If VAD used
            "transcription_time": getattr(info, "transcription_time", None) # If info provides it
        }

    @staticmethod
    def _get_transcribe_kwargs(**kwargs: Any) -> Dict[str, Any]:
        """
        Prepare keyword arguments for FasterWhisper's `transcribe` method.
        """
        # signature(FasterWhisperModel.transcribe).parameters.keys() is a good way,
        # but let's list common ones for clarity and to avoid issues if signature changes.
        known_faster_options = [
            "language", "task", "beam_size", "best_of", "patience", "length_penalty",
            "repetition_penalty", "no_repeat_ngram_size", "temperature", "compression_ratio_threshold",
            "log_prob_threshold", "no_speech_threshold", "condition_on_previous_text",
            "initial_prompt", "prefix", "suppress_blank", "suppress_tokens",
            "without_timestamps", "max_initial_timestamp", "word_timestamps", "vad_filter",
            "vad_parameters", "chunk_length" # FasterWhisper specific
        ]
        
        final_kwargs = {k: v for k, v in kwargs.items() if k in known_faster_options}

        if (language := kwargs.get("language")): # Ensure language code conversion if a name is passed
            try:
                final_kwargs["language"] = FasterWhisperTranscriber.convert_to_language_code(language)
            except ValueError as e:
                warnings.warn(str(e) + " Language will not be set for FasterWhisper.")
                if "language" in final_kwargs: del final_kwargs["language"]
        
        return final_kwargs

    @staticmethod
    def convert_to_language_code(lang_input: str) -> str:
        """
        Convert a language name (e.g., "english") or code (e.g., "en")
        to a language code recognized by FasterWhisper.
        """
        if not lang_input: # Handle empty or None lang_input
            return None # FasterWhisper will auto-detect

        lang_input_lower = lang_input.lower().strip()
        if lang_input_lower in FASTER_WHISPER_LANGUAGE_CODES:
            return lang_input_lower # Already a valid code

        # Check OpenAI's mapping (name -> code)
        if lang_input_lower in OPENAI_WHISPER_TO_LANGUAGE_CODE:
            code = OPENAI_WHISPER_TO_LANGUAGE_CODE[lang_input_lower]
            if code in FASTER_WHISPER_LANGUAGE_CODES:
                return code
            else: # Mapped to a code not in FasterWhisper's explicit list (should be rare)
                 warnings.warn(f"Language name '{lang_input}' mapped to '{code}', which is not in FasterWhisper's known codes. Using '{code}' anyway.")
                 return code


        # Fallback if no mapping found
        available_codes_str = ", ".join(sorted(list(FASTER_WHISPER_LANGUAGE_CODES)))
        available_names_str = ", ".join(sorted([name for name, code in OPENAI_WHISPER_TO_LANGUAGE_CODE.items() if code in FASTER_WHISPER_LANGUAGE_CODES]))
        raise ValueError(
            f"Language '{lang_input}' is not a valid language code or name for FasterWhisper. "
            f"Known codes: {available_codes_str}. Known names (mapped to codes): {available_names_str}."
        )


# --- Factory Function to Load Transcriber ---
def load_transcriber(
    model_name: str = "medium",
    whisper_type: str = 'openai-ipex-llm', # Default to OpenAI with IPEX-LLM potential
    download_root: Optional[str] = WHISPER_DEFAULT_PATH,
    device: Optional[Union[str, torch.device]] = None, # Allow None to use SCRAIBE_TORCH_DEVICE
    **kwargs: Any  # Pass through all other specific arguments
) -> Transcriber:
    """
    Factory function to load and initialize a specific type of Whisper transcriber.

    Args:
        model_name (str): Whisper model name (e.g., "tiny", "base", "medium", "large-v2").
        whisper_type (str): Type of Whisper implementation to use.
                            Options: "openai-ipex-llm" (or "whisper"), "faster-whisper".
        download_root (Optional[str]): Path for model downloads/cache.
        device (Optional[Union[str, torch.device]]): Target device (e.g., "cpu", "cuda", "xpu").
                                                     Defaults to SCRAIBE_TORCH_DEVICE.
        **kwargs: Additional arguments passed to the specific transcriber's load_model method.
                  For "openai-ipex-llm": in_memory, use_ipex_llm, low_bit.
                  For "faster-whisper": compute_type, cpu_threads, num_workers.

    Returns:
        Transcriber: An initialized transcriber instance.
    """
    target_device = device if device is not None else SCRAIBE_TORCH_DEVICE
    whisper_type_lower = whisper_type.lower()

    if whisper_type_lower in ('openai-ipex-llm', 'whisper'):
        # `use_ipex_llm` and `low_bit` can be passed via kwargs,
        # OpenAIWhisperIPEXLLMTranscriber.load_model has defaults for them.
        return OpenAIWhisperIPEXLLMTranscriber.load_model(
            model_name=model_name,
            download_root=download_root,
            device_option=target_device,
            **kwargs # Passes in_memory, use_ipex_llm, low_bit, etc.
        )
    elif whisper_type_lower == 'faster-whisper':
        # `compute_type`, `cpu_threads`, etc. can be passed via kwargs
        return FasterWhisperTranscriber.load_model(
            model_name=model_name,
            download_root=download_root,
            device_option=target_device,
            **kwargs # Passes compute_type, cpu_threads, etc.
        )
    else:
        raise ValueError(
            f"Whisper type '{whisper_type}' not recognized. "
            f"Choose from 'openai-ipex-llm' (or 'whisper'), 'faster-whisper'."
        )

if __name__ == '__main__':
    print("Transcriber Module - Example Usage")

    # Create a dummy audio file for testing (10 seconds of silence at 16kHz)
    dummy_audio_path = "dummy_audio_scraibe.wav"
    sample_rate = 16000
    duration = 10
    try:
        import soundfile as sf
        silence = np.zeros(int(sample_rate * duration), dtype=np.float32)
        sf.write(dummy_audio_path, silence, sample_rate)
        print(f"Created dummy audio file: {dummy_audio_path}")
        audio_source = dummy_audio_path
    except Exception as e:
        print(f"Could not create dummy audio file ({e}). Using a numpy array as fallback audio source.")
        audio_source = np.random.randn(sample_rate * 5).astype(np.float32) # 5s random noise

    # --- Test OpenAI Whisper with IPEX-LLM (if XPU available and ipex-llm installed) ---
    print("\n--- Testing OpenAIWhisperIPEXLLMTranscriber ---")
    try:
        # For XPU, use low_bit like 'bf16' or 'sym_int4'
        # For CPU, ipex-llm optimization still applies if use_ipex_llm=True (e.g., for bf16 on compatible CPUs)
        # but low_bit quantization might be more CPU-specific (e.g., sym_int8)
        openai_device = "xpu" if torch.xpu.is_available() and ipex_llm is not None else "cpu"
        openai_low_bit = 'bf16' if openai_device == "xpu" else None # BF16 on XPU, no low-bit quant on CPU for this example

        transcriber_openai = load_transcriber(
            model_name="tiny", # Use a small model for quick testing
            whisper_type="openai-ipex-llm",
            device=openai_device,
            low_bit=openai_low_bit,
            use_ipex_llm=True, # Explicitly enable, though it's default in the class
            language="en", # Pass as kwarg to transcribe
            verbose=False  # Pass as kwarg to transcribe
        )
        print(f"Loaded OpenAI/IPEX-LLM transcriber on {openai_device} with low_bit='{openai_low_bit}'")
        
        result_openai = transcriber_openai.transcribe(audio_source, language="en", verbose=False)
        print(f"OpenAI/IPEX-LLM Text: '{result_openai.get('text', 'N/A')}'")
        if result_openai.get("segments"):
            print("Segments (first 3):")
            for seg in result_openai["segments"][:3]:
                print(f"  ID {seg.get('id')}: [{seg.get('start',0):.2f}s -> {seg.get('end',0):.2f}s] {seg.get('text','N/A')}")
        # transcriber_openai.save_transcript(result_openai, "openai_ipex_transcript.txt")

    except Exception as e:
        print(f"Error testing OpenAIWhisperIPEXLLMTranscriber: {e}")
        import traceback
        traceback.print_exc()

    # --- Test FasterWhisper ---
    print("\n--- Testing FasterWhisperTranscriber ---")
    try:
        fw_device = "cuda" if torch.cuda.is_available() else "cpu"
        fw_compute = "float16" if fw_device == "cuda" else "int8"
        
        transcriber_faster = load_transcriber(
            model_name="tiny",
            whisper_type="faster-whisper",
            device=fw_device,
            compute_type=fw_compute # Specific kwarg for FasterWhisper
        )
        print(f"Loaded FasterWhisper transcriber on {fw_device} with compute_type='{fw_compute}'")

        result_faster = transcriber_faster.transcribe(audio_source, language="en") # Pass language kwarg
        print(f"FasterWhisper Text: '{result_faster.get('text', 'N/A')}'")
        if result_faster.get("segments"):
            print("Segments (first 3):")
            for seg in result_faster["segments"][:3]:
                 print(f"  ID {seg.get('id')}: [{seg.get('start',0):.2f}s -> {seg.get('end',0):.2f}s] {seg.get('text','N/A')}")
        # transcriber_faster.save_transcript(result_faster, "faster_whisper_transcript.txt")

    except Exception as e:
        print(f"Error testing FasterWhisperTranscriber: {e}")
        import traceback
        traceback.print_exc()

    # Clean up dummy audio file
    if isinstance(audio_source, str):
        try:
            import os
            os.remove(dummy_audio_path)
            print(f"Removed dummy audio file: {dummy_audio_path}")
        except OSError as e:
            print(f"Error removing dummy audio file {dummy_audio_path}: {e}")

    print("\nTranscriber module example usage finished.")