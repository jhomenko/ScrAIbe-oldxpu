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

from ipex_llm.transformers import AutoModelForSpeechSeq2Seq
from transformers import WhisperProcessor

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
    Transcriber for OpenAI's Whisper model, using ipex_llm.transformers for loading
    and optimization, especially for Intel XPUs.
    """
    def __init__(self, 
                 model_name: str, 
                 model_instance: AutoModelForSpeechSeq2Seq, # Model instance is now from ipex_llm.transformers
                 processor_instance: WhisperProcessor,    # Explicitly store processor
                 target_device: torch.device,
                 low_bit_format: Optional[str] = None):   # Store low_bit for reference
        
        # Call super().__init__ using the new 'processor' argument in the ABC if you've added it,
        # or handle it as needed. For now, let's assume the ABC's __init__ is:
        # super().__init__(model_name, model_instance, processor_instance)
        # If your Transcriber ABC's __init__ is still (model_name, model_instance), 
        # you'll store processor separately:
        self.model_name = model_name
        self.model = model_instance
        self.processor = processor_instance # Store the WhisperProcessor
        self.target_device = target_device
        self.low_bit_format = low_bit_format
        self.verbose = False # Or get from kwargs if needed

    @classmethod
    def load_model(cls,
                   model_name: str = "medium",
                   download_root: Optional[str] = None, 
                   device_option: Optional[Union[str, torch.device]] = None,
                   use_ipex_llm: bool = True,
                   low_bit: str = 'bf16',
                   use_auth_token: Optional[str] = None,
                   in_memory: bool = False, # Kept for signature compatibility
                   **kwargs: Any 
                   ) -> 'OpenAIWhisperIPEXLLMTranscriber':

        if not use_ipex_llm:
            warnings.warn("OpenAIWhisperIPEXLLMTranscriber is intended for use_ipex_llm=True. "
                          "Proceeding with IPEX-LLM loading.", UserWarning)

        if ipex_llm is None: # Ensure ipex_llm is imported and checked at module level
            raise ImportError("IPEX-LLM library not found, cannot use OpenAIWhisperIPEXLLMTranscriber.")

        target_device_str = str(device_option if device_option else SCRAIBE_TORCH_DEVICE) # Use your constants
        target_device = torch.device(target_device_str)

        # --- Map short model names to full Hugging Face Hub identifiers ---
        hf_model_id = model_name
        # Common Whisper model sizes from OpenAI
        official_short_names = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
        
        # Check if model_name is a short name (e.g., "medium" or "tiny.en")
        # and doesn't already look like a full path (e.g., "openai/whisper-medium")
        potential_short_name = model_name.replace(".en", "") # Handle "tiny.en" -> "tiny" for the check
        if potential_short_name in official_short_names and "/" not in model_name:
            hf_model_id = f"openai/whisper-{model_name}"
            print(f"Interpreted model_name '{model_name}' as Hugging Face ID '{hf_model_id}'")
        # --- End mapping ---

        print(f"Loading Whisper model '{hf_model_id}' using ipex_llm.transformers "
              f"with low_bit='{low_bit}' for device '{target_device_str}'")

        # 1. Load WhisperProcessor
        try:
            processor = WhisperProcessor.from_pretrained(
                hf_model_id, # Use mapped ID
                cache_dir=download_root,
                token=use_auth_token
            )
        except Exception as e:
            warnings.warn(f"Failed to load WhisperProcessor for '{hf_model_id}': {e}", RuntimeWarning)
            raise

        # 2. Load model using ipex_llm.transformers.AutoModelForSpeechSeq2Seq
        torch_dtype_map = {
            'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32,
        }
        effective_torch_dtype = torch_dtype_map.get(low_bit.lower() if isinstance(low_bit, str) else "fp32", "auto")
        if effective_torch_dtype == "auto" and low_bit not in ['bf16', 'fp16', 'fp32', None]:
            effective_torch_dtype = None 

        if target_device.type == 'cpu' and effective_torch_dtype == torch.float16:
            warnings.warn("FP16 is not natively supported or optimal on CPU. "
                          "IPEX-LLM might upcast or manage dtype with 'load_in_low_bit'.")
        
        from_pretrained_kwargs = kwargs.copy()
        from_pretrained_kwargs.setdefault('trust_remote_code', True)

        try:
            model_instance = AutoModelForSpeechSeq2Seq.from_pretrained(
                hf_model_id, # Use mapped ID
                load_in_low_bit=low_bit,
                optimize_model=True,
                torch_dtype=effective_torch_dtype if effective_torch_dtype else "auto",
                cache_dir=download_root,
                token=use_auth_token,
                **from_pretrained_kwargs
            )
        except Exception as e:
            warnings.warn(f"Failed to load model '{hf_model_id}' using IPEX-LLM AutoModelForSpeechSeq2Seq: {e}", RuntimeWarning)
            raise

        model_instance = model_instance.eval()

        try:
            model_instance = model_instance.to(target_device)
            loaded_model_dtype = next(model_instance.parameters()).dtype
            print(f"IPEX-LLM Whisper model '{hf_model_id}' loaded. Target device: '{target_device}'. "
                  f"Effective model dtype: {loaded_model_dtype}. Low-bit optimization: '{low_bit}'.")
        except Exception as e:
            warnings.warn(f"Failed to move IPEX-LLM loaded model to '{target_device}': {e}. "
                          f"Model remains on its current device: {model_instance.device}", RuntimeWarning)
            target_device = model_instance.device

        return cls(hf_model_id, model_instance, processor, target_device, low_bit_format=low_bit)

    def transcribe(self, audio: Union[str, torch.Tensor, np.ndarray], **kwargs: Any) -> Dict[str, Any]:
        """
        Transcribe audio using the IPEX-LLM loaded Whisper model and its generate method.
        Uses decoder_input_ids to set initial language and task tokens.
        """
        if not self.processor or not self.model:
            raise RuntimeError("Transcriber is not properly initialized with a model and processor.")

        if "verbose" in kwargs: # Update instance verbose if passed for this call
            self.verbose = kwargs["verbose"]

        if isinstance(audio, str):
            raise NotImplementedError("This transcribe method expects a waveform Tensor, not a path. "
                                      "Ensure Scraibe passes audio_processor.waveform.")
        elif isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio.astype(np.float32))
        elif not isinstance(audio, torch.Tensor):
            raise TypeError(f"Expected audio to be str, Tensor, or ndarray, but got {type(audio)}")

        # Ensure audio is a 1D float32 tensor for the processor
        if audio.ndim > 1: audio = audio.squeeze()
        if audio.ndim != 1: raise ValueError(f"Audio waveform must be 1D, but got {audio.ndim} dimensions.")
        if audio.dtype != torch.float32: audio = audio.to(torch.float32)

        # 1. Preprocess audio to input features
        try:
            input_features = self.processor(
                audio.cpu().numpy(), # Processor might expect numpy array
                sampling_rate=16000, # Whisper standard
                return_tensors="pt"
            ).input_features
        except Exception as e:
            warnings.warn(f"Error during processor feature extraction: {e}", RuntimeWarning)
            raise

        # 2. Move input features to the same device and dtype as the model expects
        try:
            model_first_param_dtype = next(self.model.parameters()).dtype
            if input_features.device != self.target_device:
                input_features = input_features.to(self.target_device)
            if input_features.dtype != model_first_param_dtype:
                if self.verbose:
                    print(f"Casting input_features from {input_features.dtype} to model dtype {model_first_param_dtype} "
                          f"for device {self.target_device}.")
                input_features = input_features.to(model_first_param_dtype)
        except Exception as e:
            warnings.warn(f"Could not move/cast input_features to target device/dtype: {e}", RuntimeWarning)
            # Proceeding, but there might be issues.

        # 3. Prepare decoder_input_ids for language and task
        language = kwargs.get("language", "en")
        task = kwargs.get("task", "transcribe")
        
        initial_decoder_ids_tensor: Optional[torch.Tensor] = None
        try:
            # get_decoder_prompt_ids returns a list of lists, e.g., [[50258, 50259, 50359, 50363]]
            # We need a tensor of shape (batch_size, sequence_length) for decoder_input_ids
            prompt_ids_list = self.processor.get_decoder_prompt_ids(language=language, task=task)
            if prompt_ids_list: 
                 initial_decoder_ids_tensor = torch.tensor(prompt_ids_list, device=self.target_device).long()
                 # Ensure it's 2D: (batch_size, sequence_length)
                 if initial_decoder_ids_tensor.ndim == 1: 
                     initial_decoder_ids_tensor = initial_decoder_ids_tensor.unsqueeze(0)
            else:
                warnings.warn("processor.get_decoder_prompt_ids returned empty or None.", UserWarning)
        except Exception as e:
            warnings.warn(f"Could not get initial_decoder_ids for language='{language}', task='{task}': {e}. "
                          "Model will use its defaults for starting sequence.", UserWarning)
        
        if self.verbose:
            print(f"Transcribing with: language='{language}', task='{task}', "
                  f"model_device='{self.model.device}', input_features_device='{input_features.device}', "
                  f"input_features_dtype='{input_features.dtype}'")
            if initial_decoder_ids_tensor is not None:
                 try:
                     # Squeeze to make it 1D for convert_ids_to_tokens if batch size is 1
                     display_ids_list = initial_decoder_ids_tensor.squeeze().tolist()
                     decoded_prompt_tokens = self.processor.tokenizer.convert_ids_to_tokens(display_ids_list)
                     print(f"Using initial decoder_input_ids (prompt): {decoded_prompt_tokens}")
                 except Exception:
                     print(f"Using initial decoder_input_ids: (unable to decode for display)")

        # 4. Get other generate options, ensuring 'forced_decoder_ids' is not among them
        generate_options = self._get_transcribe_kwargs(**kwargs) 
        generate_options.pop('forced_decoder_ids', None) # We are using decoder_input_ids instead

        # 5. Generate token IDs
        transcription_text = ""
        with torch.no_grad():
            try:
                generate_options.setdefault('use_cache', True) # As in benchmark
                
                predicted_ids = self.model.generate(
                    input_features,
                    decoder_input_ids=initial_decoder_ids_tensor, # Pass the prepared decoder input IDs
                    **generate_options
                )

                if self.target_device.type == "xpu":
                    torch.xpu.synchronize()

                # 6. Decode token IDs to text
                transcription_text = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                
                if self.verbose:
                    print(f"--- Transcription ---")
                    print(transcription_text)
                    print(f"--- End Transcription ---")

            except Exception as e:
                warnings.warn(f"Error during model.generate() or decoding: {e}", RuntimeWarning)
                import traceback
                traceback.print_exc() # Print full traceback for the error in generate
                return {"text": "", "segments": [], "language": language}

        # Basic segment creation (full text as one segment)
        segments = []
        if transcription_text:
            segments.append({"start": 0.0, "end": 0.0, "text": transcription_text.strip()}) 

        return {
            "text": transcription_text.strip(),
            "segments": segments,
            "language": language 
        }

    @staticmethod
    def _get_transcribe_kwargs(**kwargs: Any) -> Dict[str, Any]:
        """
        Prepare keyword arguments for Hugging Face model's `generate` method.
        This filters kwargs passed to `transcribe` and keeps only those valid for `generate`.
        """
        # Common parameters for `generate`. Refer to Hugging Face docs for full list.
        # (transformers.generation.GenerationConfig)
        known_generate_options = [
            # Controlling output length
            "max_length", "max_new_tokens", "min_length", "min_new_tokens",
            # Strategy
            "early_stopping", "num_beams", "num_beam_groups", "do_sample", "use_cache",
            # Sampling parameters (if do_sample=True)
            "temperature", "top_k", "top_p", "typical_p", "epsilon_cutoff", "eta_cutoff",
            # Advanced
            "repetition_penalty", "length_penalty", "no_repeat_ngram_size",
            "encoder_no_repeat_ngram_size", "bad_words_ids", "force_words_ids",
            "forced_bos_token_id", "forced_eos_token_id", "remove_invalid_values",
            "suppress_tokens", "begin_suppress_tokens", "forced_decoder_ids",
            "num_return_sequences", "output_attentions", "output_hidden_states",
            "output_scores", "return_dict_in_generate",
            # Whisper specific that might be passed to generate or handled by processor:
            # "language", "task", (these are handled for forced_decoder_ids)
            # Parameters from original OpenAI Whisper DecodingOptions that map to generate:
            "patience", # (can map to early_stopping related logic or beam search patience if available)
            # "sample_len" -> max_new_tokens
            # "best_of" -> num_return_sequences (and then select best, or num_beams with do_sample=True)
            # "beam_size" -> num_beams
            # "prompt" -> "prompt_ids" or "prefix_allowed_tokens_fn" (more complex)
            # "prefix" -> (similar to prompt)
            # "suppress_blank" -> (handled by tokenizer options usually, or post-processing)
            # "without_timestamps" -> (if model supports, or post-processing)
            # "max_initial_timestamp" -> (specific to whisper's timestamp logic)
        ]
        
        final_kwargs = {}
        for k, v in kwargs.items():
            if k in known_generate_options:
                final_kwargs[k] = v
            elif k == "sample_len" and "max_new_tokens" not in final_kwargs: # map sample_len
                final_kwargs["max_new_tokens"] = v
            elif k == "beam_size" and "num_beams" not in final_kwargs: # map beam_size
                 final_kwargs["num_beams"] = v
            # Add more mappings if needed from whisper's DecodingOptions to HF generate()
        
        # Ensure num_beams is > 0 if set, and do_sample is False for beam search
        if final_kwargs.get("num_beams", 0) > 0:
            final_kwargs.setdefault("do_sample", False)
            if final_kwargs["num_beams"] == 1 and not final_kwargs.get("do_sample", False): # Greedy
                final_kwargs.pop("num_beams", None) # No need for num_beams=1 in greedy

        # Default to greedy search if no strategy is specified
        if not final_kwargs.get("do_sample", False) and final_kwargs.get("num_beams", 1) <=1:
            # This is effectively greedy. Ensure no conflicting sampling params.
            final_kwargs.pop("temperature", None)
            final_kwargs.pop("top_k", None)
            final_kwargs.pop("top_p", None)

        return final_kwargs


# Ensure your FasterWhisperTranscriber and load_transcriber factory function are still in this file
# The load_transcriber factory will need to call this new version of 
# OpenAIWhisperIPEXLLMTranscriber.load_model correctly.
# The existing call in load_transcriber:
#   return OpenAIWhisperIPEXLLMTranscriber.load_model(
#       model_name=model_name,
#       download_root=download_root,
#       device_option=target_device, # This 'device_option' is the param name in the new load_model
#       **kwargs # kwargs here are from load_transcriber's **kwargs
#                # which are from Scraibe.__init__'s component_kwargs.
#                # These include `low_bit`, `use_ipex_llm`, `use_auth_token`.
#   )
# This call structure should still be compatible with the refactored class method.

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