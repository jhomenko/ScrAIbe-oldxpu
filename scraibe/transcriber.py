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
from typing import TypeVar, Union, Optional, Dict, Any, List, Tuple
import tqdm # For the progress bar

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

# --- Define Whisper constants (adapted from openai-whisper/whisper/audio.py) ---
# These are typically derived from the model's config or feature_extractor,
# but defining them explicitly based on Whisper standards for clarity in this port.
# Ensure these match your loaded model/processor if they differ.
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160 # self.processor.feature_extractor.hop_length
CHUNK_LENGTH = 30 # seconds
N_SAMPLES_PER_CHUNK = CHUNK_LENGTH * SAMPLE_RATE # 480000
N_FRAMES_PER_CHUNK = N_SAMPLES_PER_CHUNK // HOP_LENGTH # 3000 frames for a 30-second window
# N_MELS = 80 # self.processor.feature_extractor.feature_size or self.model.config.num_mel_bins

# For token context
# MAX_TEXT_TOKEN_LENGTH = 448 # self.model.config.max_length or similar for decoder

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
                   use_ipex_llm: bool = True, # This class is IPEX-LLM specific
                   low_bit: str = 'bf16',    # Controls quantization: "bf16", "fp16", "int4", "sym_int4", etc.
                   use_auth_token: Optional[str] = None,
                   in_memory: bool = False, # Kept for signature compatibility if needed by other parts
                   verbose: bool = False,   # Pass verbose flag for loader prints
                   **kwargs: Any 
                   ) -> 'OpenAIWhisperIPEXLLMTranscriber':

        if not use_ipex_llm or ipex_llm is None: # Ensure ipex_llm is available
            raise ImportError("IPEX-LLM library not found or use_ipex_llm is False. "
                              "Cannot use OpenAIWhisperIPEXLLMTranscriber.")

        target_device_str = str(device_option if device_option else SCRAIBE_TORCH_DEVICE)
        target_device = torch.device(target_device_str)

        # Map short model names to full Hugging Face Hub identifiers
        hf_model_id = model_name
        official_short_names = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
        potential_short_name = model_name.replace(".en", "")
        if potential_short_name in official_short_names and "/" not in model_name:
            hf_model_id = f"openai/whisper-{model_name}"
            if verbose: print(f"Interpreted model_name '{model_name}' as Hugging Face ID '{hf_model_id}'")

        if verbose:
            print(f"Loading Whisper model '{hf_model_id}' using ipex_llm.transformers "
                  f"with low_bit configuration: '{low_bit}' for device '{target_device_str}'")

        # Load WhisperProcessor
        try:
            processor = WhisperProcessor.from_pretrained(
                hf_model_id, cache_dir=download_root, token=use_auth_token
            )
        except Exception as e:
            warnings.warn(f"Failed to load WhisperProcessor for '{hf_model_id}': {e}", RuntimeWarning)
            raise

        # Prepare arguments for AutoModelForSpeechSeq2Seq.from_pretrained
        from_pretrained_main_args = {} # For load_in_4bit or load_in_low_bit
        from_pretrained_other_kwargs = kwargs.copy() # For other passthrough args
        from_pretrained_other_kwargs.setdefault('trust_remote_code', True)
        from_pretrained_other_kwargs['cache_dir'] = download_root
        from_pretrained_other_kwargs['token'] = use_auth_token
        
        # Add cpu_embedding=True for XPU device, based on IPEX-LLM docs
        if target_device.type == 'xpu':
            from_pretrained_other_kwargs.setdefault('cpu_embedding', True)
            if verbose: print(f"INFO: Setting cpu_embedding=True for XPU device.")

        normalized_low_bit = low_bit.lower() if isinstance(low_bit, str) else ""
        effective_torch_dtype = "auto" # Default, let IPEX-LLM / Transformers decide

        # Path for boolean 4-bit loading (like the INT4 example script)
        if normalized_low_bit in ["int4", "4bit"]: # User explicitly asks for generic 4-bit
            if verbose: print(f"INFO: Using INT4 settings: load_in_4bit=True, optimize_model=False (for low_bit='{low_bit}')")
            from_pretrained_main_args['load_in_4bit'] = True
            from_pretrained_main_args['optimize_model'] = False 
            # For load_in_4bit, torch_dtype is typically not specified or "auto"
            effective_torch_dtype = None # Let IPEX-LLM handle dtype for this pure 4-bit loading
        
        # Path for specific named low_bit formats (like bf16, fp16, or specific int strings like "sym_int4")
        else:
            if verbose: print(f"INFO: Using specific low_bit string: load_in_low_bit='{low_bit}', optimize_model=True")
            from_pretrained_main_args['load_in_low_bit'] = low_bit # Pass the string e.g., "bf16", "sym_int4"
            from_pretrained_main_args['optimize_model'] = True  # Generally True for specific low_bit optimizations

            if normalized_low_bit == "bf16":
                effective_torch_dtype = torch.bfloat16
            elif normalized_low_bit == "fp16":
                effective_torch_dtype = torch.float16
                if target_device.type == 'cpu':
                    warnings.warn("FP16 may not be optimal on CPU for all operations.", UserWarning)
            elif normalized_low_bit in ["fp32", "none", ""]: # Explicit FP32 or no quantization
                effective_torch_dtype = torch.float32
                # If low_bit was specifically to turn off quantization, clear load_in_low_bit
                if 'load_in_low_bit' in from_pretrained_main_args and (normalized_low_bit in ["fp32", "none", ""]):
                    from_pretrained_main_args.pop('load_in_low_bit')
            else: # For other specific quantization strings (e.g., "sym_int4", "nf4")
                  # let IPEX-LLM infer/handle the torch_dtype.
                effective_torch_dtype = None 

        # Set torch_dtype if it's determined and not 'auto' or None
        if effective_torch_dtype is not None and effective_torch_dtype != "auto":
            from_pretrained_main_args['torch_dtype'] = effective_torch_dtype
        # If effective_torch_dtype ended up as None (e.g. for int4 or specific quant strings),
        # ensure torch_dtype is not in main_args to let from_pretrained default.
        elif 'torch_dtype' in from_pretrained_main_args and effective_torch_dtype is None:
            from_pretrained_main_args.pop('torch_dtype')

        # Combine all arguments for from_pretrained
        final_from_pretrained_args = {**from_pretrained_main_args, **from_pretrained_other_kwargs}

        if verbose:
            print(f"DEBUG: Final from_pretrained arguments for model loading: {final_from_pretrained_args}")

        try:
            model_instance = AutoModelForSpeechSeq2Seq.from_pretrained(
                hf_model_id,
                **final_from_pretrained_args
            )
        except Exception as e:
            warnings.warn(f"Failed to load model '{hf_model_id}' using IPEX-LLM with args {final_from_pretrained_args}: {e}", RuntimeWarning)
            raise

        model_instance = model_instance.eval()

        try:
            model_instance = model_instance.to(target_device)
            loaded_model_dtype = next(model_instance.parameters()).dtype
            print(f"IPEX-LLM Whisper model '{hf_model_id}' loaded. Target device: '{target_device}'. "
                  f"Effective model dtype: {loaded_model_dtype}. Low-bit config: '{low_bit}'.")
        except Exception as e:
            warnings.warn(f"Failed to move IPEX-LLM loaded model to '{target_device}': {e}. "
                          f"Model remains on its current device: {model_instance.device}", RuntimeWarning)
            target_device = model_instance.device

        return cls(hf_model_id, model_instance, processor, target_device, low_bit_format=low_bit)

    def _pad_or_trim_features(self, features: torch.Tensor, length: int = N_FRAMES_PER_CHUNK) -> torch.Tensor:
        """Pads or trims the input mel spectrogram features to a specified length."""
        if features.shape[-1] > length:
            features = features[..., :length]
        elif features.shape[-1] < length:
            padding = length - features.shape[-1]
            features = torch.nn.functional.pad(features, (0, padding))
        return features

    def transcribe(self, audio: Union[str, torch.Tensor, np.ndarray], **kwargs: Any) -> Dict[str, Any]:
        """
        Transcribe audio using the IPEX-LLM loaded Whisper model.
        This version relies on the model's internal long-form processing capabilities
        and timestamp generation.
        """
        if not self.processor or not self.model:
            raise RuntimeError("Transcriber is not properly initialized with a model and processor.")

        if "verbose" in kwargs: self.verbose = kwargs["verbose"]

        if isinstance(audio, str):
            raise NotImplementedError("This method expects a pre-processed waveform Tensor.")
        elif isinstance(audio, np.ndarray):
            audio_waveform = torch.from_numpy(audio.astype(np.float32))
        elif isinstance(audio, torch.Tensor):
            audio_waveform = audio.to(torch.float32)
        else:
            raise TypeError(f"Expected audio to be Tensor or ndarray, got {type(audio)}")

        if audio_waveform.ndim > 1: audio_waveform = audio_waveform.squeeze()
        if audio_waveform.ndim != 1: raise ValueError(f"Audio waveform must be 1D, got {audio_waveform.ndim} dims.")

        if self.verbose: print("Processing full audio with self.processor...")
        try:
            input_features = self.processor(
                audio_waveform.cpu().numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt"
            ).input_features
        except Exception as e:
            warnings.warn(f"Error during full audio feature extraction: {e}", RuntimeWarning)
            raise

        model_dtype = next(self.model.parameters()).dtype
        if input_features.device != self.target_device:
            input_features = input_features.to(self.target_device)
        if input_features.dtype != model_dtype:
            if self.verbose: print(f"Casting full_input_features to model dtype {model_dtype}")
            input_features = input_features.to(model_dtype)

        # Prepare options for model.generate, using defaults from _get_transcribe_kwargs
        # and allowing overrides from **kwargs passed to this transcribe method.
        generate_options = self._get_transcribe_kwargs(**kwargs)
        
        # Language and task are key arguments for the model's generate method
        # and are included by _get_transcribe_kwargs if passed in kwargs.
        # If not, they default to None/transcribe inside _get_transcribe_kwargs,
        # and the model's generate should handle auto-detection or its own defaults.
        current_language = generate_options.get("language") # This will be None if not specified, for auto-detect
        
        if self.verbose:
            print(f"Calling model.generate() for full audio. Options: {generate_options}")
            print(f"Model device: {self.model.device}, Input features device: {input_features.device}, dtype: {input_features.dtype}")

        transcription_output = None
        with torch.no_grad():
            try:
                # The `generate` method of `WhisperForConditionalGeneration` (which ipex_llm model is based on)
                # has built-in long-form transcription capabilities and can return timestamps and segments.
                # We need to ensure `return_timestamps=True` is passed for segments.
                # And `return_dict_in_generate=True` to get a structured output.
                
                generate_options.setdefault("return_dict_in_generate", True)
                generate_options.setdefault("return_timestamps", True) # Request timestamps
                # generate_options.setdefault("chunk_length_s", 30) # Might be needed for internal chunking if not default
                # generate_options.setdefault("stride_length_s", 5) # For overlapping if model's generate supports it

                transcription_output = self.model.generate(
                    input_features,
                    **generate_options
                )

                if self.target_device.type == "xpu":
                    torch.xpu.synchronize()

                # Process the output. `generate` with `return_dict_in_generate=True` returns a ModelOutput object
                # or a list of dicts if `return_segments=True` was also handled.
                # For Whisper, it typically returns a sequence of token IDs.
                # We need to decode these and potentially extract segments if the output structure provides them.
                
                # If `transcription_output` is a tensor of IDs:
                predicted_ids = transcription_output if isinstance(transcription_output, torch.Tensor) else transcription_output.sequences

                full_text = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
                
                segments = []
                # The HF `generate` for Whisper can return segments if `return_timestamps=True`
                # The actual structure of `transcription_output` needs to be inspected.
                # If `transcription_output` is a dict and contains 'segments':
                if isinstance(transcription_output, dict) and "segments" in transcription_output:
                    segments = transcription_output["segments"] # Assuming this format
                elif hasattr(transcription_output, "segments"): # If it's an object with a segments attribute
                    segments = transcription_output.segments
                else: # Fallback if detailed segments are not directly available from this call
                    if full_text:
                        # For now, create a single segment for the whole text if no detailed ones
                        # This needs to be improved to match the detailed segments from original whisper
                        # by parsing timestamp tokens if predicted_ids include them.
                        warnings.warn("Detailed segments with timestamps not yet fully extracted from model.generate() output. "
                                      "Returning chunk-level or full text segment.", UserWarning)
                        segments.append({"id": 0, "seek": 0, "start": 0.0, "end": audio_waveform.shape[0]/SAMPLE_RATE, "text": full_text, "tokens": predicted_ids.squeeze().tolist()})

                if self.verbose:
                    print(f"--- Transcription ---")
                    print(full_text)
                    if segments and segments[0]['text'] != full_text : print(f"Segments: {segments[:2]}...") # Print first few if different
                    print(f"--- End Transcription ---")

                # Determine final language
                # If language was None, model.generate might fill it in the output or config
                # For now, use the language passed to generate_options or default
                final_language = generate_options.get("language", "en") # Fallback
                if hasattr(transcription_output, "language"):
                    final_language = transcription_output.language
                elif isinstance(transcription_output, dict) and "language" in transcription_output:
                    final_language = transcription_output["language"]


                return {
                    "text": full_text,
                    "segments": segments,
                    "language": final_language
                }

            except Exception as e:
                warnings.warn(f"Error during model.generate() or full audio decoding: {e}", RuntimeWarning)
                import traceback
                traceback.print_exc()
                return {"text": "", "segments": [], "language": generate_options.get("language", "en")}


    @staticmethod
    def _get_transcribe_kwargs(**kwargs: Any) -> Dict[str, Any]:
        """
        Prepare keyword arguments for the model's sophisticated `generate` method,
        which handles long-form transcription and timestamping internally.
        """
        generate_params = {}
        
        # Parameters from the ipex_llm/transformers WhisperModel.generate signature
        # and common generate/TranscriptionOptions parameters.
        known_options = [
            "language", "task", "temperature", "compression_ratio_threshold", 
            "logprob_threshold", "no_speech_threshold", "condition_on_prev_tokens",
            "initial_prompt", "return_timestamps", "return_token_timestamps",
            "num_beams", "patience", "length_penalty", "repetition_penalty",
            "no_repeat_ngram_size", "suppress_tokens", "max_new_tokens", "use_cache",
            "do_sample", "top_k", "top_p",
            # from original whisper/TranscriptionOptions that might map
            "beam_size", "best_of", "prefix", "suppress_blank", 
            "without_timestamps", "max_initial_timestamp", "word_timestamps", # word_timestamps often controls return_token_timestamps
            "prepend_punctuations", "append_punctuations" 
        ]

        for k, v in kwargs.items():
            if k in known_options:
                generate_params[k] = v
            elif k == "sample_len" and "max_new_tokens" not in generate_params:
                generate_params["max_new_tokens"] = v
            # Map beam_size from CLI to num_beams for generate
            elif k == "beam_size" and "num_beams" not in generate_params:
                 generate_params["num_beams"] = v
        
        # --- Sensible Defaults for long-form if not provided ---
        generate_params.setdefault('language', None) # Let generate handle detection if not specified by user
        generate_params.setdefault('task', "transcribe")
        generate_params.setdefault('return_timestamps', True) # ESSENTIAL for diarization later
        generate_params.setdefault('condition_on_prev_tokens', True)
        
        # Temperature: model.generate can often take a tuple for fallback
        generate_params.setdefault('temperature', (0.0, 0.2, 0.4, 0.6, 0.8, 1.0))
        
        generate_params.setdefault('no_repeat_ngram_size', 3) # Good general default

        # Beam search vs. Sampling (align with temperature)
        user_num_beams = generate_params.get('num_beams')
        current_temp = generate_params.get('temperature')
        # If temperature is a tuple, take the first for this logic
        first_temp = current_temp[0] if isinstance(current_temp, (list, tuple)) and current_temp else current_temp

        if user_num_beams is not None and user_num_beams > 1: # User wants beam search
            generate_params['do_sample'] = False
            if first_temp is None or first_temp > 0.2: # Beam search usually uses low/zero temp
                generate_params['temperature'] = 0.0 
        elif generate_params.get('do_sample', False): # User wants sampling
            generate_params['num_beams'] = 1 # Ensure no beam search
            if first_temp is None: generate_params['temperature'] = 0.2 # Default sampling temp
        else: # Default to greedy or beam if temperature is very low
            if first_temp is None or first_temp <= 0.1: # Could be greedy or beam
                 generate_params.setdefault('num_beams', 1) # Default to greedy for speed unless user specified >1
                 if generate_params['num_beams'] == 1:
                     generate_params.pop('temperature', None) # No temp for pure greedy
                     generate_params['do_sample'] = False
            else: # Higher initial temp implies sampling
                 generate_params['do_sample'] = True
                 generate_params['num_beams'] = 1


        # Max new tokens per chunk (generate handles chunking internally)
        # The model's generate has its own way of figuring out max_length per internal segment.
        # max_new_tokens at this top level might control total tokens for the whole audio,
        # or it might be a per-chunk setting if generate's API uses it that way.
        # For now, let Whisper's generate manage this based on its internal chunking.
        # We can remove our previous N_FRAMES_PER_CHUNK // 2 default for this if generate handles it.
        generate_params.pop('max_new_tokens', None) # Let the model's generate handle this.
                                                   # Or set a very large value if it's for the whole audio.
                                                   # The HF pipeline example sets max_new_tokens=128 per chunk.
                                                   # The `whisper_generate` has internal `num_segment_frames`.
        if 'max_new_tokens' not in generate_params and kwargs.get('chunk_length_s'): # If user provides chunk_length_s
            generate_params['max_new_tokens'] = 128 # A common default per chunk for pipeline

        generate_params.setdefault('use_cache', True)

        if generate_params.get('temperature') == 0.0 and \
           not generate_params.get('do_sample', False) and \
           generate_params.get('num_beams', 1) == 1:
            generate_params.pop('temperature', None) # Silence transformers warning for pure greedy

        return generate_params


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