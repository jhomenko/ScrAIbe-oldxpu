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

from transformers import pipeline
from ipex_llm.transformers import AutoModelForSpeechSeq2Seq
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# FasterWhisper imports
from faster_whisper import WhisperModel as FasterWhisperModel
from faster_whisper.tokenizer import _LANGUAGE_CODES as FASTER_WHISPER_LANGUAGE_CODES

from transformers import WhisperForConditionalGeneration # Import standard HF model
try:
    import intel_extension_for_pytorch as ipex # Import base IPEX
    print("INFO: Intel Extension for PyTorch (IPEX) library found.")
except ImportError:
    ipex = None
    print("WARNING: Intel Extension for PyTorch (IPEX) library not found. "
          "XPU optimizations via ipex.optimize() will not be available.")

# --- Define Whisper constants (adapted from openai-whisper/whisper/audio.py) ---
# These are typically derived from the model's config or feature_extractor,
# but defining them explicitly based on Whisper standards for clarity in this port.
# Ensure these match your loaded model/processor if they differ.
SAMPLE_RATE = 16000
#N_FFT = 400
#HOP_LENGTH = 160 # self.processor.feature_extractor.hop_length
chunk_length_s = 30 # seconds
stride_length_s = 5 # seconds
batch_size = 8
return_timestamps = True # For segment-level timestamps
#N_SAMPLES_PER_CHUNK = CHUNK_LENGTH * SAMPLE_RATE # 480000
#N_FRAMES_PER_CHUNK = N_SAMPLES_PER_CHUNK // HOP_LENGTH # 3000 frames for a 30-second window
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
    def __init__(self, model_name: str, model_instance: Any, processor: Optional[Any] = None, verbose: bool = False) -> None:
        self.model_name = model_name
        self.model = model_instance
        self.processor = processor
        self.verbose = verbose

    @abstractmethod
    def transcribe(self, audio: Union[str, Tensor, np.ndarray], **kwargs) -> Dict[str, Any]:
        pass

    @staticmethod
    def save_transcript(transcript_data: Dict[str, Any], save_path: str) -> None:
        text_content = transcript_data.get("text", "")
        if not text_content and "segments" in transcript_data: # Try to reconstruct if only segments with text exist
            text_content = " ".join(
                seg.get("text", "").strip() for seg in transcript_data["segments"] if seg.get("text")
            ).strip()
        
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(text_content)
            # Check if file was actually written with content, or if text_content itself was empty
            if os.path.exists(save_path) and (os.path.getsize(save_path) > 0 or not text_content):
                print(f'Transcript text saved to {save_path}')
            else:
                # This case could happen if text_content is empty and file is created empty.
                print(f'Warning: Transcript text file at {save_path} appears empty.')
        except Exception as e:
            print(f"Error saving transcript to {save_path}: {e}")

    @classmethod
    @abstractmethod
    def load_model(cls, model_name: str, download_root: Optional[str] = None, 
                   device_option: Optional[Union[str, device]] = None, 
                   **kwargs: Any) -> 'Transcriber':
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_name='{self.model_name}')"

class OpenAIWhisperIPEXLLMTranscriber(Transcriber):
    """
    Transcriber for Whisper model using ipex_llm.transformers, leveraging
    the model's internal long-form transcription capabilities.
    """
    def __init__(self, 
                 model_name: str, 
                 model_instance: AutoModelForSpeechSeq2Seq, # Correct type
                 processor_instance: WhisperProcessor,
                 target_device: torch.device,
                 low_bit_format: Optional[str] = None,
                 verbose: bool = False):
        super().__init__(model_name, model_instance, processor_instance, verbose=verbose)
        self.target_device = target_device
        self.low_bit_format = low_bit_format
        # self.model, self.processor, self.verbose are set by super()

    @classmethod
    def load_model(cls,
                   model_name: str = "medium",
                   download_root: Optional[str] = None, 
                   device_option: Optional[Union[str, torch.device]] = None,
                   use_ipex_llm: bool = True, # Should always be True for this class
                   low_bit: str = 'bf16', 
                   use_auth_token: Optional[str] = None,
                   verbose: bool = False, 
                   **kwargs: Any # Catches trust_remote_code etc.
                   ) -> 'OpenAIWhisperIPEXLLMTranscriber':

        if not use_ipex_llm or ipex_llm is None: # Ensure ipex_llm is available
            raise ImportError("IPEX-LLM library not found or use_ipex_llm is False for OpenAIWhisperIPEXLLMTranscriber.")

        target_device_str = str(device_option if device_option else SCRAIBE_TORCH_DEVICE)
        target_device = torch.device(target_device_str)

        hf_model_id = model_name
        official_short_names = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
        # Handle .en models correctly by stripping .en for the short name check, but using original for full ID construction
        potential_short_name_base = model_name.split('.')[0] 
        if potential_short_name_base in official_short_names and "/" not in model_name:
            hf_model_id = f"openai/whisper-{model_name}" # Reconstructs openai/whisper-tiny.en if model_name was tiny.en
            if verbose: print(f"Interpreted model_name '{model_name}' as Hugging Face ID '{hf_model_id}'")
        
        # This print was from your log, confirming it was already there
        # if verbose: 
        #     print(f"Loading Whisper model '{hf_model_id}' using ipex_llm.transformers "
        #           f"with low_bit config: '{low_bit}' for device '{target_device_str}'")

        try:
            processor = WhisperProcessor.from_pretrained(
                hf_model_id, cache_dir=download_root, token=use_auth_token
            )
        except Exception as e:
            warnings.warn(f"Failed to load WhisperProcessor for '{hf_model_id}': {e}", RuntimeWarning)
            raise

        from_pretrained_args = kwargs.copy() 
        from_pretrained_args.setdefault('trust_remote_code', True)
        from_pretrained_args['cache_dir'] = download_root
        from_pretrained_args['token'] = use_auth_token
        
        if target_device.type == 'xpu':
            from_pretrained_args.setdefault('cpu_embedding', True)
            if verbose: print(f"INFO: Setting cpu_embedding=True for XPU device.")

        normalized_low_bit = low_bit.lower() if isinstance(low_bit, str) else ""
        effective_torch_dtype = "auto" # Default, IPEX-LLM/Transformers can infer

        # Logic for setting IPEX-LLM specific loading args based on `low_bit`
        if normalized_low_bit in ["int4", "4bit"]: 
            if verbose: print(f"INFO: Configuring for INT4: load_in_4bit=True, optimize_model=False (for low_bit='{low_bit}')")
            from_pretrained_args['load_in_4bit'] = True
            from_pretrained_args['optimize_model'] = False 
            effective_torch_dtype = None # Let IPEX-LLM handle dtype for its 4-bit loading
        elif normalized_low_bit in ["bf16", "fp16", "sym_int8", "woq_int4"]: # Add other specific IPEX-LLM strings
            # For these, we use load_in_low_bit="STRING" and optimize_model=True
            if verbose: print(f"INFO: Configuring for specific low_bit: load_in_low_bit='{low_bit}', optimize_model=True")
            from_pretrained_args['load_in_low_bit'] = low_bit 
            from_pretrained_args['optimize_model'] = True  

            if normalized_low_bit == "bf16": effective_torch_dtype = torch.bfloat16
            elif normalized_low_bit == "fp16":
                effective_torch_dtype = torch.float16
                if target_device.type == 'cpu': warnings.warn("FP16 may not be optimal on CPU.", UserWarning)
            else: # For sym_int8, woq_int4, etc.
                effective_torch_dtype = None # Let IPEX-LLM handle dtype
        elif normalized_low_bit in ["fp32", "none", ""]: # Explicit FP32 or no quantization via IPEX-LLM
            if verbose: print(f"INFO: Configuring for FP32 (no IPEX-LLM low-bit conversion), optimize_model=True for general IPEX optimizations.")
            if 'load_in_low_bit' in from_pretrained_args: from_pretrained_args.pop('load_in_low_bit')
            if 'load_in_4bit' in from_pretrained_args: from_pretrained_args.pop('load_in_4bit')
            from_pretrained_args['optimize_model'] = True # Allow general IPEX graph optimization
            effective_torch_dtype = torch.float32
        else: 
            warnings.warn(f"Unrecognized low_bit value '{low_bit}'. Attempting to pass as is to "
                          f"load_in_low_bit with optimize_model=True.", UserWarning)
            from_pretrained_args['load_in_low_bit'] = low_bit 
            from_pretrained_args['optimize_model'] = True
            effective_torch_dtype = None


        if effective_torch_dtype is not None and effective_torch_dtype != "auto":
            from_pretrained_args['torch_dtype'] = effective_torch_dtype
        elif 'torch_dtype' in from_pretrained_args and effective_torch_dtype is None:
             from_pretrained_args.pop('torch_dtype', None) # Remove if it was set by kwargs but now overridden

        if verbose:
            print(f"DEBUG: Final from_pretrained arguments for model loading: {from_pretrained_args}")

        try:
            model_instance = AutoModelForSpeechSeq2Seq.from_pretrained(hf_model_id, **from_pretrained_args)
        except Exception as e:
            # Catching the error here to provide more context before re-raising
            detailed_error_msg = f"Failed to load model '{hf_model_id}' with IPEX-LLM using args {from_pretrained_args}. Original error: {e}"
            if verbose:
                import traceback
                print("--- TRACEBACK FOR MODEL LOAD FAILURE ---")
                traceback.print_exc()
                print("--------------------------------------")
            warnings.warn(detailed_error_msg, RuntimeWarning)
            raise RuntimeError(detailed_error_msg) from e # Re-raise to stop execution

        model_instance = model_instance.eval()
        try:
            model_instance = model_instance.to(target_device)
            loaded_model_dtype = next(model_instance.parameters()).dtype
            # This log was in your successful output, keep it.
            print(f"IPEX-LLM Whisper model '{hf_model_id}' loaded. Target device: '{target_device}'. "
                  f"Effective model dtype: {loaded_model_dtype}. Low-bit config: '{low_bit}'.")
        except Exception as e:
            warnings.warn(f"Failed to move model to '{target_device}': {e}. Using device: {model_instance.device}", RuntimeWarning)
            target_device = model_instance.device

        return cls(hf_model_id, model_instance, processor, target_device, low_bit_format=low_bit, verbose=verbose)

    def _get_transcribe_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Prepare keyword arguments for model.generate(), mapping from Scraibe's kwargs.
        Prioritizes parameters relevant to WhisperForConditionalGeneration's long-form processing.
        """
        generate_params = {}
        
        # Parameters directly from the `whisper_generate` signature (transformers/models/whisper/generation_whisper.py)
        # and common GenerationConfig parameters.
        known_generate_options = [
            "language", "task", "temperature", 
            "compression_ratio_threshold", "logprob_threshold", "no_speech_threshold", 
            "condition_on_prev_tokens", "initial_prompt", "prompt_ids", "decoder_input_ids",
            "return_timestamps", "return_token_timestamps", 
            "num_beams", "patience", "length_penalty", "repetition_penalty",
            "no_repeat_ngram_size", "suppress_tokens", "begin_suppress_tokens",
            "max_length", "max_new_tokens", 
            "use_cache", "do_sample", "top_k", "top_p",
            "return_dict_in_generate", "attention_mask", # attention_mask is passed explicitly to generate
            "is_multilingual", "num_segment_frames", "time_precision",
            "time_precision_features", "return_segments", "force_unique_generate_call",
            "prompt_condition_type"
        ]

        for k, v in kwargs.items():
            if k in known_generate_options:
                generate_params[k] = v
            elif k == "beam_size" and "num_beams" not in generate_params: # Map CLI --beam_size if used
                 generate_params["num_beams"] = v
            elif k == "word_timestamps" and v is True: # User explicitly wants word timestamps
                 generate_params["return_token_timestamps"] = True
                 generate_params.setdefault("return_timestamps", True) # Usually need both

        # --- Sensible Defaults for robust long-form transcription via model.generate ---
        generate_params.setdefault('language', None) 
        generate_params.setdefault('task', "transcribe")
        
        generate_params.setdefault('return_timestamps', True) 
        generate_params.setdefault('return_dict_in_generate', True) 
        generate_params.setdefault('condition_on_prev_tokens', True)
        
        # Temperature: The model's generate() can handle a tuple for its internal fallback logic.
        # The ValueError about "short-form... temperature tuple" should not occur if input_features
        # are for the full audio, correctly signaling long-form mode to generate().
        generate_params.setdefault('temperature', (0.0, 0.2, 0.4, 0.6, 0.8, 1.0))
        
        # Default to greedy if num_beams not set or is 1. User can override with --num-beams.
        generate_params.setdefault('num_beams', 1) 
        if generate_params['num_beams'] > 1:
            generate_params['do_sample'] = False # Beam search and sampling are typically exclusive
        
        generate_params.setdefault('use_cache', True)

        # To silence the "temperature is set to None/0.0 with do_sample=False" warning for pure greedy.
        # The model's generate should handle this correctly, but the warning can be noisy.
        current_temp_for_logic = generate_params.get('temperature')
        first_temp_val = current_temp_for_logic[0] if isinstance(current_temp_for_logic, (list,tuple)) and current_temp_for_logic else current_temp_for_logic
        if generate_params.get('num_beams',1) == 1 and \
           not generate_params.get('do_sample', False) and \
           (first_temp_val == 0.0):
            # For greedy, can pop temp. If model needs it as 0.0, it will use default or handle None.
            generate_params.pop('temperature', None) 

        # max_new_tokens/max_length: Best to let the model's internal long-form chunking manage this.
        # Overriding at top level might be too restrictive or too loose.
        # The HF ASR pipeline uses max_new_tokens for *its* chunks when calling generate.
        generate_params.pop('max_new_tokens', None) 
        generate_params.pop('max_length', None) 

        return generate_params

    def transcribe(self, audio: Union[str, torch.Tensor, np.ndarray], **kwargs: Any) -> Dict[str, Any]:
        """
        Transcribe audio using the IPEX-LLM loaded Whisper model's internal
        long-form processing capabilities by calling model.generate() once.
        """
        if not self.processor or not self.model:
            raise RuntimeError("Transcriber is not properly initialized with model and processor.")

        current_call_verbose = kwargs.get("verbose", self.verbose)

        if isinstance(audio, str):
            raise NotImplementedError("This method expects a pre-processed waveform Tensor from Scraibe's AudioProcessor.")
        elif isinstance(audio, np.ndarray):
            audio_waveform = torch.from_numpy(audio.astype(np.float32))
        elif isinstance(audio, torch.Tensor):
            audio_waveform = audio.to(torch.float32)
        else:
            raise TypeError(f"Expected audio to be Tensor or ndarray, got {type(audio)}")

        if audio_waveform.ndim > 1: audio_waveform = audio_waveform.squeeze(0)
        if audio_waveform.ndim != 1: raise ValueError(f"Audio waveform must be 1D, got {audio_waveform.ndim} dims.")

        if current_call_verbose: 
            print(f"Processing full audio waveform (duration: {audio_waveform.shape[0]/SAMPLE_RATE:.2f}s) "
                  f"with self.processor...")
        
        try:
            # Process the *entire* audio, ensure no truncation, get attention_mask.
            inputs = self.processor(
                audio_waveform.cpu().numpy(), 
                sampling_rate=SAMPLE_RATE, 
                return_tensors="pt",
                return_attention_mask=True, 
                truncation=False, # CRITICAL: Do not truncate for long audio
                # padding="longest" # Optional for single item, but good if processor expects it with truncation=False
                                  # Let's test without it first if truncation=False is enough.
                                  # If processor complains about non-rectangular batch for single long item, add padding=True
            )
            input_features = inputs.input_features
            attention_mask = inputs.get("attention_mask")

            if current_call_verbose:
                print(f"Shape of full input_features from processor: {input_features.shape}")
                if attention_mask is not None: print(f"Shape of full attention_mask: {attention_mask.shape}")
                
                # Sanity check feature length against audio duration
                expected_frames_approx = int((audio_waveform.shape[0] / SAMPLE_RATE) * (SAMPLE_RATE / (self.processor.feature_extractor.hop_length if self.processor else 160)))
                if abs(input_features.shape[-1] - expected_frames_approx) > 100: # Allow some margin
                     warnings.warn(f"Processor returned {input_features.shape[-1]} frames for a "
                                   f"{audio_waveform.shape[0]/SAMPLE_RATE:.2f}s audio (expected ~{expected_frames_approx}). "
                                   "If significantly less, long-form transcription might be incomplete. "
                                   "Ensure `truncation=False` is effective.", UserWarning)
        except Exception as e:
            warnings.warn(f"Error during full audio feature extraction: {e}", RuntimeWarning)
            raise

        # Move features to target device and ensure correct dtype
        model_dtype = next(self.model.parameters()).dtype
        try:
            if input_features.device != self.target_device:
                input_features = input_features.to(self.target_device)
            if input_features.dtype != model_dtype:
                if current_call_verbose: print(f"Casting full_input_features from {input_features.dtype} to model dtype {model_dtype}")
                input_features = input_features.to(model_dtype)
            if attention_mask is not None and attention_mask.device != self.target_device:
                attention_mask = attention_mask.to(self.target_device)
        except Exception as e:
             warnings.warn(f"Could not move/cast input_features/attention_mask: {e}", RuntimeWarning)

        # Prepare generate_options using the helper, passing through kwargs from this transcribe call
        generate_options = self._get_transcribe_kwargs(**kwargs)
        
        # Language for the final returned dict. model.generate might also return detected lang.
        final_language_to_report = generate_options.get("language") 
        if final_language_to_report is None and self.model.config.is_multilingual:
            if current_call_verbose: print("Language not specified; model will attempt auto-detection.")
        elif final_language_to_report is None: # Not multilingual, default to en
            final_language_to_report = "en"
            generate_options['language'] = "en" # Ensure it's passed if model is en-only

        if current_call_verbose:
            print(f"Calling self.model.generate() for full audio. Options: {generate_options}")
            print(f"Model device: {self.model.device}, Input features: {input_features.shape}, {input_features.dtype}, {input_features.device}")
            if attention_mask is not None: print(f"Attention mask: {attention_mask.shape}, {attention_mask.device}")

        full_text = ""
        segments_data = []

        with torch.no_grad():
            try:
                # Using `input_features=` explicitly as per deprecation warning for `inputs=`
                output = self.model.generate(
                    input_features=input_features, 
                    attention_mask=attention_mask,
                    **generate_options 
                )

                if self.target_device.type == "xpu":
                    torch.xpu.synchronize()
                
                # Process output from generate
                predicted_ids = output.sequences if hasattr(output, "sequences") else output
                full_text = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
                
                raw_segments = None
                # The actual key for segments in the output of model.generate when return_timestamps=True
                # for WhisperForConditionalGeneration is often not directly "segments" or "chunks" in the
                # top-level output object unless it's a specialized GenerationOutput class that adds it.
                # More commonly, you get token IDs and need to process them if return_token_timestamps=True.
                # Or, the `generate` method's *internal* `final_segments` (as in generation_whisper.py)
                # would need to be exposed, which isn't standard for the base .generate().
                # The HF ASR *Pipeline* constructs these "chunks".
                # For now, let's check if `output` itself is a list of segments (if patched) or has a common key.
                
                # If `return_timestamps="word"` or `return_token_timestamps=True` was used effectively,
                # the output might contain detailed token timestamps.
                # This part requires careful inspection of the actual `output` object.
                # The `generation_whisper.py` shows `final_segments` being constructed internally.
                # If `return_dict_in_generate=True`, `output` is a `WhisperGenerationOutput` which might have them.
                if hasattr(output, "김포"): # This is a placeholder for the actual attribute name for segments
                    # This is highly speculative and depends on the exact GenerationOutput structure
                    # when return_timestamps=True is active.
                    # Example based on a possible structure similar to HF pipeline chunks:
                    # raw_segments = getattr(output, "김포", None) # Replace "김포" with actual key
                    pass # Need to know actual output structure here

                if raw_segments and isinstance(raw_segments, list): # Placeholder
                    for i, seg_data in enumerate(raw_segments):
                        segments_data.append({
                            "id": i, "seek": seg_data.get("seek", 0), 
                            "start": round(float(seg_data.get("start", 0.0)), 3), 
                            "end": round(float(seg_data.get("end", 0.0)), 3),
                            "text": seg_data.get("text", "").strip(), "tokens": seg_data.get("tokens", []),
                        })
                    if current_call_verbose: print(f"Extracted {len(segments_data)} segments.")
                elif full_text: 
                    warnings.warn("Detailed segment extraction from model.generate() output is not fully implemented "
                                  "for this transcriber or was not available in output. Creating a single segment.", UserWarning)
                    segments_data.append({
                        "id": 0, "seek": 0, "start": 0.0, "end": round(audio_waveform.shape[0]/SAMPLE_RATE, 3), 
                        "text": full_text, "tokens": predicted_ids.squeeze().tolist() if isinstance(predicted_ids, torch.Tensor) else []
                    })

                # Language
                if hasattr(output, "language"): final_language_to_report = output.language
                elif final_language_to_report is None: final_language_to_report = "en"

                if current_call_verbose:
                    print(f"--- Transcription ---")
                    print(f"Language: {final_language_to_report}")
                    print(full_text)
                    if segments_data:
                        for seg in segments_data[:3]: print(f"  Segment Preview: [{seg['start']:.3f} -> {seg['end']:.3f}] {seg['text'][:100]}...")
                    print(f"--- End Transcription ---")

            except Exception as e:
                warnings.warn(f"Error during model.generate() or decoding: {e}", RuntimeWarning)
                import traceback; traceback.print_exc()
                final_lang_err = final_language_to_report if 'final_language_to_report' in locals() and final_language_to_report is not None else kwargs.get("language", "en")
                return {"text": "", "segments": [], "language": final_lang_err}
        
        final_lang_ret = final_language_to_report if 'final_language_to_report' in locals() and final_language_to_report is not None else kwargs.get("language", "en")
        return {"text": full_text, "segments": segments_data, "language": final_lang_ret}



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