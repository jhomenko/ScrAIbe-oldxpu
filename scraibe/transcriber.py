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
#CHUNK_LENGTH = 30 # seconds
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
    warnings.warn("Could not import from .misc. Using placeholder constants.", UserWarning)
    WHISPER_DEFAULT_PATH = None
    SCRAIBE_TORCH_DEVICE = "cpu"
    SCRAIBE_NUM_THREADS = 4


# Using a generic TypeVar for the model instance in the ABC
ModelType = TypeVar('ModelType')


class Transcriber(ABC): # Your existing ABC
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
        if not text_content and "segments" in transcript_data:
            text_content = " ".join(seg.get("text", "") for seg in transcript_data["segments"]).strip()
        
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(text_content)
            if os.path.exists(save_path) and os.path.getsize(save_path) > 0 or not text_content:
                print(f'Transcript text saved to {save_path}')
            else:
                print(f'Warning: Transcript text file at {save_path} might be empty or was not saved correctly.')
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
    Transcriber for OpenAI's Whisper model, using ipex_llm.transformers for loading
    and optimization. Relies on the model's internal long-form transcription capabilities.
    """
    def __init__(self, 
                 model_name: str, 
                 # Model instance can now be either IPEX-LLM AutoModel or IPEX-optimized HF model
                 model_instance: Union[AutoModelForSpeechSeq2Seq, WhisperForConditionalGeneration], 
                 processor_instance: WhisperProcessor,
                 target_device: torch.device,
                 low_bit_format: Optional[str] = None,
                 verbose: bool = False):
        super().__init__(model_name, model_instance, processor_instance, verbose=verbose)
        self.target_device = target_device
        self.low_bit_format = low_bit_format

    @classmethod
    def load_model(cls,
                   model_name: str = "medium",
                   download_root: Optional[str] = None, 
                   device_option: Optional[Union[str, torch.device]] = None,
                   use_ipex_llm: bool = True, # This flag now means "try IPEX-LLM specific loading first"
                   low_bit: str = 'bf16', 
                   use_auth_token: Optional[str] = None,
                   verbose: bool = False,
                   **kwargs: Any 
                   ) -> 'OpenAIWhisperIPEXLLMTranscriber':

        target_device_str = str(device_option if device_option else SCRAIBE_TORCH_DEVICE)
        target_device = torch.device(target_device_str)

        hf_model_id = model_name
        official_short_names = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
        potential_short_name = model_name.replace(".en", "")
        if potential_short_name in official_short_names and "/" not in model_name:
            hf_model_id = f"openai/whisper-{model_name}"
            if verbose: print(f"Interpreted model_name '{model_name}' as Hugging Face ID '{hf_model_id}'")

        if verbose:
            print(f"Loading Whisper model '{hf_model_id}' for device '{target_device_str}' with low_bit config: '{low_bit}'")

        try:
            processor = WhisperProcessor.from_pretrained(
                hf_model_id, cache_dir=download_root, token=use_auth_token
            )
        except Exception as e:
            warnings.warn(f"Failed to load WhisperProcessor for '{hf_model_id}': {e}", RuntimeWarning)
            raise

        model_instance = None
        normalized_low_bit = low_bit.lower() if isinstance(low_bit, str) else ""
        
        # --- Strategy: Try IPEX-LLM loader first for actual low-bit (int4, specific strings).
        # --- For fp16/bf16, try standard HF load + ipex.optimize() if IPEX is available.
        
        use_standard_hf_plus_ipex_optimize = False
        torch_dtype_for_hf_load = "auto" # Default for standard HF loading

        if normalized_low_bit in ["bf16", "fp16", "float16"]:
            use_standard_hf_plus_ipex_optimize = True
            if normalized_low_bit == "bf16":
                torch_dtype_for_hf_load = torch.bfloat16
            else: # fp16
                torch_dtype_for_hf_load = torch.float16
            if verbose: print(f"INFO: Will attempt to load standard HF model in {normalized_low_bit} and apply ipex.optimize().")
        
        elif use_ipex_llm and ipex_llm is not None and normalized_low_bit not in ["fp32", "none", ""]:
            # Use IPEX-LLM AutoModel for specific low-bit quantizations (e.g., int4, woq_int4)
            if verbose: print(f"INFO: Attempting to load with ipex_llm.transformers for low_bit='{low_bit}'")
            from_pretrained_args = kwargs.copy()
            from_pretrained_args.update({
                'trust_remote_code': from_pretrained_args.get('trust_remote_code', True),
                'cache_dir': download_root, 'token': use_auth_token
            })
            if target_device.type == 'xpu': from_pretrained_args.setdefault('cpu_embedding', False)

            if normalized_low_bit in ["int4", "4bit"]:
                from_pretrained_args.update({'load_in_4bit': True, 'optimize_model': False})
                if 'torch_dtype' in from_pretrained_args: from_pretrained_args.pop('torch_dtype') # Let IPEX-LLM handle
            else: # Specific IPEX-LLM string like "woq_int4", "sym_int4", etc.
                from_pretrained_args.update({'load_in_low_bit': low_bit, 'optimize_model': True})
                if 'torch_dtype' in from_pretrained_args: from_pretrained_args.pop('torch_dtype')

            if verbose: print(f"DEBUG: IPEX-LLM from_pretrained args: {from_pretrained_args}")
            try:
                model_instance = AutoModelForSpeechSeq2Seq.from_pretrained(hf_model_id, **from_pretrained_args)
            except Exception as e:
                warnings.warn(f"Failed to load model '{hf_model_id}' using IPEX-LLM AutoModel: {e}. "
                              "Will attempt standard Hugging Face loading if applicable.", RuntimeWarning)
                if normalized_low_bit in ["bf16", "fp16", "float16"]: # Fallback to standard HF + IPEX optimize
                    use_standard_hf_plus_ipex_optimize = True
                else:
                    raise # If it was a specific IPEX-LLM low-bit string and failed, re-raise
        else: # Default to standard HF loading (e.g., for fp32 or if IPEX-LLM not used/available)
            use_standard_hf_plus_ipex_optimize = True
            if normalized_low_bit == "bf16": torch_dtype_for_hf_load = torch.bfloat16
            elif normalized_low_bit == "fp16": torch_dtype_for_hf_load = torch.float16
            else: torch_dtype_for_hf_load = torch.float32 # Default to FP32 for standard load if no low_bit match

        if use_standard_hf_plus_ipex_optimize:
            if verbose: print(f"INFO: Loading standard Hugging Face model '{hf_model_id}' with torch_dtype='{torch_dtype_for_hf_load}'.")
            try:
                model_instance = WhisperForConditionalGeneration.from_pretrained(
                    hf_model_id,
                    torch_dtype=torch_dtype_for_hf_load,
                    cache_dir=download_root,
                    token=use_auth_token,
                    trust_remote_code=kwargs.get('trust_remote_code', True)
                )
                if verbose: print("Standard HF model loaded. Applying ipex.optimize() if available and on XPU.")
                
                if target_device.type == 'xpu' and ipex is not None:
                    try:
                        if verbose: print(f"INFO: Applying ipex.optimize(model, dtype={torch_dtype_for_hf_load}) for XPU.")
                        # For ipex.optimize, cpu_embedding is not a direct param.
                        # It's more about optimizing the existing model structure for XPU.
                        model_instance = ipex.optimize(model_instance.eval(), dtype=torch_dtype_for_hf_load, inplace=True, weights_prepack=False)
                        if hasattr(model_instance, 'to'): # Optimized model should still have .to
                             model_instance = model_instance.to(target_device) # Move after optimize
                        else: # if ipex.optimize changes model type significantly
                             warnings.warn("Model type changed after ipex.optimize, cannot call .to(device). Assuming it's on target device or CPU.", UserWarning)
                             model_instance.eval() # Ensure eval modebatch_decode
                        
                        # If cpu_embedding was true for XPU, and we used standard HF load,
                        # one might need to manually move embedding layers back to CPU if ipex.optimize put them on XPU.
                        # This is complex. For now, let's assume ipex.optimize handles dtypes and device placement appropriately.
                        if kwargs.get('cpu_embedding', False) and target_device.type == 'xpu':
                             warnings.warn("cpu_embedding=True was requested, but after standard HF load + ipex.optimize, "
                                           "manual movement of embeddings to CPU is not implemented here. Embeddings will be on model device.", UserWarning)

                    except Exception as e_ipex:
                        warnings.warn(f"ipex.optimize() failed: {e_ipex}. Using model without this IPEX optimization.", RuntimeWarning)
                        if hasattr(model_instance, 'to'): model_instance = model_instance.to(target_device) # Still try to move original
                        model_instance.eval()
                else: # Not XPU or IPEX not available
                    if hasattr(model_instance, 'to'): model_instance = model_instance.to(target_device)
                    model_instance.eval()

            except Exception as e_hf_load:
                warnings.warn(f"Failed to load model '{hf_model_id}' using standard Hugging Face loader: {e_hf_load}", RuntimeWarning)
                raise

        if model_instance is None:
            raise RuntimeError(f"Model instance could not be created for '{hf_model_id}' with config '{low_bit}'.")

        model_instance = model_instance.eval() # Final ensure eval mode
        
        # Final check and move to device, though should be done by loaders/optimizer
        try:
            if model_instance.device != target_device and hasattr(model_instance, 'to'):
                model_instance = model_instance.to(target_device)
            actual_model_dtype = next(model_instance.parameters()).dtype
            if verbose:
                print(f"Final Whisper model '{hf_model_id}' ready. Device: '{model_instance.device}'. "
                      f"Dtype: {actual_model_dtype}. Original low_bit request: '{low_bit}'.")
        except Exception as e_final_move:
            warnings.warn(f"Error during final model device placement/check: {e_final_move}", UserWarning)


        return cls(hf_model_id, model_instance, processor, model_instance.device, low_bit_format=low_bit, verbose=verbose)

    def _get_transcribe_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        # ... (as provided in my previous response, starting with generate_params = {})
        # ... (it sets return_timestamps=True, return_dict_in_generate=True, temperature tuple etc.)
        # ... (and handles num_beams, aiming to pass a float temperature if greedy for the ValueError)
        # --- Start copy of _get_transcribe_kwargs from previous correct response ---
        generate_params = {}
        known_options = [
            "language", "task", "temperature", "compression_ratio_threshold", 
            "logprob_threshold", "no_speech_threshold", "condition_on_prev_tokens",
            "initial_prompt", "prompt_ids", "decoder_input_ids",
            "return_timestamps", "return_token_timestamps", 
            "num_beams", "patience", "length_penalty", "repetition_penalty",
            "no_repeat_ngram_size", "suppress_tokens", "begin_suppress_tokens",
            "max_length", "max_new_tokens", 
            "use_cache", "do_sample", "top_k", "top_p",
            "return_dict_in_generate", "attention_mask",
            "is_multilingual", "num_segment_frames", "time_precision",
            "time_precision_features", "return_segments", "force_unique_generate_call",
            "prompt_condition_type"
        ]
        for k, v in kwargs.items():
            if k in known_options:
                generate_params[k] = v
            elif k == "beam_size" and "num_beams" not in generate_params:
                 generate_params["num_beams"] = v
            elif k == "word_timestamps" and v is True:
                 generate_params["return_token_timestamps"] = True
                 generate_params.setdefault("return_timestamps", True) 

        generate_params.setdefault('language', None) 
        generate_params.setdefault('task', "transcribe")
        generate_params.setdefault('return_timestamps', True) 
        generate_params.setdefault('return_dict_in_generate', True) 
        generate_params.setdefault('condition_on_prev_tokens', True)
        
        default_temp_tuple = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
        user_temp_setting = kwargs.get('temperature', default_temp_tuple)
        
        # To address the ValueError: "temperature cannot be set to tuple for short-form"
        # We pass only the *first* temperature from the desired sequence to the main generate call.
        # The model's internal long-form logic (if robust) should use other thresholds for quality.
        if isinstance(user_temp_setting, (list, tuple)) and user_temp_setting:
            generate_params['temperature'] = float(user_temp_setting[0])
        elif isinstance(user_temp_setting, (float, int)):
            generate_params['temperature'] = float(user_temp_setting)
        else: # Should not happen if default is tuple
            generate_params['temperature'] = 0.0 
            
        generate_params.setdefault('num_beams', 1) # Default to greedy from CLI if not set
        if generate_params['num_beams'] > 1:
            generate_params['do_sample'] = False
            # If beam searching, ensure temp is low (e.g. 0.0)
            if generate_params.get('temperature', 0.0) > 0.1: # Allow small non-zero for beam
                 warnings.warn(f"Beam search (num_beams={generate_params['num_beams']}) is active, "
                               f"but temperature is {generate_params['temperature']}. Consider temp<=0.1 for beam search.", UserWarning)
        elif generate_params.get('do_sample', False): # Explicit sampling
             pass # Use temperature as set (which is now a float)
        else: # Greedy (num_beams=1, do_sample=False implicitly or explicitly)
            generate_params['do_sample'] = False
            if generate_params.get('temperature') == 0.0: # Pure greedy
                 generate_params.pop('temperature', None) # Remove to silence warning

        generate_params.setdefault('use_cache', True)
        generate_params.pop('max_new_tokens', None)
        return generate_params
        # --- End copy of _get_transcribe_kwargs ---

    def transcribe(self, audio: Union[str, torch.Tensor, np.ndarray], **kwargs: Any) -> Dict[str, Any]:
        # ... (This method should be the one that calls self.processor once with truncation=False,
        #      then self.model.generate once with full features and options from _get_transcribe_kwargs.
        #      As per my previous full class example that aimed to fix the "short-form" ValueError)
        # --- Start copy of transcribe from previous correct response ---
        if not self.processor or not self.model:
            raise RuntimeError("Transcriber is not properly initialized with a model and processor.")

        current_call_verbose = kwargs.get("verbose", self.verbose)

        if isinstance(audio, str):
            raise NotImplementedError("This method expects a pre-processed waveform Tensor.")
        elif isinstance(audio, np.ndarray):
            audio_waveform = torch.from_numpy(audio.astype(np.float32))
        elif isinstance(audio, torch.Tensor):
            audio_waveform = audio.to(torch.float32)
        else:
            raise TypeError(f"Expected audio to be Tensor or ndarray, got {type(audio)}")

        if audio_waveform.ndim > 1: audio_waveform = audio_waveform.squeeze(0)
        if audio_waveform.ndim != 1: raise ValueError(f"Audio waveform must be 1D, got {audio_waveform.ndim} dims.")

        if current_call_verbose: 
            print(f"Processing full audio waveform (duration: {audio_waveform.shape[0]/SAMPLE_RATE:.2f}s) with self.processor...")
        
        try:
            inputs = self.processor(
                audio_waveform.cpu().numpy(), 
                sampling_rate=SAMPLE_RATE, 
                return_tensors="pt",
                return_attention_mask=True,
                truncation=False, 
                padding="longest" 
            )
            input_features = inputs.input_features
            attention_mask = inputs.get("attention_mask")

            if current_call_verbose:
                print(f"Shape of full input_features from processor: {input_features.shape}")
                if attention_mask is not None:
                    print(f"Shape of full attention_mask from processor: {attention_mask.shape}")
                expected_min_frames = int((audio_waveform.shape[0] / SAMPLE_RATE) * (SAMPLE_RATE / (HOP_LENGTH if 'HOP_LENGTH' in globals() and HOP_LENGTH > 0 else 160))) - 100
                if input_features.shape[-1] < expected_min_frames and input_features.shape[-1] <= 3000 :
                     warnings.warn(f"Processor returned only {input_features.shape[-1]} frames for a "
                                   f"{audio_waveform.shape[0]/SAMPLE_RATE:.2f}s audio. "
                                   "Long-form transcription might fail or be incomplete.", UserWarning)
        except Exception as e:
            warnings.warn(f"Error during full audio feature extraction: {e}", RuntimeWarning)
            raise

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

        generate_options = self._get_transcribe_kwargs(**kwargs)
        final_language_to_report = generate_options.get("language", "en") 

        if current_call_verbose:
            print(f"Calling self.model.generate() for full audio. Options: {generate_options}")
            print(f"Model device: {self.model.device}, Input features: {input_features.shape}, {input_features.dtype}, {input_features.device}")
            if attention_mask is not None: print(f"Attention mask: {attention_mask.shape}, {attention_mask.device}")

        full_text = ""
        segments_data = []

        with torch.no_grad():
            try:
                output = self.model.generate(
                    input_features=input_features, 
                    attention_mask=attention_mask,
                    **generate_options 
                )

                if self.target_device.type == "xpu":
                    torch.xpu.synchronize()
                
                predicted_ids = output.sequences if hasattr(output, "sequences") else output
                full_text = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
                
                raw_segments = None
                if hasattr(output, "segments") and output.segments is not None: raw_segments = output.segments
                elif hasattr(output, "chunks") and output.chunks is not None: raw_segments = output.chunks
                
                if raw_segments and isinstance(raw_segments, list):
                    for i, seg_data in enumerate(raw_segments):
                        text = seg_data.get("text", "").strip()
                        ts = seg_data.get("timestamp", (0.0, 0.0))
                        start_time = ts[0] if isinstance(ts, (list, tuple)) and len(ts) > 0 else 0.0
                        end_time = ts[1] if isinstance(ts, (list, tuple)) and len(ts) > 1 and ts[1] is not None else start_time 
                        segments_data.append({
                            "id": i, "seek": seg_data.get("seek", int(float(start_time) * SAMPLE_RATE / (HOP_LENGTH if 'HOP_LENGTH' in globals() and HOP_LENGTH > 0 else 160))), 
                            "start": round(float(start_time), 3), "end": round(float(end_time), 3),
                            "text": text, "tokens": seg_data.get("tokens", []),
                        })
                    if current_call_verbose: print(f"Extracted {len(segments_data)} segments from model output.")
                elif full_text: 
                    warnings.warn("Detailed segments not found. Creating single segment.", UserWarning)
                    segments_data.append({
                        "id": 0, "seek": 0, "start": 0.0, "end": round(audio_waveform.shape[0]/SAMPLE_RATE, 3), 
                        "text": full_text, "tokens": predicted_ids.squeeze().tolist() if isinstance(predicted_ids, torch.Tensor) else []
                    })

                if hasattr(output, "language"): final_language_to_report = output.language
                elif final_language_to_report is None: final_language_to_report = "en"

                if current_call_verbose:
                    print(f"--- Transcription ---")
                    print(f"Language: {final_language_to_report}")
                    print(full_text) # (Segment printing logic)
                    print(f"--- End Transcription ---")

            except Exception as e:
                warnings.warn(f"Error during model.generate() or decoding: {e}", RuntimeWarning)
                import traceback; traceback.print_exc()
                final_lang_err = final_language_to_report if 'final_language_to_report' in locals() and final_language_to_report is not None else kwargs.get("language", "en")
                return {"text": "", "segments": [], "language": final_lang_err}
        
        final_lang_ret = final_language_to_report if 'final_language_to_report' in locals() and final_language_to_report is not None else kwargs.get("language", "en")
        return {"text": full_text, "segments": segments_data, "language": final_lang_ret}


# New InsanelyFastWhisperTranscriber implementation based on example repo logic
# Re-implement InsanelyFastWhisperTranscriber natively based on example repo
class InsanelyFastWhisperTranscriber(Transcriber):
    """
    Transcriber for Insanely Fast Whisper models using OpenVINO and optimum-intel.
    Re-implemented natively based on the example repo.
    """
    def __init__(self, model_name: str, model_instance: Any, verbose: bool = False) -> None:
        super().__init__(model_name, model_instance, verbose=verbose)
        self.verbose = verbose
        self.current_model_size = None
        self.current_compute_type = None
        self.model = None

    @classmethod
    def load_model(cls,
                   model_name: str = "tiny",
                   download_root: Optional[str] = None,
                   device_option: Optional[Union[str, torch.device]] = None,
                   verbose: bool = False,
                   **kwargs: Any
                   ) -> 'InsanelyFastWhisperTranscriber':
        import os
        from transformers import pipeline
        from transformers.utils import is_flash_attn_2_available
        from rich.progress import Progress, TimeElapsedColumn, BarColumn, TextColumn
        from huggingface_hub import hf_hub_download

        # Determine device string for OpenVINO
        target_device_str = str(device_option if device_option else SCRAIBE_TORCH_DEVICE).lower()
        if target_device_str == "xpu":
            ov_device = "xpu"
        elif target_device_str == "cuda":
            ov_device = "cuda"
        else:
            ov_device = "cpu"

        if verbose:
            print(f"Loading Insanely Fast Whisper model '{model_name}' on device '{ov_device}'")

        model_dir = download_root if download_root else "insanely_fast_whisper_models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, model_name)

        # Download model files if not present
        if not os.path.isdir(model_path) or not os.listdir(model_path):
            if verbose: print(f"Downloading model files for '{model_name}' to '{model_path}'")
            download_list = [
                "model.safetensors",
                "config.json",
                "generation_config.json",
                "preprocessor_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "added_tokens.json",
                "special_tokens_map.json",
                "vocab.json",
            ]
            repo_id = f"openai/whisper-{model_name}" if not model_name.startswith("distil") else f"distil-whisper/{model_name}"
            for item in download_list:
                hf_hub_download(repo_id=repo_id, filename=item, local_dir=model_path)

        compute_type = kwargs.get("compute_type", "int8")

        # Setup model kwargs
        model_kwargs = {
            # Remove torch_dtype or set to "auto" to avoid int8 error
            "device": ov_device,
            "model_kwargs": {"attn_implementation": "flash_attention_2"} if is_flash_attn_2_available() else {"attn_implementation": "sdpa"},
        }

        # Load pipeline
        try:
            model_instance = pipeline(
                "automatic-speech-recognition",
                model=model_path,
                **model_kwargs
            )
            if model_instance is None:
                raise RuntimeError("Pipeline creation returned None")
        except Exception as e:
            if verbose:
                print(f"Error loading pipeline: {e}")
            raise RuntimeError(f"Failed to load pipeline for model '{model_name}': {e}")

        # Create instance
        transcriber = cls(model_name, model_instance, verbose=verbose)
        transcriber.current_model_size = model_name
        transcriber.current_compute_type = compute_type
        return transcriber

    def transcribe(self, audio: Union[str, np.ndarray, torch.Tensor], **kwargs: Any) -> Dict[str, Any]:
        if self.verbose:
            print(f"Transcribing audio with InsanelyFastWhisperTranscriber, model '{self.model_name}'")

        # Extract parameters from kwargs
        no_speech_threshold = kwargs.get("no_speech_threshold", 0.6)
        temperature = kwargs.get("temperature", 0.0)
        compression_ratio_threshold = kwargs.get("compression_ratio_threshold", 2.4)
        log_prob_threshold = kwargs.get("log_prob_threshold", -1.0)
        lang = kwargs.get("language", None)
        is_translate = kwargs.get("task", "transcribe") == "translate"
        chunk_length = kwargs.get("chunk_length", 30)
        batch_size = kwargs.get("batch_size", 16)

        # Prepare generate kwargs
        generate_kwargs = {
            "no_speech_threshold": no_speech_threshold,
            "temperature": temperature,
            "compression_ratio_threshold": compression_ratio_threshold,
            "logprob_threshold": log_prob_threshold,
        }
        if lang and not self.current_model_size.endswith(".en"):
            generate_kwargs["language"] = lang
            generate_kwargs["task"] = "translate" if is_translate else "transcribe"

        # Call model pipeline
        segments = self.model(
            inputs=audio,
            return_timestamps=True,
            chunk_length_s=chunk_length,
            batch_size=batch_size,
            generate_kwargs=generate_kwargs
        )

        # Process segments
        segments_result = []
        for item in segments["chunks"]:
            start, end = item["timestamp"][0], item["timestamp"][1]
            if end is None:
                end = start
            segments_result.append({
                "start": start,
                "end": end,
                "text": item["text"]
            })

        full_text = " ".join(item["text"] for item in segments["chunks"]).strip()

        return {
            "text": full_text,
            "segments": segments_result
        }

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
                 warnings.warn(f"Language name '{lang_input}' mapped to '{code}', not in FasterWhisper's known codes. Using '{code}' anyway.")
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
    whisper_type: str = 'faster-whisper', # Changed default to faster-whisper
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

    # Pass all **kwargs down, which includes 'verbose' and other specific loader args
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
    elif whisper_type_lower == 'insanely-fast-whisper':
        # Load InsanelyFastWhisperTranscriber
        return InsanelyFastWhisperTranscriber.load_model(
            model_name=model_name,
            download_root=download_root,
            device_option=target_device,
            **kwargs
        )
    else:
        raise ValueError(
            f"Whisper type '{whisper_type}' not recognized. "
            f"Choose from 'openai-ipex-llm' (or 'whisper'), 'faster-whisper', 'insanely-fast-whisper'."
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
