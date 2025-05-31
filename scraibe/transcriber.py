# scraibe/transcriber.py

from abc import ABC, abstractmethod
from inspect import signature
from typing import TypeVar, Union, Optional, Dict, Any, List, Tuple # Ensure all are here
import torch
from torch import Tensor, device as torch_device # Renamed to avoid conflict with device variables
import numpy as np
import warnings
import os
import yaml # For Diariser.load_model if it still uses it (from user's full file)
from pathlib import Path # For Diariser.load_model

# --- Imports for OpenAIWhisperIPEXLLMTranscriber ---
from ipex_llm.transformers import AutoModelForSpeechSeq2Seq
from transformers import WhisperProcessor, WhisperForConditionalGeneration # Added WhisperForConditionalGeneration
try:
    import intel_extension_for_pytorch as ipex
    print("INFO: Intel Extension for PyTorch (IPEX) library found.")
except ImportError:
    ipex = None
    print("WARNING: Intel Extension for PyTorch (IPEX) library not found. "
          "XPU optimizations via ipex.optimize() for FP16/BF16 float types will not be available.")

# --- Imports for FasterWhisperTranscriber (from your file) ---
from faster_whisper import WhisperModel as FasterWhisperModel
from faster_whisper.tokenizer import _LANGUAGE_CODES as FASTER_WHISPER_LANGUAGE_CODES # Corrected variable name
from whisper.tokenizer import TO_LANGUAGE_CODE as OPENAI_WHISPER_TO_LANGUAGE_CODE # Keep for FasterWhisper lang conversion


# --- Constants ---
SAMPLE_RATE = 16000
# HOP_LENGTH, CHUNK_LENGTH etc. are not strictly needed at module level for the new transcribe method.
# They are used internally by Whisper models/processors.

# IPEX-LLM import (optional, but OpenAIWhisperIPEXLLMTranscriber depends on it)
try:
    import ipex_llm
    # print("INFO: IPEX-LLM library found.") # Already printed if other import succeeds
except ImportError:
    ipex_llm = None # This will be checked in OpenAIWhisperIPEXLLMTranscriber.load_model

# Local project imports
try:
    from .misc import WHISPER_DEFAULT_PATH, SCRAIBE_TORCH_DEVICE, SCRAIBE_NUM_THREADS
except ImportError:
    warnings.warn("Could not import from .misc. Using placeholder constants for transcriber module.", UserWarning)
    WHISPER_DEFAULT_PATH = None
    SCRAIBE_TORCH_DEVICE = "cpu"
    SCRAIBE_NUM_THREADS = 4


ModelType = TypeVar('ModelType')

class Transcriber(ABC):
    def __init__(self, model_name: str, model_instance: ModelType, 
                 processor: Optional[Any] = None, verbose: bool = False) -> None:
        self.model_name = model_name
        self.model = model_instance
        self.processor = processor
        self.verbose = verbose # Added verbose

    @abstractmethod
    def transcribe(self, audio: Union[str, Tensor, np.ndarray], **kwargs) -> Dict[str, Any]:
        pass

    @staticmethod
    def save_transcript(transcript_data: Dict[str, Any], save_path: str) -> None:
        text_content = transcript_data.get("text", "")
        if not text_content and "segments" in transcript_data:
            text_content = " ".join(
                seg.get("text", "").strip() for seg in transcript_data["segments"] if seg.get("text")
            ).strip()
        
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(text_content)
            if os.path.exists(save_path) and (os.path.getsize(save_path) > 0 or not text_content):
                print(f'Transcript text saved to {save_path}')
            else:
                print(f'Warning: Transcript text file at {save_path} appears empty or was not saved correctly.')
        except Exception as e:
            print(f"Error saving transcript to {save_path}: {e}")


    @classmethod
    @abstractmethod
    def load_model(cls,
                   model_name: str,
                   download_root: Optional[str] = WHISPER_DEFAULT_PATH,
                   device_option: Optional[Union[str, torch_device]] = SCRAIBE_TORCH_DEVICE,
                   **kwargs: Any # Must include verbose if __init__ needs it
                   ) -> 'Transcriber':
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_name='{self.model_name}')"


class OpenAIWhisperIPEXLLMTranscriber(Transcriber):
    """
    Transcriber for Whisper model using IPEX-LLM for low-bit integer quantizations,
    or standard Hugging Face Transformers with base IPEX optimization for FP16/BF16 on XPU.
    Relies on the model's internal long-form transcription capabilities.
    """
    def __init__(self, 
                 model_name: str, 
                 model_instance: Union[AutoModelForSpeechSeq2Seq, WhisperForConditionalGeneration],
                 processor_instance: WhisperProcessor,
                 target_device: torch_device,
                 low_bit_format: Optional[str] = None,
                 verbose: bool = False):
        super().__init__(model_name, model_instance, processor_instance, verbose=verbose)
        self.target_device = target_device # Already on target device from load_model
        self.low_bit_format = low_bit_format
        # self.model, self.processor, self.verbose are set by super()

    @classmethod
    def load_model(cls, #... same arguments ...
                   model_name: str = "medium",
                   download_root: Optional[str] = None, 
                   device_option: Optional[Union[str, torch_device]] = None,
                   low_bit: str = 'bf16', # Defaulting to bf16, but you'll test with fp32
                   use_auth_token: Optional[str] = None,
                   verbose: bool = False,
                   **kwargs: Any # Catches trust_remote_code, use_ipex_llm_specific_loader (though we might ignore this one now)
                   ) -> 'OpenAIWhisperIPEXLLMTranscriber':

        target_device_str = str(device_option if device_option else SCRAIBE_TORCH_DEVICE)
        target_device = torch_device(target_device_str)

        hf_model_id = model_name # Assumes mapping to openai/whisper-small happened in CLI or is passed directly
        if verbose: print(f"Attempting to load model '{hf_model_id}' with low_bit='{low_bit}' for device '{target_device_str}'")

        try:
            processor = WhisperProcessor.from_pretrained(
                hf_model_id, cache_dir=download_root, token=use_auth_token
            )
        except Exception as e:
            warnings.warn(f"Failed to load WhisperProcessor for '{hf_model_id}': {e}", RuntimeWarning); raise

        model_instance = None
        normalized_low_bit = low_bit.lower() if isinstance(low_bit, str) else ""

        # Determine loading strategy
        use_ipex_llm_loader_for_int_quant = False
        if normalized_low_bit in ["int4", "4bit", "sym_int4", "asym_int4", "woq_int4", "sym_int8"]:
            if ipex_llm is not None:
                use_ipex_llm_loader_for_int_quant = True
            else:
                warnings.warn(f"IPEX-LLM not found, cannot perform specified INT quantization '{low_bit}'. Falling back.", UserWarning)
                normalized_low_bit = "fp32" # Fallback to fp32 if IPEX-LLM int quant is requested but lib not found

        # Path 1: Standard Transformers + base IPEX.optimize() for FP32, BF16, FP16 on XPU
        if target_device.type == 'xpu' and ipex is not None and \
           (normalized_low_bit in ["fp32", "bf16", "fp16", "float16", "none", ""] or not use_ipex_llm_loader_for_int_quant):
            
            torch_dtype_for_hf_load = torch.float32 
            if normalized_low_bit == "bf16": torch_dtype_for_hf_load = torch.bfloat16
            elif normalized_low_bit == "fp16" or normalized_low_bit == "float16": torch_dtype_for_hf_load = torch.float16
            
            if verbose: print(f"INFO: XPU Path: Loading standard HF model '{hf_model_id}' (dtype: {torch_dtype_for_hf_load}) then ipex.optimize().")
            try:
                model_instance = WhisperForConditionalGeneration.from_pretrained(
                    hf_model_id,
                    torch_dtype=torch_dtype_for_hf_load,
                    cache_dir=download_root, token=use_auth_token,
                    trust_remote_code=kwargs.get('trust_remote_code', True)
                )
                model_instance = model_instance.eval().to(target_device)
                
                # cpu_embedding for this path: IPEX optimize doesn't take it directly.
                # If cpu_embedding=True was intended via CLI's component_kwargs, it's tricky to apply here.
                # We are keeping embeddings on XPU for this ipex.optimize path.
                if kwargs.get('cpu_embedding') and verbose:
                     warnings.warn("cpu_embedding=True from kwargs is not directly applied with ipex.optimize(). Embeddings are on XPU.", UserWarning)

                if verbose: print(f"INFO: Applying ipex.optimize(dtype={torch_dtype_for_hf_load}, weights_prepack=False)")
                model_instance = ipex.optimize(model_instance, dtype=torch_dtype_for_hf_load, inplace=True, weights_prepack=False)
            except Exception as e_hf_ipex:
                warnings.warn(f"Standard HF + ipex.optimize() path failed: {e_hf_ipex}", RuntimeWarning)
                model_instance = None 

        # Path 2: IPEX-LLM AutoModel loader (primarily for INTx, or if Path 1 failed/not taken for floats)
        if model_instance is None and ipex_llm is not None:
            if verbose: print(f"INFO: Fallback/INTx Path: Attempting load via ipex_llm.AutoModel for low_bit='{low_bit}'")
            from_pretrained_args = kwargs.copy()
            from_pretrained_args.update({
                'trust_remote_code': from_pretrained_args.get('trust_remote_code', True),
                'cache_dir': download_root, 'token': use_auth_token
            })
            
            effective_torch_dtype_llm = "auto"
            if normalized_low_bit in ["int4", "4bit"]:
                from_pretrained_args.update({'load_in_4bit': True, 'optimize_model': False})
                effective_torch_dtype_llm = None
                if target_device.type == 'xpu': from_pretrained_args.setdefault('cpu_embedding', False) # Ensure embeddings on XPU for INT4
            elif normalized_low_bit in ["sym_int8", "woq_int4"]: # Add other specific IPEX-LLM INT types
                from_pretrained_args.update({'load_in_low_bit': low_bit, 'optimize_model': True})
                effective_torch_dtype_llm = None
                if target_device.type == 'xpu': from_pretrained_args.setdefault('cpu_embedding', False)
            elif normalized_low_bit in ["bf16", "fp16", "float16"]: # If IPEX-LLM path taken for these
                from_pretrained_args.update({'load_in_low_bit': low_bit, 'optimize_model': True})
                if normalized_low_bit == "bf16": effective_torch_dtype_llm = torch.bfloat16
                else: effective_torch_dtype_llm = torch.float16
                if target_device.type == 'xpu': from_pretrained_args.setdefault('cpu_embedding', True)
            # Omitting FP32 "none" "" path for IPEX-LLM AutoModel unless specific need

            if effective_torch_dtype_llm and effective_torch_dtype_llm != "auto":
                from_pretrained_args['torch_dtype'] = effective_torch_dtype_llm
            elif 'torch_dtype' in from_pretrained_args and effective_torch_dtype_llm is None:
                 from_pretrained_args.pop('torch_dtype', None)
            
            if verbose: print(f"DEBUG: IPEX-LLM AutoModel .from_pretrained() args: {from_pretrained_args}")
            try:
                model_instance = AutoModelForSpeechSeq2Seq.from_pretrained(hf_model_id, **from_pretrained_args)
            except Exception as e_ipex_llm_load:
                warnings.warn(f"IPEX-LLM AutoModel load failed: {e_ipex_llm_load}.", RuntimeWarning)
                model_instance = None

        # Path 3: Final fallback (standard HF model, no IPEX/IPEX-LLM specific optimizations)
        if model_instance is None:
            # ... (your existing fallback logic to load standard WhisperForConditionalGeneration in FP32) ...
             if verbose: print(f"INFO: All optimization paths failed/skipped. Loading standard HF model '{hf_model_id}' in FP32.")
             try:
                model_instance = WhisperForConditionalGeneration.from_pretrained(
                    hf_model_id, torch_dtype=torch.float32,
                    cache_dir=download_root, token=use_auth_token,
                    trust_remote_code=kwargs.get('trust_remote_code', True)
                )
             except Exception as e_final_fallback:
                 warnings.warn(f"Final fallback to standard HF model failed: {e_final_fallback}", RuntimeWarning)
                 raise RuntimeError(f"Could not load model {hf_model_id} through any method.") from e_final_fallback


        # ... (Final model.eval(), .to(target_device), print, return cls(...) as before) ...
        model_instance = model_instance.eval()
        if model_instance.device != target_device and hasattr(model_instance, 'to'):
             model_instance = model_instance.to(target_device)
        
        loaded_model_dtype = next(model_instance.parameters()).dtype
        print(f"Whisper model '{hf_model_id}' loaded. Type: {type(model_instance).__name__}. Target device: '{model_instance.device}'. "
              f"Effective model dtype: {loaded_model_dtype}. Low-bit request: '{low_bit}'.")
        
        return cls(hf_model_id, model_instance, processor, model_instance.device, low_bit_format=low_bit, verbose=verbose)

    def _get_transcribe_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        # This is the _get_transcribe_kwargs from your last version, which aims to pass appropriate
        # parameters to the model's internal long-form generate.
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
            if k in known_options: generate_params[k] = v
            elif k == "beam_size" and "num_beams" not in generate_params: generate_params["num_beams"] = v
            elif k == "word_timestamps" and v is True:
                generate_params["return_token_timestamps"] = True
                generate_params.setdefault("return_timestamps", True) 

        generate_params.setdefault('language', None) 
        generate_params.setdefault('task', "transcribe")
        generate_params.setdefault('return_timestamps', True) 
        generate_params.setdefault('return_dict_in_generate', True) 
        generate_params.setdefault('condition_on_prev_tokens', True)
        
        user_temp_setting = kwargs.get('temperature', (0.0, 0.2, 0.4, 0.6, 0.8, 1.0))
        if isinstance(user_temp_setting, (list, tuple)) and user_temp_setting:
            generate_params['temperature'] = float(user_temp_setting[0]) # Pass first temp as float
        elif isinstance(user_temp_setting, (float, int)):
            generate_params['temperature'] = float(user_temp_setting)
        else: generate_params['temperature'] = 0.0 
            
        generate_params.setdefault('num_beams', 1)
        if generate_params['num_beams'] > 1: generate_params['do_sample'] = False
        
        if generate_params.get('num_beams',1) == 1 and \
           not generate_params.get('do_sample', False) and \
           generate_params.get('temperature') == 0.0:
            generate_params.pop('temperature', None)

        generate_params.setdefault('use_cache', True)
        generate_params.pop('max_new_tokens', None)
        return generate_params

    def transcribe(self, audio: Union[str, torch.Tensor, np.ndarray], **kwargs: Any) -> Dict[str, Any]:
        # This is the transcribe method from your last version that calls model.generate() once
        # and relies on its internal long-form processing.
        if not self.processor or not self.model:
            raise RuntimeError("Transcriber not properly initialized.")

        current_call_verbose = kwargs.get("verbose", self.verbose)

        if isinstance(audio, str): raise NotImplementedError("Expects waveform Tensor.")
        elif isinstance(audio, np.ndarray): audio_waveform = torch.from_numpy(audio.astype(np.float32))
        elif isinstance(audio, torch.Tensor): audio_waveform = audio.to(torch.float32)
        else: raise TypeError(f"Expected Tensor or ndarray, got {type(audio)}")

        if audio_waveform.ndim > 1: audio_waveform = audio_waveform.squeeze(0)
        if audio_waveform.ndim != 1: raise ValueError(f"Audio waveform must be 1D.")

        if current_call_verbose: 
            print(f"Processing full audio waveform (duration: {audio_waveform.shape[0]/SAMPLE_RATE:.2f}s) with processor...")
        
        try:
            inputs = self.processor(
                audio_waveform.cpu().numpy(), sampling_rate=SAMPLE_RATE, 
                return_tensors="pt", return_attention_mask=True, truncation=False
            )
            input_features = inputs.input_features
            attention_mask = inputs.get("attention_mask")
            if current_call_verbose:
                print(f"Shape of full input_features: {input_features.shape}")
                if attention_mask is not None: print(f"Shape of full attention_mask: {attention_mask.shape}")
                # Sanity check (uses HOP_LENGTH from module constants, ensure it's correct or get from processor)
                hop_length_val = self.processor.feature_extractor.hop_length if self.processor else 160
                expected_frames_approx = int((audio_waveform.shape[0] / SAMPLE_RATE) * (SAMPLE_RATE / hop_length_val))
                if abs(input_features.shape[-1] - expected_frames_approx) > 200: # Allow larger margin
                     warnings.warn(f"Processor returned {input_features.shape[-1]} frames for a "
                                   f"{audio_waveform.shape[0]/SAMPLE_RATE:.2f}s audio (expected ~{expected_frames_approx}).", UserWarning)
        except Exception as e:
            warnings.warn(f"Error during feature extraction: {e}", RuntimeWarning); raise

        model_dtype = next(self.model.parameters()).dtype
        try:
            if input_features.device != self.target_device: input_features = input_features.to(self.target_device)
            if input_features.dtype != model_dtype:
                if current_call_verbose: print(f"Casting input_features to model dtype {model_dtype}")
                input_features = input_features.to(model_dtype)
            if attention_mask is not None and attention_mask.device != self.target_device:
                attention_mask = attention_mask.to(self.target_device)
        except Exception as e:
             warnings.warn(f"Could not move/cast inputs: {e}", RuntimeWarning)

        generate_options = self._get_transcribe_kwargs(**kwargs)
        final_language_to_report = generate_options.get("language", self.model.config.forced_decoder_ids[0][1] 
                                                        if hasattr(self.model.config, "forced_decoder_ids") and 
                                                           self.model.config.forced_decoder_ids and 
                                                           len(self.model.config.forced_decoder_ids[0]) > 1 else "en")


        if current_call_verbose:
            print(f"Calling model.generate(). Options: {generate_options}")
            print(f"Model device: {self.model.device}, Input: {input_features.shape}, {input_features.dtype}, {input_features.device}")
            if attention_mask is not None: print(f"Attention mask: {attention_mask.shape}, {attention_mask.device}")

        full_text = ""; segments_data = []
        with torch.no_grad():
            try:
                output = self.model.generate(
                    input_features=input_features, attention_mask=attention_mask, **generate_options 
                )
                if self.target_device.type == "xpu": torch.xpu.synchronize()
                
                predicted_ids = output.sequences if hasattr(output, "sequences") else output
                full_text = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
                
                raw_segments = None
                if hasattr(output, "segments"): raw_segments = output.segments
                elif hasattr(output, "chunks"): raw_segments = output.chunks
                
                if raw_segments and isinstance(raw_segments, list):
                    for i, seg_data in enumerate(raw_segments):
                        # ... (segment parsing logic - keep as is from last version)
                        text = seg_data.get("text", "").strip()
                        ts = seg_data.get("timestamp", (0.0, 0.0)) 
                        start_time = ts[0] if isinstance(ts, (list, tuple)) and len(ts) > 0 and ts[0] is not None else 0.0
                        end_time = ts[1] if isinstance(ts, (list, tuple)) and len(ts) > 1 and ts[1] is not None else start_time 
                        segments_data.append({
                            "id": i, "seek": seg_data.get("seek", int(float(start_time) * SAMPLE_RATE / (HOP_LENGTH if 'HOP_LENGTH' in globals() and HOP_LENGTH > 0 else 160))), 
                            "start": round(float(start_time), 3), "end": round(float(end_time), 3),
                            "text": text, "tokens": seg_data.get("tokens", []),
                        })
                    if current_call_verbose: print(f"Extracted {len(segments_data)} segments.")
                elif full_text: 
                    warnings.warn("Detailed segments not found. Creating single segment.", UserWarning)
                    segments_data.append({
                        "id": 0, "seek": 0, "start": 0.0, "end": round(audio_waveform.shape[0]/SAMPLE_RATE, 3), 
                        "text": full_text, "tokens": predicted_ids.squeeze().tolist() if isinstance(predicted_ids, torch.Tensor) else []
                    })

                if hasattr(output, "language"): final_language_to_report = output.language
                elif final_language_to_report is None : final_language_to_report = "en"

                if current_call_verbose:
                    print(f"--- Transcription ---"); print(f"Language: {final_language_to_report}"); print(full_text)
                    # ... (segment printing)
                    print(f"--- End Transcription ---")
            except Exception as e:
                warnings.warn(f"Error during model.generate() or decoding: {e}", RuntimeWarning)
                import traceback; traceback.print_exc()
                final_lang_err = final_language_to_report if 'final_language_to_report' in locals() and final_language_to_report is not None else kwargs.get("language", "en")
                return {"text": "", "segments": [], "language": final_lang_err}
        
        final_lang_ret = final_language_to_report if 'final_language_to_report' in locals() and final_language_to_report is not None else kwargs.get("language", "en")
        return {"text": full_text, "segments": segments_data, "language": final_lang_ret}

# --- Your FasterWhisperTranscriber class definition (ensure it's complete and correct) ---
# ... (paste your full FasterWhisperTranscriber class here) ...
class FasterWhisperTranscriber(Transcriber):
    """
    Transcriber for FasterWhisper models.
    """
    def __init__(self, model_name: str, model_instance: FasterWhisperModel, verbose: bool = False) -> None: # Added verbose
        super().__init__(model_name, model_instance, verbose=verbose) # Pass verbose

    @classmethod
    def load_model(cls,
                   model_name: str = "medium",
                   download_root: Optional[str] = WHISPER_DEFAULT_PATH,
                   device_option: Optional[Union[str, torch.device]] = SCRAIBE_TORCH_DEVICE,
                   compute_type: str = "default", 
                   cpu_threads: int = SCRAIBE_NUM_THREADS,
                   num_workers: int = 1,
                   verbose: bool = False, # Added verbose
                   **kwargs: Any
                   ) -> 'FasterWhisperTranscriber':

        target_device_str = str(device_option if device_option else SCRAIBE_TORCH_DEVICE).lower()
        actual_fw_device = "cpu"
        effective_compute_type = compute_type

        if "cuda" in target_device_str:
            actual_fw_device = "cuda"
            if effective_compute_type == "default": effective_compute_type = "float16"
        elif "xpu" in target_device_str:
            actual_fw_device = "cpu" # FasterWhisper on XPU via OpenVINO often uses CPU device type
            if effective_compute_type == "default": effective_compute_type = "int8" 
            if verbose: warnings.warn(f"FasterWhisper on XPU: setting device to 'cpu' (for OpenVINO backend if used). Compute type: '{effective_compute_type}'.")
        else: # CPU
            actual_fw_device = "cpu"
            if effective_compute_type == "default": effective_compute_type = "int8"
        
        if actual_fw_device == 'cpu' and effective_compute_type == 'float16' and verbose:
            warnings.warn(f"Compute type 'float16' with device 'cpu' for FasterWhisper may not be optimal. Consider 'int8'.")

        if verbose: 
            print(f"Loading FasterWhisper model: {model_name}, device: {actual_fw_device}, compute_type: {effective_compute_type}")
        
        # Pass through any other relevant kwargs from the original FasterWhisper example if needed
        model_fw_kwargs = {k:v for k,v in kwargs.items() if k in signature(FasterWhisperModel).parameters}


        _model = FasterWhisperModel(model_size_or_path=model_name,
                                    download_root=download_root,
                                    device=actual_fw_device,
                                    compute_type=effective_compute_type,
                                    cpu_threads=cpu_threads,
                                    num_workers=num_workers,
                                    **model_fw_kwargs)
        return cls(model_name, _model, verbose=verbose)

    def transcribe(self, audio: Union[str, Tensor, np.ndarray], **kwargs: Any) -> Dict[str, Any]:
        processed_audio = audio
        if isinstance(audio, Tensor):
            processed_audio = audio.cpu().numpy().astype(np.float32)
        elif isinstance(audio, np.ndarray):
            processed_audio = audio.astype(np.float32)

        transcribe_kwargs = self._get_faster_whisper_transcribe_kwargs(**kwargs) # Use specific helper
        
        segments_iterable, info = self.model.transcribe(processed_audio, **transcribe_kwargs)
        
        full_text_parts = []; segments_data = []
        # tqdm for FasterWhisper if log_progress is True (default in its transcribe is False)
        # For now, direct iteration.
        segment_iterator = segments_iterable
        if kwargs.get("log_progress", False): # FasterWhisper transcribe has log_progress
            # Cannot easily get total number of segments beforehand for tqdm with a generator
            # We could convert to list first, or just iterate.
            if self.verbose: print("Transcribing segments with FasterWhisper...")


        for i, seg in enumerate(segment_iterator): # Use the iterator
            segment_text = seg.text.strip()
            full_text_parts.append(segment_text)
            segments_data.append({
                "id": i, "start": round(seg.start, 3), "end": round(seg.end, 3), 
                "text": segment_text, "tokens": list(seg.tokens), # Ensure tokens are list
                "temperature": float(seg.temperature), "avg_logprob": float(seg.avg_logprob),
                "compression_ratio": float(seg.compression_ratio), "no_speech_prob": float(seg.no_speech_prob)
            })
        
        full_text = " ".join(full_text_parts).strip()
        return {
            "text": full_text, "segments": segments_data,
            "language": info.language, "language_probability": float(info.language_probability)
            # Add other info fields if desired
        }

    @staticmethod # Make this specific to FasterWhisper
    def _get_faster_whisper_transcribe_kwargs(**kwargs: Any) -> Dict[str, Any]:
        known_faster_options = [
            "language", "task", "beam_size", "best_of", "patience", "length_penalty",
            "repetition_penalty", "no_repeat_ngram_size", "temperature", 
            "compression_ratio_threshold", "log_prob_threshold", "no_speech_threshold", 
            "condition_on_previous_text", "initial_prompt", "prefix", "suppress_blank", 
            "suppress_tokens", "without_timestamps", "max_initial_timestamp", 
            "word_timestamps", "vad_filter", "vad_parameters", "chunk_length",
            "hotwords" # Added from their docs
        ]
        final_kwargs = {k: v for k, v in kwargs.items() if k in known_faster_options}
        if (language := kwargs.get("language")):
            try: # Use the existing static method for conversion
                final_kwargs["language"] = FasterWhisperTranscriber.convert_to_language_code(language)
            except ValueError as e:
                warnings.warn(f"{e} Language will not be set for FasterWhisper.", UserWarning)
                if "language" in final_kwargs: del final_kwargs["language"]
        return final_kwargs

    # convert_to_language_code method remains as you had it.
    @staticmethod
    def convert_to_language_code(lang_input: str) -> Optional[str]:
        if not lang_input: return None
        lang_input_lower = lang_input.lower().strip()
        if lang_input_lower in FASTER_WHISPER_LANGUAGE_CODES: return lang_input_lower
        if lang_input_lower in OPENAI_WHISPER_TO_LANGUAGE_CODE:
            code = OPENAI_WHISPER_TO_LANGUAGE_CODE[lang_input_lower]
            if code in FASTER_WHISPER_LANGUAGE_CODES: return code
            else: warnings.warn(f"Lang '{lang_input}' mapped to '{code}', not in FasterWhisper codes. Using '{code}'.", UserWarning); return code
        available_codes_str = ", ".join(sorted(list(FASTER_WHISPER_LANGUAGE_CODES)))
        available_names_str = ", ".join(sorted([name for name, code in OPENAI_WHISPER_TO_LANGUAGE_CODE.items() if code in FASTER_WHISPER_LANGUAGE_CODES]))
        raise ValueError(f"Lang '{lang_input}' invalid. Known: {available_codes_str}. Names: {available_names_str}.")


# --- Your load_transcriber factory function (ensure verbose is passed) ---
def load_transcriber(
    model_name: str = "medium",
    whisper_type: str = 'openai-ipex-llm',
    download_root: Optional[str] = WHISPER_DEFAULT_PATH,
    device: Optional[Union[str, torch.device]] = None,
    **kwargs: Any # This will catch 'verbose', 'low_bit', 'num_beams', etc. from cli.py
) -> Transcriber:
    target_device = device if device is not None else SCRAIBE_TORCH_DEVICE
    whisper_type_lower = whisper_type.lower()

    # Pass all **kwargs down, which includes 'verbose' and other specific loader args
    if whisper_type_lower in ('openai-ipex-llm', 'whisper'):
        return OpenAIWhisperIPEXLLMTranscriber.load_model(
            model_name=model_name,
            download_root=download_root,
            device_option=target_device,
            **kwargs 
        )
    elif whisper_type_lower == 'faster-whisper':
        return FasterWhisperTranscriber.load_model(
            model_name=model_name,
            download_root=download_root,
            device_option=target_device,
            **kwargs
        )
    else:
        raise ValueError(f"Whisper type '{whisper_type}' not recognized.")


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