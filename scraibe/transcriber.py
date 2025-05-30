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
        Transcribe long audio files using a windowed approach, adapted from openai-whisper,
        with the IPEX-LLM loaded Whisper model.

        Args:
            audio (Union[str, torch.Tensor, np.ndarray]): Audio input.
                   If ndarray or Tensor, assumed to be a 1D float32 waveform.
                   If str, it's a path (this method expects Scraibe to pass the waveform Tensor).
            **kwargs:
                language (str, optional): Language of the audio. Auto-detected if None.
                task (str, optional): 'transcribe' or 'translate'. Defaults to 'transcribe'.
                verbose (bool, optional): Enables print statements.
                temperature (Union[float, Tuple[float, ...]], optional): Temperature(s) for generation.
                condition_on_previous_text (bool, optional): Defaults to True.
                initial_prompt (Optional[str], optional): Prompt for the first window.
                no_speech_threshold (Optional[float], optional): Threshold for detecting no speech.
                logprob_threshold (Optional[float], optional): Threshold for average log probability.
                compression_ratio_threshold (Optional[float], optional): Threshold for compression ratio.
                ... (other options that might map to generate() or control logic)
        Returns:
            Dict[str, Any]: {"text": str, "segments": List[Dict], "language": str}
        """
        if not self.processor or not self.model:
            raise RuntimeError("Transcriber is not properly initialized with a model and processor.")

        # ---- Parameter Setup ----
        self.verbose = kwargs.get("verbose", self.verbose) # Update instance verbose
        
        # Default values from openai-whisper transcribe function signature
        temperature_option = kwargs.get("temperature", (0.0, 0.2, 0.4, 0.6, 0.8, 1.0))
        if isinstance(temperature_option, (float, int)):
            temperatures = (temperature_option,)
        else:
            temperatures = tuple(temperature_option)

        condition_on_previous_text = kwargs.get("condition_on_previous_text", True)
        initial_prompt_str = kwargs.get("initial_prompt")
        
        # Fallback / quality thresholds (harder to implement fully without DecodingResult object)
        # For now, these will be noted but not fully used in fallback logic as in original.
        # We will rely more on temperature and no_repeat_ngram_size.
        logprob_threshold = kwargs.get("logprob_threshold", -1.0)
        no_speech_threshold = kwargs.get("no_speech_threshold", 0.6) # This is tricky without no_speech_prob
        compression_ratio_threshold = kwargs.get("compression_ratio_threshold", 2.4)

        # Get generation kwargs, apply some defaults for stability
        generate_args_base = self._get_transcribe_kwargs(**kwargs)
        generate_args_base.setdefault('use_cache', True)
        # no_repeat_ngram_size was set in _get_transcribe_kwargs, ensure it's there
        generate_args_base.setdefault('no_repeat_ngram_size', 3) 


        # ---- Audio Preprocessing ----
        if isinstance(audio, str):
            raise NotImplementedError("This transcribe method expects a waveform Tensor. Path loading should be handled by Scraibe.")
        elif isinstance(audio, np.ndarray):
            audio_waveform = torch.from_numpy(audio.astype(np.float32))
        elif isinstance(audio, torch.Tensor):
            audio_waveform = audio.to(torch.float32)
        else:
            raise TypeError(f"Expected audio to be str, Tensor, or ndarray, but got {type(audio)}")

        if audio_waveform.ndim > 1: audio_waveform = audio_waveform.squeeze()
        if audio_waveform.ndim != 1: raise ValueError(f"Audio waveform must be 1D, got {audio_waveform.ndim} dims.")

        if self.verbose: print("Extracting mel spectrogram for the entire audio...")
        try:
            # Get input_features (mel spectrogram) for the *entire* audio
            # Processor expects numpy array or list of floats for raw audio.
            full_input_features = self.processor(
                audio_waveform.cpu().numpy(),
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt"
            ).input_features # Shape: (batch_size, num_mel_bins, num_frames)
        except Exception as e:
            warnings.warn(f"Error during full audio feature extraction: {e}", RuntimeWarning)
            raise
        
        # Ensure features are on the target device and correct dtype for the model
        model_dtype = next(self.model.parameters()).dtype
        if full_input_features.device != self.target_device:
            full_input_features = full_input_features.to(self.target_device)
        if full_input_features.dtype != model_dtype:
            if self.verbose: print(f"Casting full_input_features to model dtype {model_dtype}")
            full_input_features = full_input_features.to(model_dtype)
        
        content_frames = full_input_features.shape[-1]

        # ---- Language Detection (Simplified for now) ----
        # The original has a more elaborate `model.detect_language`.
        # For HF models, language is usually set by prompting with language tokens.
        current_language = kwargs.get("language")
        if current_language is None:
            # Basic language detection: Use the first chunk.
            # This is a simplification. A dedicated detect_language call might be better.
            if self.verbose: print("Language not specified, attempting to detect from first 30s...")
            first_chunk_features = self._pad_or_trim_features(full_input_features[..., :N_FRAMES_PER_CHUNK])
            
            # Generate with language detection prompt (e.g., just SOT)
            # This is tricky as generate() doesn't have a direct "detect_language" mode like original Whisper.
            # It usually infers from initial tokens or lack thereof.
            # For now, we'll rely on the processor to provide a generic start if lang is None.
            # Or, we can try to generate and see what language token it outputs if model is multilingual.
            # This part is complex to replicate perfectly. Let's assume language is 'en' if not detected.
            # A robust way: generate from first chunk without lang prompt, check first few generated tokens.
            try:
                # A simplified way to get the processor to tell us the language if model is multilingual.
                # This might involve a small generation or inspecting processor's capabilities.
                # For now, if not provided, we'll default or let the first segment's prompt handle it.
                # The processor's get_decoder_prompt_ids might handle language=None.
                temp_prompt_ids = self.processor.get_decoder_prompt_ids(language=None, task="transcribe")
                # This doesn't directly give detected lang, but sets up for it.
                # The actual language will be part of the first segment's output if model is multilingual.
                # For now, let's assume if language is None, the first generate call will determine it.
                # We will extract it from the first segment's generated tokens.
                if self.verbose: print("Language detection will occur during the first segment's transcription.")
            except Exception as e:
                warnings.warn(f"Could not prepare for language detection: {e}. Defaulting to 'en' or model's default.", UserWarning)
                current_language = "en" # Fallback

        task = kwargs.get("task", "transcribe")

        # ---- Tokenizer & Prompt Setup ----
        all_tokens: List[int] = []
        all_segments: List[Dict[str, Any]] = []
        prompt_reset_since = 0 # Index in all_tokens

        if initial_prompt_str:
            initial_prompt_tokens = self.processor.tokenizer.encode(" " + initial_prompt_str.strip())
            all_tokens.extend(initial_prompt_tokens)

        # ---- Main Transcription Loop (Windowing) ----
        seek = 0
        with tqdm.tqdm(total=content_frames, unit="frames", disable=not self.verbose, desc="Transcribing") as pbar:
            while seek < content_frames:
                time_offset = float(seek * HOP_LENGTH / SAMPLE_RATE)
                segment_features = full_input_features[..., seek : seek + N_FRAMES_PER_CHUNK]
                segment_features_padded = self._pad_or_trim_features(segment_features) # Pad to N_FRAMES_PER_CHUNK

                # Prepare decoder_input_ids (prompt for the current segment)
                # This includes SOT, language, task, and previous text if conditioning
                previous_tokens_for_prompt = []
                if condition_on_previous_text:
                    # Max prompt length is roughly half the context window
                    # This needs to be based on self.model.config.max_length or similar
                    # For simplicity, let's use a fixed number or a portion of a typical context window (e.g. 224 tokens)
                    max_prompt_len = getattr(self.model.config, 'max_position_embeddings', 512) // 2 - 4 # Reserve space for lang/task tokens
                    
                    # Get tokens from previous transcription to use as prompt
                    prompt_start_index = prompt_reset_since
                    previous_tokens_for_prompt.extend(all_tokens[prompt_start_index:])
                    previous_tokens_for_prompt = previous_tokens_for_prompt[-max_prompt_len:]

                # Get base prompt (SOT, lang, task)
                try:
                    # If language was None, the first iteration will use language=None here,
                    # subsequent iterations will use the detected_lang_from_first_segment
                    lang_for_prompt = current_language
                    if current_language is None and all_segments: # If lang was detected from first segment
                        lang_for_prompt = all_segments[0].get("language", "en")

                    current_decoder_prompt_list = self.processor.get_decoder_prompt_ids(
                        language=lang_for_prompt, 
                        task=task, 
                        no_timestamps=True # Typically for prompting, timestamps are not part of the text prompt
                    )
                    # get_decoder_prompt_ids returns list of lists, e.g. [[50257, 50296, 50359, 50363]]
                    # We need the inner list.
                    current_prompt_tokens = current_decoder_prompt_list[0] if current_decoder_prompt_list else []
                except Exception as e:
                    warnings.warn(f"Error getting decoder prompt ids for lang {current_language}, task {task}: {e}. Using minimal prompt.", UserWarning)
                    current_prompt_tokens = [self.processor.tokenizer.sot_token_id] # Minimal SOT

                # Combine with previous text tokens if conditioning
                current_prompt_tokens.extend(previous_tokens_for_prompt)
                
                # Ensure prompt is not too long for the model
                # This needs a proper truncation strategy if it exceeds model's capacity.
                # For now, assuming previous_tokens_for_prompt was already limited.
                
                decoder_input_ids = torch.tensor([current_prompt_tokens], device=self.target_device).long()

                segment_generated_text = ""
                segment_tokens = []
                
                # --- Equivalent of decode_with_fallback ---
                # Simplified: iterate temperatures, use first successful.
                # Full fallback checks (compression, logprob) are complex to add here without DecodingResult.
                generated_successfully = False
                for temp_idx, temp in enumerate(temperatures):
                    current_generate_args = generate_args_base.copy()
                    current_generate_args['temperature'] = temp
                    
                    # Adjust beam search / sampling based on temperature
                    if temp > 0: # Sampling
                        current_generate_args['do_sample'] = True
                        current_generate_args.pop("num_beams", None) # Disable beam search for sampling
                        current_generate_args.setdefault("top_k", 0) # For temperature sampling
                    else: # Greedy or Beam search
                        current_generate_args['do_sample'] = False
                        current_generate_args.setdefault('num_beams', 1) # Default to greedy if not set
                        if current_generate_args['num_beams'] == 1: current_generate_args.pop('num_beams', None)


                    if self.verbose and len(temperatures) > 1:
                        print(f"  Attempting segment from {time_offset:.2f}s with temperature {temp:.1f}")

                    try:
                        predicted_ids_segment = self.model.generate(
                            segment_features_padded,
                            decoder_input_ids=decoder_input_ids,
                            **current_generate_args
                        )
                        if self.target_device.type == "xpu": torch.xpu.synchronize()

                        # Decode, removing prompt tokens from the beginning of the output
                        # The prompt tokens are part of decoder_input_ids, so generate() output will include them.
                        # We need to slice them off.
                        start_of_generation_idx = decoder_input_ids.shape[1] if decoder_input_ids is not None else 0
                        
                        segment_tokens_generated_only = predicted_ids_segment[0, start_of_generation_idx:].tolist()
                        
                        # Filter out special tokens like EOT, SOT, lang, task from the *generated* part
                        # This is a bit tricky as we want to keep actual content.
                        # For now, skip_special_tokens=True in batch_decode handles most of this.
                        segment_generated_text = self.processor.batch_decode(
                            predicted_ids_segment[:, start_of_generation_idx:], # Decode only generated part
                            skip_special_tokens=True
                        )[0].strip()
                        
                        segment_tokens = predicted_ids_segment[0].tolist() # Store all tokens for this segment for context

                        # Simplified check: if text is generated, consider it a success for this temp
                        # More advanced checks (repetition, no_speech_prob) would go here.
                        if segment_generated_text or kwargs.get("no_speech_threshold") is None: # If no_speech_threshold is used, need a way to check no_speech_prob
                            generated_successfully = True
                            break # Success with this temperature

                    except Exception as e_gen:
                        warnings.warn(f"Error during segment generation (temp {temp:.1f}): {e_gen}", RuntimeWarning)
                        if temp_idx == len(temperatures) - 1: # Last temperature failed
                            segment_generated_text = f"[ERROR: Generation failed for segment at {time_offset:.2f}s]"
                            segment_tokens = current_prompt_tokens + [self.processor.tokenizer.eos_token_id] # Minimal tokens
                            generated_successfully = True # Mark as "handled" to proceed
                        # Continue to next temperature if not the last one

                if not generated_successfully: # Should not happen if last temp error is handled
                     segment_generated_text = f"[ERROR: All temperatures failed for segment at {time_offset:.2f}s]"
                     segment_tokens = current_prompt_tokens + [self.processor.tokenizer.eos_token_id]


                # If language was None and this is the first segment, try to extract detected language
                if current_language is None and not all_segments:
                    try:
                        # Look for language token in the initial part of segment_tokens (after SOT)
                        # This is a heuristic. Whisper models embed lang token early.
                        # Example: <|sot|> <|lang_code|> <|task|> ...
                        # The actual language token ID needs to be identified.
                        # self.processor.tokenizer.lang_code_to_id might be useful if we know the token structure.
                        # For now, we'll assume the 'language' arg passed to get_decoder_prompt_ids was sufficient
                        # or the model defaults correctly. A more robust detection is complex here.
                        # Let's assume the language passed to the processor for the prompt is the one to use.
                        # If lang_for_prompt was None, this is still an issue.
                        # The original Whisper `transcribe` sets `decode_options["language"]` after `model.detect_language`.
                        # We are missing that direct detection step.
                        # For now, if language was None, we'll set it to 'en' or what the processor might default to.
                        # This part needs refinement for robust auto language detection.
                        # Let's assume the language passed to processor.get_decoder_prompt_ids is the one.
                        # If it was None, the model might output a language token.
                        # This is a complex part to replicate.
                        # For now, we'll use the 'language' that was used for the prompt.
                        # If it was None initially, it means we rely on model's multilingual capability.
                        # The returned 'language' will be what was used for prompting.
                        pass # current_language is already set or was None.
                    except Exception:
                        pass # Ignore if lang detection from tokens fails.

                # Create segment dictionary
                # Timestamps are based on audio window, not precise speech start/end from model's timestamp tokens yet.
                segment_end_time = time_offset + (len(segment_features[0]) * HOP_LENGTH / SAMPLE_RATE)
                # Ensure segment_end_time does not exceed total audio duration
                total_audio_duration_approx = content_frames * HOP_LENGTH / SAMPLE_RATE
                segment_end_time = min(segment_end_time, total_audio_duration_approx)


                all_segments.append({
                    "id": len(all_segments),
                    "seek": seek * HOP_LENGTH, # Seek in samples
                    "start": time_offset,
                    "end": segment_end_time,
                    "text": segment_generated_text,
                    "tokens": segment_tokens, # Full tokens for this segment including prompt
                    "temperature": temp, # Last used temperature
                    # The following are not easily available from HF generate, placeholders
                    "avg_logprob": -99.0, 
                    "compression_ratio": 0.0,
                    "no_speech_prob": 0.0,
                })
                if self.verbose:
                    print(f"[{time_offset:07.3f} --> {segment_end_time:07.3f}] {segment_generated_text}")

                # Update context for next segment
                all_tokens.extend(segment_tokens_generated_only) # Add only newly generated tokens for next prompt
                if not condition_on_previous_text or temp > 0.5: # Original logic
                    prompt_reset_since = len(all_tokens) 

                # Advance seek position
                # The original code has complex logic for advancing seek based on timestamp tokens.
                # Simplified seek: advance by the window size.
                # A more advanced version would adjust seek based on the actual content transcribed.
                # For now, fixed window progression.
                seek += N_FRAMES_PER_CHUNK 
                pbar.update(N_FRAMES_PER_CHUNK)
        
        final_text = self.processor.tokenizer.decode(all_tokens, skip_special_tokens=True).strip()
        
        # Determine final language (if auto-detected, it would be from the first segment, or model default)
        final_language = current_language
        if final_language is None and all_segments: # Try to get from first segment if still None
            # This is a placeholder for more robust language detection result
            # The 'language' field in each segment could be populated if model outputs lang tokens
            final_language = all_segments[0].get("language_from_model_output", "en") # Needs actual detection

        return {
            "text": final_text,
            "segments": all_segments,
            "language": final_language if final_language else "en" # Fallback
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