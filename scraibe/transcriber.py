"""
Transcriber Module
------------------

This module provides the Transcriber class, a comprehensive tool for working with Whisper models.
The Transcriber class offers functionalities such as loading different Whisper models, transcribing audio files,
and saving transcriptions to text files. It acts as an interface between various Whisper models and the user,
simplifying the process of audio transcription.

Main Features:
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
"""

from abc import ABC, abstractmethod
from inspect import signature
from typing import TypeVar, Union, Optional, Dict, Any, List, Tuple
import tqdm # For the progress bar

import torch
from torch import Tensor, device
from numpy import ndarray
import warnings
import numpy as np

# Type definition for the Transcriber model instance
ModelType = TypeVar('ModelType')

# OpenAI Whisper imports
from whisper.tokenizer import TO_LANGUAGE_CODE

from ipex_llm.transformers import AutoModelForSpeechSeq2Seq
from transformers import WhisperProcessor, pipeline
from transformers.pipelines.pt_utils import PipelineIterator

# --- Define Whisper constants ---
SAMPLE_RATE = 16000
CHUNK_LENGTH = 30 # seconds

# IPEX-LLM import
try:
    import ipex_llm
    import intel_extension_for_pytorch as ipex
    print("INFO: IPEX-LLM library found.")
except ImportError:
    ipex_llm = None
    ipex = None
    print("WARNING: IPEX-LLM library not found. IPEX-LLM specific optimizations will not be available for Whisper.")

# Local project imports (assuming these exist in scraibe.misc)
try:
    from .misc import WHISPER_DEFAULT_PATH, SCRAIBE_TORCH_DEVICE, SCRAIBE_NUM_THREADS
except ImportError:
    print("WARNING: Could not import from .misc. Using placeholder constants.")
    WHISPER_DEFAULT_PATH = None
    SCRAIBE_TORCH_DEVICE = "cpu"
    SCRAIBE_NUM_THREADS = 4

# Apply threading settings to PyTorch
if SCRAIBE_NUM_THREADS is not None:
    torch.set_num_threads(SCRAIBE_NUM_THREADS)


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
        Transcribe audio using the IPEX-LLM optimized Whisper model with hybrid CPU-GPU approach.
        This implementation properly chunks audio, processes features on CPU, and then batches
        them for efficient GPU processing.
        """
        if not self.processor or not self.model:
            raise RuntimeError("Transcriber is not properly initialized with a model and processor.")

        self.verbose = kwargs.get("verbose", self.verbose)
        language = kwargs.get("language", "en")
        task = kwargs.get("task", "transcribe")
        initial_prompt = kwargs.get("initial_prompt")
        batch_size = kwargs.get("batch_size", 8)  # Default batch size of 8 chunks
        
        # Load audio: accept file path, numpy array, or tensor
        if isinstance(audio, str):
            import soundfile as sf
            data, sr = sf.read(audio)
            if sr != SAMPLE_RATE:
                warnings.warn(f"Resampling from {sr} to {SAMPLE_RATE}")
                import librosa
                data = librosa.resample(data.astype(np.float32), orig_sr=sr, target_sr=SAMPLE_RATE)
            audio_array = data.astype(np.float32)
        elif isinstance(audio, np.ndarray):
            audio_array = audio.astype(np.float32)
        elif isinstance(audio, torch.Tensor):
            audio_array = audio.cpu().numpy().astype(np.float32)
        else:
            raise TypeError(f"Unsupported audio input type: {type(audio)}")
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio_array)
        if audio_tensor.ndim > 1:
            audio_tensor = audio_tensor.squeeze()
        if self.verbose:
            print(f"Processing audio with shape {audio_tensor.shape}, using chunk_length={CHUNK_LENGTH}s")
        
        # Get model dtype and device for proper input feature conversion
        model_dtype = next(self.model.parameters()).dtype
        model_device = next(self.model.parameters()).device
        
        # Prepare generate kwargs
        generate_kwargs = self._get_transcribe_kwargs(**kwargs)
        
        # Set forced_decoder_ids for language and task
        if language and task:
            decoder_prompt_ids = self.processor.get_decoder_prompt_ids(language=language, task=task)
            if decoder_prompt_ids:
                self.model.config.forced_decoder_ids = decoder_prompt_ids
        
        # Calculate number of samples per chunk
        samples_per_chunk = CHUNK_LENGTH * SAMPLE_RATE
        
        # Calculate number of chunks
        audio_length = audio_tensor.shape[0]
        num_chunks = max(1, int(np.ceil(audio_length / samples_per_chunk)))
        
        all_segments = []
        full_text_parts = []
        
        # Process chunks in batches
        for batch_start in range(0, num_chunks, batch_size):
            batch_end = min(batch_start + batch_size, num_chunks)
            current_batch_size = batch_end - batch_start
            
            if self.verbose:
                print(f"Processing batch of {current_batch_size} chunks ({batch_start+1}-{batch_end}/{num_chunks})")
            
            # Prepare batch of input features on CPU
            batch_features = []
            chunk_boundaries = []
            
            for i in range(batch_start, batch_end):
                # Calculate chunk boundaries
                start_sample = i * samples_per_chunk
                end_sample = min(start_sample + samples_per_chunk, audio_length)
                
                # Extract chunk
                chunk_start = max(0, start_sample)
                chunk_end = min(end_sample, audio_length)
                chunk = audio_tensor[chunk_start:chunk_end]
                
                # Store boundaries for later use
                chunk_boundaries.append((chunk_start, chunk_end))
                
                # Process features on CPU with float32
                with torch.no_grad():
                    features = self.processor(
                        chunk.cpu().numpy(),
                        sampling_rate=SAMPLE_RATE,
                        return_tensors="pt"
                    ).input_features.to(torch.float32)
                    # Cast features to model device and dtype to match model parameters
                    features = features.to(model_device, dtype=model_dtype)

                    batch_features.append(features)
            
            # Stack features if more than one chunk in the batch, otherwise use the single features tensor
            if len(batch_features) > 1:
                input_features = torch.cat(batch_features, dim=0)
            else:
                input_features = batch_features[0]
            generated_ids = None # Or some other default
            # Add initial prompt if specified and this is the first batch
            if initial_prompt and batch_start == 0:
                prompt_ids = self.processor.tokenizer.encode(initial_prompt, add_special_tokens=False)
                prompt_ids = torch.tensor([prompt_ids], device="cpu")  # Keep on CPU initially
                generate_kwargs["decoder_input_ids"] = prompt_ids
            
            # Generate and decode on target device
            input_features = input_features.to(model_device, dtype=model_dtype)
            if "decoder_input_ids" in generate_kwargs:
                generate_kwargs["decoder_input_ids"] = generate_kwargs["decoder_input_ids"].to(model_device)
            generated_ids = self.model.generate(input_features, **generate_kwargs)
            if model_device.type == "xpu":
                torch.xpu.synchronize()
            transcriptions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            
            # Process each result in the batch
            for i, (transcription, (chunk_start, chunk_end)) in enumerate(zip(transcriptions, chunk_boundaries)):
                chunk_idx = batch_start + i
                
                # Calculate timestamps
                chunk_start_time = chunk_start / SAMPLE_RATE
                chunk_end_time = chunk_end / SAMPLE_RATE
                
                # Add to segments
                all_segments.append({
                    "id": chunk_idx,
                    "start": chunk_start_time,
                    "end": chunk_end_time,
                    "text": transcription,
                    "temperature": generate_kwargs.get("temperature", 0.0),
                    "avg_logprob": -99.0,  # Placeholder
                    "compression_ratio": 0.0,  # Placeholder
                    "no_speech_prob": 0.0,  # Placeholder
                })
                
                full_text_parts.append(transcription)
                
                if self.verbose:
                    print(f"Chunk {chunk_idx+1}/{num_chunks}: {chunk_start_time:.2f}s - {chunk_end_time:.2f}s")
        
        # Combine all transcriptions
        full_text = " ".join(full_text_parts)
        
        return {
            "text": full_text,
            "segments": all_segments,
            "language": language
        }


    @staticmethod
    def _get_transcribe_kwargs(**kwargs: Any) -> Dict[str, Any]:
        """
        Prepare keyword arguments for Hugging Face model's `generate` method,
        with defaults aimed at improving long-form transcription stability.
        """
        generate_params = {}
        # Parameters directly supported by Hugging Face generate or easily mapped
        known_generate_options = [
            "max_length", "max_new_tokens", "min_length", "min_new_tokens",
            "early_stopping", "num_beams", "num_beam_groups", "do_sample", "use_cache",
            "temperature", "top_k", "top_p", "typical_p", "epsilon_cutoff", "eta_cutoff",
            "repetition_penalty", "length_penalty", "no_repeat_ngram_size",
            "encoder_no_repeat_ngram_size", "bad_words_ids", "force_words_ids",
            "forced_bos_token_id", "forced_eos_token_id", "remove_invalid_values",
            "suppress_tokens", "begin_suppress_tokens",
            "num_return_sequences", "output_attentions", "output_hidden_states",
            "output_scores", "return_dict_in_generate",
            "patience", # From original Whisper options, maps to HF generate patience
        ]
        
        for k, v in kwargs.items():
            if k in known_generate_options:
                generate_params[k] = v
            elif k == "sample_len" and "max_new_tokens" not in generate_params:
                generate_params["max_new_tokens"] = v
            # beam_size from kwargs (e.g. CLI) will be used for num_beams default below if not directly passed
        
        # --- Defaults to help with stability for long audio, if not overridden by user via **kwargs ---
        # Use a slight temperature to allow some variation and escape loops, unless user specifies one
        generate_params.setdefault('temperature', 0.2) 
        
        # Prevent short N-gram repetitions, unless user specifies one
        generate_params.setdefault('no_repeat_ngram_size', 3) 
        
        # Default to beam search if temperature is low and not explicitly sampling,
        # using 'beam_size' from kwargs if provided (e.g., from CLI), else default to 5.
        # The 'temperature' here refers to the one potentially set by the user in kwargs,
        # or our default of 0.2 if user didn't provide one.
        # The windowed transcribe loop will override this per iteration. This sets a base strategy.
        current_temp_for_logic = generate_params.get('temperature', 0.2) # Use setdefault value if no kwarg
        
        if not generate_params.get("do_sample", False): # If not explicitly asking for sampling
            if current_temp_for_logic <= 0.2: # For low/greedy temperatures
                # Default to beam search if num_beams is not already specified
                generate_params.setdefault('num_beams', kwargs.get('beam_size', 5)) 
            # If num_beams is now > 1 (either from kwargs or default), ensure do_sample is False
            if generate_params.get('num_beams', 1) > 1:
                generate_params['do_sample'] = False # Beam search and sampling are often mutually exclusive
                # Clean up conflicting sampling parameters if beam search is active
                generate_params.pop("top_k", None)
                generate_params.pop("top_p", None)
                generate_params.pop("typical_p", None)
        else: # If do_sample is explicitly True
            generate_params.pop("num_beams", None) # Ensure beam search is off for sampling

        # forced_decoder_ids is handled by passing decoder_input_ids in the main transcribe loop
        generate_params.pop('forced_decoder_ids', None)
        
        # Max new tokens: Whisper segments are ~30s. 
        # This should be a sensible default if not provided.
        # self.model.config.max_length is for the *entire* sequence (prompt + generation).
        # For generate, max_new_tokens or max_length (as total) is more relevant for a chunk.
        generate_params.setdefault('max_new_tokens', 1500) # Approx max tokens for 30s chunk
        
        # Note: We don't need to explicitly disable flash attention as it's not supported by the model

        return generate_params


# --- Factory Function to Load Transcriber ---
def load_transcriber(
    model_name: str = "medium",
    whisper_type: str = 'openai-ipex-llm', # Default to OpenAI with IPEX-LLM
    download_root: Optional[str] = WHISPER_DEFAULT_PATH,
    device: Optional[Union[str, torch.device]] = None, # Allow None to use SCRAIBE_TORCH_DEVICE
    **kwargs: Any  # Pass through all other specific arguments
) -> Transcriber:
    """
    Factory function to load and initialize a Whisper transcriber with IPEX-LLM optimization.

    Args:
        model_name (str): Whisper model name (e.g., "tiny", "base", "medium", "large-v2").
        whisper_type (str): Type of Whisper implementation to use.
                            Currently only supports "openai-ipex-llm" (or "whisper").
        download_root (Optional[str]): Path for model downloads/cache.
        device (Optional[Union[str, torch.device]]): Target device (e.g., "cpu", "cuda", "xpu").
                                                     Defaults to SCRAIBE_TORCH_DEVICE.
        **kwargs: Additional arguments passed to the transcriber's load_model method.
                  For "openai-ipex-llm": in_memory, use_ipex_llm, low_bit.

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
    else:
        raise ValueError(
            f"Whisper type '{whisper_type}' not recognized. "
            f"Only 'openai-ipex-llm' (or 'whisper') is supported."
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
        openai_low_bit = 'bf16' if openai_device == "xpu" else None # BF16

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
    except Exception as e:
        print(f"Error testing OpenAIWhisperIPEXLLMTranscriber: {e}")
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
