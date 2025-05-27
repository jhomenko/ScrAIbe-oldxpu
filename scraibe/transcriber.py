"""
Transcriber Module
------------------

This module provides the Transcriber class, a comprehensive tool for working with Whisper models.
The Transcriber class offers functionalities such as loading different Whisper models, transcribing audio files,
and saving transcriptions to text files. It acts as an interface between various Whisper models and the user,
simplifying the process of audio transcription.

Main Features:
    - Loading different sizes and versions of Whisper models.
    - Transcribing audio in various formats including str, Tensor, and nparray.
    - Saving the transcriptions to the specified paths.
    - Adaptable to various language specifications.
    - Options to control the verbosity of the transcription process.
    
Constants:
    WHISPER_DEFAULT_PATH: Default path for downloading and loading Whisper models.
"""
"""
Example Usage:
>>> transcriber = Transcriber.load_model(model="medium")
>>> transcript = transcriber.transcribe(audio="path/to/audio.wav")
>>> transcriber.save_transcript(transcript, "path/to/save.txt")
"""
from transformers import WhisperForConditionalGeneration, AutoProcessor
import intel_extension_for_pytorch as ipex
import torch # Assuming torch is imported elsewhere based on context, explicitly importing it here for clarity in diff
import subprocess
import json
import os
import tempfile

from whisper import Whisper
except ImportError:
from whisper.tokenizer import TO_LANGUAGE_CODE
from faster_whisper import WhisperModel as FasterWhisperModel
from faster_whisper.tokenizer import _LANGUAGE_CODES as FASTER_WHISPER_LANGUAGE_CODES
from typing import TypeVar, Union, Optional
from torch import Tensor, device
from numpy import ndarray
from inspect import signature
from abc import abstractmethod
import warnings

from whisper import load_model as whisper_load_model # Moved this import here for clarity
from .misc import WHISPER_DEFAULT_PATH, SCRAIBE_TORCH_DEVICE, SCRAIBE_NUM_THREADS
whisper = TypeVar('whisper')


class Transcriber:
    """
    Transcriber Class
    -----------------

    The Transcriber class serves as a wrapper around Whisper models for efficient audio
    transcription. By encapsulating the intricacies of loading models, processing audio,
    and saving transcripts, it offers an easy-to-use interface
    for users to transcribe audio files.

    Attributes:
        model (whisper): The Whisper model used for transcription.

    Methods:
        transcribe: Transcribes the given audio file.
        save_transcript: Saves the transcript to a file.
        load_model: Loads a specific Whisper model.
        _get_whisper_kwargs: Private method to get valid keyword arguments for the whisper model.

    Examples:
        >>> transcriber = Transcriber.load_model(model="medium")
        >>> transcript = transcriber.transcribe(audio="path/to/audio.wav")
        >>> transcriber.save_transcript(transcript, "path/to/save.txt")

    Note:
        The class supports various sizes and versions of Whisper models. Please refer to
        the load_model method for available options.
    """

    def __init__(self, model: whisper, model_name: str) -> None:
        """
        Initialize the Transcriber class with a Whisper model.

        Args:
            model (whisper): The Whisper model to use for transcription.
            model_name (str): The name of the model.
        """

        self.model = model

        self.model_name = model_name

    @abstractmethod
    def transcribe(self, audio: Union[str, Tensor, ndarray],
                   *args, **kwargs) -> str:
        """
        Transcribe an audio file.

        Args:
            audio (Union[str, Tensor, nparray]): The audio file to transcribe.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments, 
                        such as the language of the audio file.

        Returns:
            str: The transcript as a string.
        """
        pass

    @staticmethod
    def save_transcript(transcript: str, save_path: str) -> None:
        """
        Save a transcript to a file.

        Args:
            transcript (str): The transcript as a string.
            save_path (str): The path to save the transcript.

        Returns:
            None
        """

        with open(save_path, 'w') as f:
            f.write(transcript)

        print(f'Transcript saved to {save_path}')

    @classmethod
    @abstractmethod
    def load_model(cls,
                   model: str = "medium",
                   whisper_type: str = 'whisper',
                   download_root: str = WHISPER_DEFAULT_PATH,
                   device: Optional[Union[str, device]] = SCRAIBE_TORCH_DEVICE,
                   in_memory: bool = False,
                   *args, **kwargs
                   ) -> None:
        """
        Load whisper model.

        Args:
            model (str): Whisper model. Available models include:
                        - 'tiny.en'
                        - 'tiny'
                        - 'base.en'
                        - 'base'
                        - 'small.en'
                        - 'small'
                        - 'medium.en'
                        - 'medium'
                        - 'large-v1'
                        - 'large-v2'
                        - 'large-v3'
                        - 'large'
            whisper_type (str):
                                Type of whisper model to load. "whisper" or "faster-whisper".
            download_root (str, optional): Path to download the model.
                                            Defaults to WHISPER_DEFAULT_PATH.
            device (Optional[Union[str, torch.device]], optional): 
                                        Device to load model on. Defaults to None.
            in_memory (bool, optional): Whether to load model in memory. 
                                        Defaults to False.
            args: Additional arguments only to avoid errors.
            kwargs: Additional keyword arguments only to avoid errors.

        Returns:
            None: abscract method.
        """
        pass

    @staticmethod
    def _get_whisper_kwargs(**kwargs) -> dict:
        """
        Get kwargs for whisper model. Ensure that kwargs are valid.

        Returns:
            dict: Keyword arguments for whisper model.
        """
        pass

    def __repr__(self) -> str:
        return f"Transcriber(model_name={self.model_name}, model={self.model})"


class WhisperTranscriber(Transcriber):
    def __init__(self, model: WhisperForConditionalGeneration, processor: AutoProcessor, model_name: str) -> None:
        super().__init__(model, model_name) # Store the WhisperForConditionalGeneration model
        self._processor = processor # Store the processor


    def transcribe(self, audio: Union[str, Tensor, ndarray],
                   *args, **kwargs) -> str:
        """
        Transcribe an audio file.

        Args:
            audio (Union[str, Tensor, ndarray]): The audio input (waveform tensor).
            *args: Additional arguments.
            **kwargs: Additional keyword arguments, 
                        such as the language of the audio file.

        Returns:
        # Process audio using the loaded processor
        inputs = self._processor(waveform.squeeze(0), return_tensors="pt", sampling_rate=sample_rate).input_features

        # Move inputs to the model's device
        inputs = inputs.to(self.model.device)

        # Generate transcription using the model's generate method
        generated_ids = self.model.generate(inputs)


        # Decode the generated IDs to text
        transcription = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return transcription

    @classmethod
    def load_model(cls,
                   model: str = "medium",
                   download_root: str = WHISPER_DEFAULT_PATH,
                   device: Optional[Union[str, device]] = SCRAIBE_TORCH_DEVICE,
                   in_memory: bool = False, # This argument is not used in the new loading logic but kept for compatibility
                   *args, **kwargs
                   ) -> 'WhisperTranscriber':
        """
        Load whisper model using the transformers library with potential IPEX optimization.

        Args:
            model (str): Whisper model name (e.g., "medium", "large-v3").
            download_root (str, optional): Path to download the model (handled by transformers). Defaults to WHISPER_DEFAULT_PATH.
            device (Optional[Union[str, torch.device]], optional): Device to load model on. Defaults to SCRAIBE_TORCH_DEVICE.
            in_memory (bool, optional): This argument is not used in this implementation. Defaults to False.
            args: Additional arguments (ignored).
            kwargs: Additional keyword arguments (ignored).

        Returns:
            WhisperTranscriber: A WhisperTranscriber object initialized with the loaded model and processor.
        """
        # Determine the full model name from Hugging Face
        hf_model_name = f'openai/whisper-{model}' if '/' not in model else model

        # Determine the torch dtype based on the device
        torch_dtype = torch.float16 if device != 'cpu' else torch.float32

        _model = None
        _processor = None

        if str(device) == 'xpu' and hasattr(torch, 'xpu') and ipex is not None:
            try:
                # Attempt to load the model using ipex.OnDevice context manager
                with ipex.OnDevice(dtype=torch_dtype, device=device):
                    _model = WhisperForConditionalGeneration.from_pretrained(
                        hf_model_name,
                        torch_dtype=torch_dtype,
                        low_cpu_mem_usage=True,
                        trust_remote_code=True,
                    )
            except Exception as e:
                warnings.warn(f"Failed to load model with ipex.OnDevice: {e}. Falling back to regular loading.")
                # Fallback to regular loading on the specified device if IPEX loading fails
                try:
                    _model = WhisperForConditionalGeneration.from_pretrained(
                        hf_model_name,
                        torch_dtype=torch_dtype,
                        low_cpu_mem_usage=True,
                        trust_remote_code=True,
                    ).to(device)
                except Exception as fallback_e:
                    raise RuntimeError(f"Failed to load model even without ipex.OnDevice: {fallback_e}") from fallback_e
        else:
            # Load the model without ipex.OnDevice if device is not xpu or ipex is not available
            try:
                _model = WhisperForConditionalGeneration.from_pretrained(
                    hf_model_name,
                    torch_dtype=torch_dtype,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                ).to(device)
            except Exception as e:
                raise RuntimeError(f"Failed to load model: {e}") from e

        # Load the processor
        try:
            _processor = AutoProcessor.from_pretrained(hf_model_name)
        except Exception as e:
            # Clean up the loaded model if processor loading fails
            if _model is not None:
                del _model
                torch.cuda.empty_cache() # Clear CUDA cache if applicable
            raise RuntimeError(f"Failed to load processor: {e}") from e

        return cls(_model, _processor, model_name=model)

    @staticmethod
    def _get_whisper_kwargs(**kwargs) -> dict:
        """
        This method is kept for compatibility but is not used in the Hugging Face transformers based transcription.
        """
        # In the new implementation using transformers, keyword arguments for transcribe
        # are passed directly to the model's generate method or handled by the processor.
        # This method might need to be adapted or removed depending on how you want
        # to map your existing CLI/API arguments to the transformers generate parameters.

        # For now, returning a subset of potentially relevant kwargs.
        relevant_kwargs = {}
        # Example: mapping 'language' to 'forced_decoder_ids' if needed
        # if 'language' in kwargs:
        #     # You would need logic here to convert language name/code to token IDs
        #     pass
        # Add other relevant kwargs as needed, matching transformers generate arguments
        # e.g., temperature, num_beams, etc.

        # Note: Many Whisper arguments like 'verbose' or file paths are handled
        # differently or not applicable when using the transformers pipeline.
        warnings.warn("'_get_whisper_kwargs' is not fully utilized in the transformers-based WhisperTranscriber. "
                      "Transcription arguments are handled by the Hugging Face model's generate method and processor.")

        # Returning all kwargs for now, but the transcribe method will only use what's relevant for model.generate
        # You might want to filter this based on model.generate signature
        return kwargs

    def __repr__(self) -> str:
        return f"WhisperTranscriber(model_name={self.model_name})"


class FasterWhisperTranscriber(Transcriber):
    def __init__(self, model: FasterWhisperModel, model_name: str) -> None:
        super().__init__(model, model_name)

    def transcribe(self, audio: Union[str, Tensor, ndarray],
                   *args, **kwargs) -> str:
        """
        Transcribe an audio file.

        Args:
            audio (Union[str, Tensor, nparray]): The audio file to transcribe.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments,
                        such as the language of the audio file.

        Returns:
            str: The transcript as a string.
        """
        kwargs = self._get_whisper_kwargs(**kwargs)

        if isinstance(audio, Tensor):
            audio = audio.cpu().numpy()
        result, _ = self.model.transcribe(audio, *args, **kwargs)
        text = ""
        for seg in result:
            text += seg.text
        return text

    @classmethod
    def load_model(cls,
                   model: str = "medium",
                   download_root: str = WHISPER_DEFAULT_PATH,
                   device: Optional[Union[str, device]] = SCRAIBE_TORCH_DEVICE,
                   *args, **kwargs
                   ) -> 'FasterWhisperTranscriber':
        """
        Load faster-whisper model.

        Args:
            model (str): Whisper model. Available models include:
                        - 'tiny.en'
                        - 'tiny'
                        - 'base.en'
                        - 'base'
                        - 'small.en'
                        - 'small'
                        - 'medium.en'
                        - 'medium'
                        - 'large-v1'
                        - 'large-v2'
                        - 'large-v3'
                        - 'large'

            download_root (str, optional): Path to download the model.
                                            Defaults to WHISPER_DEFAULT_PATH.

            device (Optional[Union[str, torch.device]], optional):
                                        Device to load model on. Defaults to SCRAIBE_TORCH_DEVICE.
            in_memory (bool, optional): Whether to load model in memory.
                                        Defaults to False.
            args: Additional arguments only to avoid errors.
            kwargs: Additional keyword arguments only to avoid errors.

        Returns:
            Transcriber: A Transcriber object initialized with the specified model.
        """
        # Explicitly try to use 'xpu' if device is not specified
        if device is None:
            device = "xpu" if torch.xpu.is_available() else SCRAIBE_TORCH_DEVICE
        if not isinstance(device, str):

            device = str(device)

        compute_type = kwargs.get('compute_type', 'float16')
        if device == 'cpu' and compute_type == 'float16':
            warnings.warn(f'Compute type {compute_type} not compatible with '
                          f'device {device}! Changing compute type to int8.')
            compute_type = 'int8'
        # Determine if we should use XPU based on the device string and torch.xpu availability
        use_xpu = (device == 'xpu' or (isinstance(device, str) and 'xpu' in device) or (isinstance(device, torch.device) and device.type == 'xpu')) and hasattr(torch, 'xpu') and torch.xpu.is_available()

        if use_xpu:
            _model = FasterWhisperModel(model, download_root=download_root,
                                        device='auto', device_index=0, compute_type=compute_type, cpu_threads=SCRAIBE_NUM_THREADS)
        else:
            # Use the provided device string directly for non-XPU devices
            _model = FasterWhisperModel(model, download_root=download_root,
                                        device=device, compute_type=compute_type, cpu_threads=SCRAIBE_NUM_THREADS)

        return cls(_model, model_name=model)
    @staticmethod
    def _get_whisper_kwargs(**kwargs) -> dict:
        """
        Get kwargs for faster-whisper model. Ensure that kwargs are valid.

        Returns:
            dict: Keyword arguments for faster-whisper model.
        """
        # _possible_kwargs = WhisperModel.transcribe.__code__.co_varnames
        _possible_kwargs = signature(FasterWhisperModel.transcribe).parameters.keys()

        whisper_kwargs = {k: v for k,
                          v in kwargs.items() if k in _possible_kwargs}

        if (task := kwargs.get("task")):
            kwargs["verbose"] = None

            whisper_kwargs["task"] = task

        if (language := kwargs.get("language")):
            language = FasterWhisperTranscriber.convert_to_language_code(language)
            whisper_kwargs["language"] = language

        return whisper_kwargs

    def __repr__(self) -> str:
        return f"WhisperTranscriber(model_name={self.model_name}, model={self.model})"


class FasterWhisperTranscriber(Transcriber):
    def __init__(self, model: whisper, model_name: str) -> None:
        super().__init__(model, model_name)

    def transcribe(self, audio: Union[str, Tensor, ndarray],
                   *args, **kwargs) -> str:
        """
        Transcribe an audio file.

        Args:
            audio (Union[str, Tensor, nparray]): The audio file to transcribe.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments, 
                        such as the language of the audio file.

        Returns:
            str: The transcript as a string.
        """
        kwargs = self._get_whisper_kwargs(**kwargs)

        if isinstance(audio, Tensor):
            audio = audio.cpu().numpy()
        result, _ = self.model.transcribe(audio, *args, **kwargs)
        text = ""
        for seg in result:
            text += seg.text
        return text

    @classmethod
    def load_model(cls,
                   model: str = "medium",
                   download_root: str = WHISPER_DEFAULT_PATH,
                   device: Optional[Union[str, device]] = SCRAIBE_TORCH_DEVICE,
                   *args, **kwargs
                   ) -> 'FasterWhisperModel':
        """
        Load whisper model.

        Args:
            model (str): Whisper model. Available models include:
                        - 'tiny.en'
                        - 'tiny'
                        - 'base.en'
                        - 'base'
                        - 'small.en'
                        - 'small'
                        - 'medium.en'
                        - 'medium'
                        - 'large-v1'
                        - 'large-v2'
                        - 'large-v3'
                        - 'large'

            download_root (str, optional): Path to download the model.
                                            Defaults to WHISPER_DEFAULT_PATH.

            device (Optional[Union[str, torch.device]], optional): 
                                        Device to load model on. Defaults to SCRAIBE_TORCH_DEVICE.
            in_memory (bool, optional): Whether to load model in memory. 
                                        Defaults to False.
            args: Additional arguments only to avoid errors.
            kwargs: Additional keyword arguments only to avoid errors.

        Returns:
            Transcriber: A Transcriber object initialized with the specified model.
        """
        # Explicitly try to use 'xpu' if device is not specified
        if device is None:
            device = "xpu" if torch.xpu.is_available() else SCRAIBE_TORCH_DEVICE
        if not isinstance(device, str):

            device = str(device)
            
        compute_type = kwargs.get('compute_type', 'float16')
        if device == 'cpu' and compute_type == 'float16':
            warnings.warn(f'Compute type {compute_type} not compatible with '
                          f'device {device}! Changing compute type to int8.')
            compute_type = 'int8'
        if device == 'xpu':
            _model = FasterWhisperModel(model, download_root=download_root,
                                        device='auto', device_index=0, compute_type=compute_type, cpu_threads=SCRAIBE_NUM_THREADS)
        else:
            _model = FasterWhisperModel(model, download_root=download_root,
                                        device=device, compute_type=compute_type, cpu_threads=SCRAIBE_NUM_THREADS)

        return cls(_model, model_name=model)
    @staticmethod
    def _get_whisper_kwargs(**kwargs) -> dict:
        """
        Get kwargs for whisper model. Ensure that kwargs are valid.

        Returns:
            dict: Keyword arguments for whisper model.
        """
        # _possible_kwargs = WhisperModel.transcribe.__code__.co_varnames
        _possible_kwargs = signature(FasterWhisperModel.transcribe).parameters.keys()

        whisper_kwargs = {k: v for k,
                          v in kwargs.items() if k in _possible_kwargs}

        if (task := kwargs.get("task")):
            whisper_kwargs["task"] = task

        if (language := kwargs.get("language")):
            language = FasterWhisperTranscriber.convert_to_language_code(language)
            whisper_kwargs["language"] = language

        return whisper_kwargs

    @staticmethod
    def convert_to_language_code(lang : str) -> str:
        """
        Load whisper model.

        Args:
            lang (str): language as code or language name

        Returns:
            language (str) code of language 
        """
        
        # If the input is already in FASTER_WHISPER_LANGUAGE_CODES, return it directly
        if lang in FASTER_WHISPER_LANGUAGE_CODES:
            return lang

        # Normalize the input to lowercase
        lang = lang.lower()

        # Check if the language name is in the TO_LANGUAGE_CODE mapping
        if lang in TO_LANGUAGE_CODE:
            return TO_LANGUAGE_CODE[lang]

        # If the language is not recognized, raise a ValueError with the available options
        available_codes = ', '.join(FASTER_WHISPER_LANGUAGE_CODES)
        raise ValueError(f"Language '{lang}' is not a valid language code or name. "
                        f"Available language codes are: {available_codes}.")

    def __repr__(self) -> str:
        return f"FasterWhisperTranscriber(model_name={self.model_name}, model={self.model})"


class InsanelyFastWhisperTranscriber(Transcriber):
    """
    Transcriber Class for Insanely Fast Whisper CLI.

    This class wraps the functionality of the insanely-fast-whisper command-line
    tool, executing it via subprocess to perform audio transcription.
    """
    def __init__(self, model: str, device: str, flash: bool = False,
                 timestamp: str = 'chunk', hf_token: Optional[str] = None,
                 min_speakers: Optional[int] = None, max_speakers: Optional[int] = None) -> None:
        """
        Initialize the InsanelyFastWhisperTranscriber with specified parameters.

        Args:
            model (str): Name of the Whisper model to use.
            device (str): Device to use for inference (e.g., "0" for CUDA, "mps" for Mac).
            flash (bool, optional): Whether to use Flash Attention 2. Defaults to False.
            timestamp (str, optional): Timestamp level ('chunk' or 'word'). Defaults to 'chunk'.
            hf_token (Optional[str], optional): Hugging Face token for diarization. Defaults to None.
            min_speakers (Optional[int], optional): Minimum number of speakers for diarization. Defaults to None.
            max_speakers (Optional[int], optional): Maximum number of speakers for diarization. Defaults to None.
        """
        # The 'model' attribute is the model name string for this class
        self.model = model
        self.device = device
        self.flash = flash
        self.timestamp = timestamp
        self.hf_token = hf_token
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers

    def transcribe(self, audio: Union[str, Tensor, ndarray],
                   *args, **kwargs) -> str:
        """
        Transcribe an audio file using Insanely Fast Whisper.

        Args:
            audio (Union[str, Tensor, ndarray]): The audio file path to transcribe.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: The transcription results as a dictionary (parsed from JSON output).
        """
        if not isinstance(audio, str):
             raise TypeError("Insanely Fast Whisper transcriber requires an audio file path (string).")

        # Create a temporary file for the output JSON
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as tmp_file:
            output_path = tmp_file.name

        command = [
            'insanely-fast-whisper',
            '--file-name', audio,
            '--model-name', self.model,
            '--device-id', self.device,
            '--transcript-path', output_path,
            '--timestamp', self.timestamp
        ]

        if self.flash:
            command.append('--flash')

        # Note: We are not adding diarization arguments here as we decided to use ScrAIbe's internal diarization

        try:
            # Execute the command
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            # Read and parse the output JSON
            with open(output_path, 'r') as f:
                transcription_data = json.load(f)
        except subprocess.CalledProcessError as e:
            print(f"Error executing insanely-fast-whisper: {e}")
            print(f"Stderr: {e.stderr}")
            raise
        finally:
            # Clean up the temporary file
            if os.path.exists(output_path):
                os.remove(output_path)

        # The format of the returned data depends on insanely-fast-whisper's output.
        # We assume it's a dictionary containing the transcription results.
        # You might need to adjust this part based on the actual output structure
        # to make it compatible with ScrAIbe's expected format (e.g., list of segments).
        return transcription_data

    @classmethod
    def load_model(cls,
                   model: str = "medium",
                   download_root: str = WHISPER_DEFAULT_PATH,
                   device: Optional[Union[str, device]] = SCRAIBE_TORCH_DEVICE,
                   flash: bool = False,
                   timestamp: str = 'chunk',
                   hf_token: Optional[str] = None,
                   min_speakers: Optional[int] = None,
                   max_speakers: Optional[int] = None,
                   *args, **kwargs # Catch any extra args
                   ) -> 'InsanelyFastWhisperTranscriber':
        """
        Load Insanely Fast Whisper model.

        Args:
            model (str): Whisper model. Available models include:
            device (Optional[Union[str, torch.device]], optional):
                                        Device to load model on. Defaults to SCRAIBE_TORCH_DEVICE.
            flash (bool, optional): Whether to use Flash Attention 2. Defaults to False.
            timestamp (str, optional): Timestamp level ('chunk' or 'word'). Defaults to 'chunk'.
            hf_token (Optional[str], optional): Hugging Face token for diarization. Defaults to None.
            min_speakers (Optional[int], optional): Minimum number of speakers for diarization. Defaults to None.
            max_speakers (Optional[int], optional): Maximum number of speakers for diarization. Defaults to None.

        Returns:
            Transcriber: A Transcriber object initialized with the specified model.
        """
        # Insanely Fast Whisper uses the device string directly
        if device is None:
             # Assuming torch is imported elsewhere and xpu availability is checked
             device = "xpu" if hasattr(torch, 'xpu') and torch.xpu.is_available() else SCRAIBE_TORCH_DEVICE

        # We don't actually load the model in this class's load_model.
        # The model is handled by the subprocess call in transcribe.
        # We just instantiate the transcriber with the config.
        # Determine the full model name for Insanely Fast Whisper
        full_model_name = model if '/' in model else f'openai/whisper-{model}'

        return cls(model=full_model_name, device=str(device), flash=flash, timestamp=timestamp,
                   hf_token=hf_token, min_speakers=min_speakers, max_speakers=max_speakers)

    @staticmethod
    def _get_whisper_kwargs(**kwargs) -> dict:
        # Insanely Fast Whisper's transcribe method signature is different
        return {k: v for k, v in kwargs.items()}


def load_transcriber(model: str = "medium",
                     whisper_type: str = 'whisper',
                     download_root: str = WHISPER_DEFAULT_PATH,
                     device: Optional[Union[str, device]] = SCRAIBE_TORCH_DEVICE,
                     in_memory: bool = False,
                   flash: bool = False,
                   timestamp: str = 'chunk',
                   hf_token: Optional[str] = None,
                   min_speakers: Optional[int] = None,
                   max_speakers: Optional[int] = None,
                   compute_type: str = 'float16', # Added compute_type for faster-whisper
                   *args,
                   **kwargs # Catch any extra args
                     ) -> Union[WhisperTranscriber, FasterWhisperTranscriber]:
    """
    Load whisper model.

    Args:
        model (str): Whisper model. Available models include:
                    - 'tiny.en'
                    - 'tiny'
                    - 'base.en'
                    - 'base'
                    - 'small.en'
                    - 'small'
                    - 'medium.en'
                    - 'medium'
                    - 'large-v1'
                    - 'large-v2'
                    - 'large-v3'
                    - 'large'
        whisper_type (str):
                            Type of whisper model to load. "whisper" or "faster-whisper".
        download_root (str, optional): Path to download the model.
                                        Defaults to WHISPER_DEFAULT_PATH.
        device (Optional[Union[str, torch.device]], optional):
                                    Device to load model on. Defaults to SCRAIBE_TORCH_DEVICE.
        in_memory (bool, optional): Whether to load model in memory.
                                    Defaults to False.
        args: Additional arguments only to avoid errors.
        kwargs: Additional keyword arguments for specific transcribers.
        flash (bool, optional): Enable Flash Attention 2 for InsanelyFastWhisper. Defaults to False.
        timestamp (str, optional): Timestamp level for InsanelyFastWhisper (\'chunk\' or \'word\'). Defaults to \'chunk\'.
        hf_token (Optional[str], optional): Hugging Face token for diarization (for InsanelyFastWhisper). Defaults to None.
        min_speakers (Optional[int], optional): Minimum number of speakers for diarization (for InsanelyFastWhisper). Defaults to None.
        max_speakers (Optional[int], optional): Maximum number of speakers for diarization (for InsanelyFastWhisper). Defaults to None.
        compute_type (str, optional): Compute type for faster-whisper. Defaults to 'float16'.

    Returns:
        Union[WhisperTranscriber, FasterWhisperTranscriber]:
        One of the Whisper variants as Transcrbier object initialized with the specified model.
    """

    if whisper_type.lower() == 'whisper':
        _model = WhisperTranscriber.load_model(
            model, download_root, device, in_memory, *args, **kwargs)
        return _model
    elif whisper_type.lower() == 'faster-whisper':
        _model = FasterWhisperTranscriber.load_model(
 model, download_root, device, *args, compute_type=compute_type, **kwargs)
        return _model

    elif whisper_type.lower() == 'insanely-fast-whisper':
         _model = InsanelyFastWhisperTranscriber.load_model(
 model=model, device=device, flash=flash, timestamp=timestamp,
 hf_token=hf_token, min_speakers=min_speakers, max_speakers=max_speakers, **kwargs)
         return _model
    else:
        raise ValueError(f'Model type not recognized, exptected "whisper" '
                         f'or "faster-whisper", got {whisper_type}.')
