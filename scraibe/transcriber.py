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
import subprocess
import json
import os
import tempfile

from whisper import Whisper
from whisper import load_model as whisper_load_model
from whisper.tokenizer import TO_LANGUAGE_CODE
from faster_whisper import WhisperModel as FasterWhisperModel
from faster_whisper.tokenizer import _LANGUAGE_CODES as FASTER_WHISPER_LANGUAGE_CODES
from typing import TypeVar, Union, Optional
from torch import Tensor, device
from numpy import ndarray
from inspect import signature
from abc import abstractmethod
import warnings

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

        if not kwargs.get("verbose"):
            kwargs["verbose"] = None

        result = self.model.transcribe(audio, *args, **kwargs)
        return result["text"]

    @classmethod
    def load_model(cls,
                   model: str = "medium",
                   download_root: str = WHISPER_DEFAULT_PATH,
                   device: Optional[Union[str, device]] = SCRAIBE_TORCH_DEVICE,
                   in_memory: bool = False,
                   *args, **kwargs
                   ) -> 'WhisperTranscriber':
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
                                        Device to load model on. Defaults to None.
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
        _model = whisper_load_model(model, download_root=download_root,

                                    device=device, in_memory=in_memory)

        return cls(_model, model_name=model)

    @staticmethod
    def _get_whisper_kwargs(**kwargs) -> dict:
        """
        Get kwargs for whisper model. Ensure that kwargs are valid.

        Returns:
            dict: Keyword arguments for whisper model.
        """
        # _possible_kwargs = WhisperModel.transcribe.__code__.co_varnames
        _possible_kwargs = signature(Whisper.transcribe).parameters.keys()

        whisper_kwargs = {k: v for k,
                          v in kwargs.items() if k in _possible_kwargs}

        if (task := kwargs.get("task")):
            whisper_kwargs["task"] = task

        if (language := kwargs.get("language")):
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
        return cls(model=model, device=str(device), flash=flash, timestamp=timestamp,
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
