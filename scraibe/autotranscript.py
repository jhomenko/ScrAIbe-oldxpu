"""
Scraibe Class
--------------------

This class serves as the core of the transcription system, responsible for handling
transcription and diarization of audio files. It leverages pretrained models for
speech-to-text (such as Whisper) and speaker diarization (such as pyannote.audio),
providing an accessible interface for audio processing tasks such as transcription,
speaker separation, and timestamping.

By encapsulating the complexities of underlying models, it allows for straightforward
integration into various applications, ranging from transcription services to voice assistants.

Available Classes:
- Scraibe: Main class for performing transcription and diarization.
                  Includes methods for loading models, processing audio files,
                  and formatting the transcription output.

Usage:
    from scraibe import Scraibe

    # Example: Initialize with specific models and device
    # model = Scraibe(whisper_model="medium", whisper_type="openai-ipex-llm", 
    #                 dia_model="pyannote/speaker-diarization", target_device="xpu",
    #                 low_bit='bf16') # low_bit is a component_kwarg for the transcriber
    # transcript = model.autotranscribe("path/to/audiofile.wav")
"""

# Standard Library Imports
import os
from subprocess import run
from typing import TypeVar, Union, Optional, Any, Dict
import warnings # For warnings

# Third-Party Imports
import torch
import numpy as np # Added for ndarray type hints and processing
from tqdm import trange

# Application-Specific Imports
from .audio import AudioProcessor
from .diarisation import Diariser
from .transcriber import Transcriber, load_transcriber
from .transcript_exporter import Transcript
from .misc import SCRAIBE_TORCH_DEVICE, WHISPER_DEFAULT_PATH

DiarisationType = TypeVar('DiarisationType') # For Diariser model instances

class Scraibe:
    """
    Scraibe is a class responsible for managing the transcription and diarization of audio files.
    It integrates transcription models (e.g., Whisper) and diarization models (e.g., Pyannote)
    to provide a comprehensive audio processing solution.
    """

    def __init__(self,
                 whisper_model: Union[str, Transcriber, None] = "medium",
                 whisper_type: str = "openai-ipex-llm",
                 dia_model: Union[str, Diariser, None] = None,
                 target_device: Optional[Union[str, torch.device]] = None,
                 download_root: Optional[str] = None, # For Whisper model downloads
                 use_auth_token: Optional[str] = None, # For Hugging Face authenticated models
                 verbose: bool = False,
                 save_setup: bool = False,
                 **component_kwargs: Any) -> None: # For other specific loader args e.g. low_bit, compute_type
        """Initializes the Scraibe class.

        Args:
            whisper_model (Union[str, Transcriber, None], optional):
                Whisper model name (e.g., "medium", "large-v2"), an existing Transcriber instance,
                or None to use the default "medium". Defaults to "medium".
            whisper_type (str, optional): Type of whisper model to load if whisper_model is a string.
                E.g., "openai-ipex-llm", "faster-whisper". Defaults to "openai-ipex-llm".
            dia_model (Union[str, Diariser, None], optional):
                Diarization model name/path (e.g., "pyannote/speaker-diarization@2.1"), 
                an existing Diariser instance, or None to load a default diarizer. Defaults to None.
            target_device (Optional[Union[str, torch.device]], optional): 
                Primary device for models (e.g. "cpu", "xpu", "cuda").
                Overrides SCRAIBE_TORCH_DEVICE if set. Defaults to SCRAIBE_TORCH_DEVICE.
            download_root (Optional[str], optional): Path to download/cache Whisper models.
                Defaults to WHISPER_DEFAULT_PATH or the library's default.
            use_auth_token (Optional[str], optional): HuggingFace token for private/gated models.
                Used by both transcriber and diarizer if applicable. Defaults to None.
            verbose (bool, optional): If True, print additional information. Defaults to False.
            save_setup (bool, optional): If True, initialization parameters are stored. Defaults to False.
            **component_kwargs: Additional keyword arguments for underlying model loaders.
                e.g., `low_bit` for IPEX-LLM, `compute_type` for FasterWhisper.
        """
        self.verbose = verbose
        self.target_device = torch.device(target_device if target_device is not None else SCRAIBE_TORCH_DEVICE)

        # --- Initialize Transcriber ---
        transcriber_load_kwargs = component_kwargs.copy()
        #transcriber_load_kwargs['device_option'] = self.target_device # Ensure correct param name for load_transcriber
        if download_root: # download_root is an explicit param of Scraibe.__init__
            transcriber_load_kwargs['download_root'] = download_root
        if use_auth_token: # use_auth_token is an explicit param of Scraibe.__init__
             transcriber_load_kwargs['use_auth_token'] = use_auth_token

        if isinstance(whisper_model, Transcriber):
            self.transcriber = whisper_model
            if self.verbose: print(f"Using provided Transcriber instance: {self.transcriber}")
        else: 
            effective_whisper_model_name = whisper_model if whisper_model is not None else "medium"
            if self.verbose: 
                # The device printed here is self.target_device, which will be passed to load_transcriber's `device` param
                print(f"Loading Transcriber: model='{effective_whisper_model_name}', type='{whisper_type}', device='{self.target_device}'")
            self.transcriber = load_transcriber(
                model_name=effective_whisper_model_name,
                whisper_type=whisper_type,
                device=self.target_device, # << PASS self.target_device to the 'device' PARAMETER of load_transcriber
                **transcriber_load_kwargs  # These kwargs should no longer contain 'device_option'
            )

        # --- Initialize Diariser ---
        # Filter component_kwargs to be more specific for Diariser.load_model
        diariser_load_kwargs = {}
        if use_auth_token: # Pyannote uses 'token' or 'use_auth_token'
            diariser_load_kwargs['use_auth_token'] = use_auth_token # Or 'token': use_auth_token if Diariser.load_model expects that
        # Pass any other relevant kwargs from component_kwargs if Diariser.load_model expects them.
        # For now, keeping it minimal to avoid unexpected argument errors.
        # If Diariser.load_model takes generic **kwargs, you can pass more from component_kwargs after filtering.
        # Example: known_diariser_params = ['some_diariser_param']
        # for k, v in component_kwargs.items():
        #     if k in known_diariser_params: diariser_load_kwargs[k] = v
            
        if isinstance(dia_model, Diariser):
            self.diariser = dia_model
            if self.verbose: print(f"Using provided Diariser instance: {self.diariser}")
        else: # Handles string name for model or None for default
            if self.verbose: 
                print(f"Loading Diariser: model='{dia_model if dia_model else 'default'}', device='{self.target_device}'")
            self.diariser = Diariser.load_model(
                model_name_or_path=dia_model, # Pass the name/path or None
                device=self.target_device,    # Explicit device argument
                **diariser_load_kwargs        # Filtered additional kwargs
            )

        if self.verbose:
            print(f"Scraibe initialized. Transcriber: {self.transcriber}, Diariser: {self.diariser} on device '{self.target_device}'")

        self.params = {}
        if save_setup:
            # Store string representation for instances to avoid circular refs or large objects
            # Ensure all primary parameters are included.
            self.params = dict(
                whisper_model=str(whisper_model.model_name) if isinstance(whisper_model, Transcriber) else whisper_model,
                whisper_type=whisper_type,
                dia_model=str(dia_model.model_name) if isinstance(dia_model, Diariser) and hasattr(dia_model, 'model_name') else dia_model,
                target_device=str(self.target_device),
                download_root=download_root,
                use_auth_token="****" if use_auth_token else None, # Mask token
                verbose=verbose,
                **component_kwargs # Store the extra component kwargs as well
            )

    def autotranscribe(self, audio_file: Union[str, torch.Tensor, np.ndarray],
                       remove_original: bool = False,
                       **kwargs: Any) -> Transcript:
        """
        Transcribes an audio file using the whisper model and pyannote diarization model.
        Segments the audio by speaker first, then transcribes each segment.

        Args:
            audio_file (Union[str, torch.Tensor, np.ndarray]): Path to the audio file,
                or audio data as a PyTorch Tensor or NumPy ndarray.
            remove_original (bool, optional): If True and audio_file is a path,
                the original audio file will be removed after processing. Defaults to False.
            **kwargs: Additional keyword arguments.
                - verbose (bool): Override instance verbosity for this call.
                - task (str): For transcriber, e.g., "transcribe", "translate".
                - language (str): Language code for transcription.
                - num_speakers (int): Hint for diarization.
                - shred (bool): If True and remove_original is True, shred the file.
                - Other kwargs for `diariser.diarization` or `transcriber.transcribe`.

        Returns:
            Transcript: An object containing the structured transcription data.
        """
        current_verbose = kwargs.get("verbose", self.verbose)

        audio_processor: AudioProcessor = self.get_audio_file(audio_file)

        dia_waveform = audio_processor.waveform.reshape(1, -1)
        if dia_waveform.dtype != torch.float32:
            dia_waveform = dia_waveform.to(dtype=torch.float32)

        dia_audio_input = {
            "waveform": dia_waveform.to(self.diariser.device), # Move to diariser's specific device
            "sample_rate": audio_processor.sr
        }
        
        if current_verbose:
            print("Starting diarisation...")

        # Filter kwargs for self.diariser.diarization method
        # This introspection can be fragile. Consider defining expected params or letting Diariser handle **kwargs.
        diarization_params = {k: v for k, v in kwargs.items() if k in self.diariser.diarization.__code__.co_varnames}
        if 'num_speakers' in kwargs and 'num_speakers' not in diarization_params: # Explicitly pass if CLI gave it
            diarization_params['num_speakers'] = kwargs['num_speakers']

        diarisation_result = self.diariser.diarization(dia_audio_input, **diarization_params)

        if not diarisation_result.get("segments"):
            if current_verbose:
                print("No speaker segments found by diariser. Transcribing entire audio as a single speaker.")
            
            transcription_result = self.transcriber.transcribe(
                audio_processor.waveform, 
                task=kwargs.get("task", "transcribe"), 
                language=kwargs.get("language"),
                verbose=kwargs.get("verbose") # Pass verbose to transcriber if it uses it
                # Other relevant transcribe kwargs will be filtered by self.transcriber.transcribe
            )
            duration_s = len(audio_processor.waveform) / audio_processor.sr
            final_transcript_data = {
                0: { # Using integer key for segment ID
                    "speakers": 'SPEAKER_01',
                    "segments": [0.0, duration_s],
                    "text": transcription_result.get("text", "")
                }
            }
            return Transcript(final_transcript_data)

        if current_verbose:
            print(f"Diarisation finished with {len(diarisation_result['segments'])} segments. Starting transcription per segment.")

        final_transcript_data = {}
        # Assuming diarisation_result["segments"] gives list of [start, end]
        # and diarisation_result["speakers"] gives corresponding list of speaker labels
        for i, (seg_start_end, speaker_label) in enumerate(
            zip(diarisation_result.get("segments", []), diarisation_result.get("speakers", []))):
            
            if not isinstance(seg_start_end, (list, tuple)) or len(seg_start_end) != 2:
                if current_verbose: print(f"Skipping invalid segment data: {seg_start_end}")
                continue
            seg_start, seg_end = seg_start_end

            # Use trange for progress bar if many segments
            if i == 0 and len(diarisation_result['segments']) > 10 : _iterator = trange(len(diarisation_result['segments']))
            else: _iterator = range(len(diarisation_result['segments'])) # Avoid re-creating trange

            # audio_processor.cut should return just the waveform for the segment
            audio_segment_waveform = audio_processor.cut(seg_start, seg_end)

            segment_transcription_result = self.transcriber.transcribe(
                audio_segment_waveform, 
                task=kwargs.get("task", "transcribe"),
                language=kwargs.get("language"),
                verbose=kwargs.get("verbose", False) # Control verbosity of segment transcription
                # Other relevant transcribe kwargs will be filtered by self.transcriber.transcribe
            )
            final_transcript_data[i] = { # Use index as segment ID key
                "speakers": speaker_label,
                "segments": [seg_start, seg_end],
                "text": segment_transcription_result.get("text", "")
            }
            if i == 0 and 'trange' in locals() and isinstance(_iterator, trange): _iterator.set_description_str(f"Transcribing segment {i+1}/{len(diarisation_result['segments'])}")
            elif isinstance(_iterator, trange): _iterator.update(1)


        if remove_original and isinstance(audio_file, str):
            shred_flag = kwargs.get("shred", False)
            self.remove_audio_file(audio_file, shred=shred_flag)

        return Transcript(final_transcript_data)

    def diarization(self, audio_file: Union[str, torch.Tensor, np.ndarray],
                    **kwargs: Any) -> Dict[str, Any]:
        """Performs speaker diarization on the audio."""
        current_verbose = kwargs.get("verbose", self.verbose)
        audio_processor: AudioProcessor = self.get_audio_file(audio_file)

        dia_waveform = audio_processor.waveform.reshape(1, -1)
        if dia_waveform.dtype != torch.float32:
            dia_waveform = dia_waveform.to(dtype=torch.float32)

        dia_audio_input = {
            "waveform": dia_waveform.to(self.diariser.device),
            "sample_rate": audio_processor.sr
        }
        if current_verbose: print("Starting diarisation (direct call)...")
        
        diarization_params = {k: v for k, v in kwargs.items() if k in self.diariser.diarization.__code__.co_varnames}
        if 'num_speakers' in kwargs and 'num_speakers' not in diarization_params:
             diarization_params['num_speakers'] = kwargs['num_speakers']

        diarisation_result = self.diariser.diarization(dia_audio_input, **diarization_params)
        return diarisation_result

    def transcribe(self, audio_file: Union[str, torch.Tensor, np.ndarray],
                   **kwargs: Any) -> Dict[str, Any]:
        """Transcribes the entire audio file without speaker diarization."""
        current_verbose = kwargs.get("verbose", self.verbose)
        audio_processor: AudioProcessor = self.get_audio_file(audio_file)
        if current_verbose: print("Starting transcription (direct call)...")
        
        # Transcriber's transcribe method already filters its specific kwargs
        transcription_result = self.transcriber.transcribe(audio_processor.waveform, **kwargs)
        return transcription_result

    def update_transcriber(self,
                           whisper_model: Union[str, Transcriber, None],
                           whisper_type: Optional[str] = None,
                           download_root: Optional[str] = None,
                           use_auth_token: Optional[str] = None,
                           **component_kwargs: Any) -> None:
        """Updates or replaces the current transcriber instance."""
        _old_model_name = self.transcriber.model_name if self.transcriber else "None"
        
        # Determine effective whisper_type: use provided, or from saved params, or default
        effective_whisper_type = whisper_type
        if effective_whisper_type is None:
            effective_whisper_type = self.params.get("whisper_type", "openai-ipex-llm") if self.params else "openai-ipex-llm"

        transcriber_load_kwargs = component_kwargs.copy()
        transcriber_load_kwargs['device_option'] = self.target_device
        if download_root:
            transcriber_load_kwargs['download_root'] = download_root
        if use_auth_token:
             transcriber_load_kwargs['use_auth_token'] = use_auth_token

        if isinstance(whisper_model, Transcriber):
            self.transcriber = whisper_model
            if self.verbose: print(f"Transcriber updated to provided instance: {self.transcriber}")
        elif isinstance(whisper_model, str) or whisper_model is None:
            effective_whisper_model_name = whisper_model if whisper_model is not None else "medium"
            if self.verbose: print(f"Updating Transcriber to: model='{effective_whisper_model_name}', type='{effective_whisper_type}'")
            self.transcriber = load_transcriber(
                model_name=effective_whisper_model_name,
                whisper_type=effective_whisper_type,
                **transcriber_load_kwargs
            )
        else:
            warnings.warn(
                f"Invalid type for whisper_model: {type(whisper_model)}. "
                f"Expected model name (str), Transcriber instance, or None. "
                f"Transcriber not updated, fallback to old '{_old_model_name}' model.", RuntimeWarning)
        # Update self.params if they were saved
        if self.params:
            self.params['whisper_model'] = str(whisper_model.model_name) if isinstance(whisper_model, Transcriber) else whisper_model
            self.params['whisper_type'] = effective_whisper_type
            if download_root: self.params['download_root'] = download_root
            if use_auth_token: self.params['use_auth_token'] = "****"


    def update_diariser(self,
                        dia_model: Union[str, Diariser, None],
                        use_auth_token: Optional[str] = None,
                        **component_kwargs: Any) -> None:
        """Updates or replaces the current diariser instance."""
        _old_diariser_info = str(self.diariser.model_name if hasattr(self.diariser, 'model_name') and self.diariser.model_name else self.diariser) if self.diariser else "None"

        diariser_load_kwargs = component_kwargs.copy()
        if use_auth_token:
            diariser_load_kwargs['use_auth_token'] = use_auth_token # Or 'token'

        if isinstance(dia_model, Diariser):
            self.diariser = dia_model
            if self.verbose: print(f"Diariser updated to provided instance: {self.diariser}")
        elif isinstance(dia_model, str) or dia_model is None:
            if self.verbose: print(f"Updating Diariser to: model='{dia_model if dia_model else 'default'}'")
            self.diariser = Diariser.load_model(
                model_name_or_path=dia_model,
                device=self.target_device,
                **diariser_load_kwargs
            )
        else:
            warnings.warn(
                f"Invalid type for dia_model: {type(dia_model)}. "
                f"Expected model name (str), Diariser instance, or None. "
                f"Diariser not updated, fallback to old '{_old_diariser_info}'.", RuntimeWarning)
        # Update self.params if they were saved
        if self.params:
            self.params['dia_model'] = str(dia_model.model_name) if isinstance(dia_model, Diariser) and hasattr(dia_model, 'model_name') else dia_model
            if use_auth_token and 'use_auth_token' in self.params: self.params['use_auth_token'] = "****" # Only update if it was part of init params

    @staticmethod
    def remove_audio_file(audio_file_path: str, shred: bool = False) -> None:
        """Removes or shreds the specified audio file."""
        if not isinstance(audio_file_path, str):
            warnings.warn(f"remove_audio_file expects a path string, got {type(audio_file_path)}. Skipping removal.")
            return
            
        if not os.path.exists(audio_file_path):
            warnings.warn(f"Audio file {audio_file_path} does not exist. Skipping removal.")
            return

        try:
            if shred:
                # Ensure `shred` is available and path is secure if from untrusted input.
                # This basic command works on many Linux/macOS systems.
                # Consider adding checks for `shred` command availability or using a cross-platform library for secure deletion.
                print(f"Shredding {audio_file_path}... This may take some time.")
                run(['shred', '-zvu', '-n', '1', audio_file_path], check=True, capture_output=True)
                print(f"Audio file {audio_file_path} shredded and removed.")
            else:
                os.remove(audio_file_path)
                print(f"Audio file {audio_file_path} removed.")
        except FileNotFoundError: # Should be caught by os.path.exists, but as a safeguard
             warnings.warn(f"Audio file {audio_file_path} was not found for removal (race condition or already deleted).")
        except Exception as e: # More specific exceptions like CalledProcessError for run
            warnings.warn(f"Error removing/shredding audio file {audio_file_path}: {e}")

    @staticmethod
    def get_audio_file(audio_source: Union[str, torch.Tensor, np.ndarray, AudioProcessor],
                       default_sample_rate: int = 16000) -> AudioProcessor:
        """
        Processes various audio inputs into a consistent AudioProcessor object.

        Args:
            audio_source (Union[str, torch.Tensor, np.ndarray, AudioProcessor]):
                The audio input. Can be a file path (str), a raw waveform (PyTorch Tensor or NumPy ndarray),
                or an existing AudioProcessor instance.
            default_sample_rate (int, optional): Sample rate to assume if a raw waveform Tensor/ndarray
                is provided without explicit sample rate information. Defaults to 16000.

        Returns:
            AudioProcessor: An AudioProcessor instance ready for use.
        
        Raises:
            TypeError: If the audio_source type is unsupported.
            # Add other potential exceptions from AudioProcessor loading
        """
        if isinstance(audio_source, AudioProcessor):
            return audio_source
        if isinstance(audio_source, str):
            # AudioProcessor.from_file should handle path validation and loading.
            return AudioProcessor.from_file(audio_source)
        elif isinstance(audio_source, torch.Tensor):
            waveform = audio_source
            if waveform.ndim > 1 and waveform.shape[0] == 1: # Standardize to (num_samples,)
                waveform = waveform.squeeze(0)
            elif waveform.ndim > 1:
                 warnings.warn(f"Input Tensor has {waveform.ndim} dimensions. Expected 1D or (1, N) waveform. Using as is.")
            # Assuming AudioProcessor constructor is: AudioProcessor(waveform: Tensor, sample_rate: int)
            return AudioProcessor(waveform=waveform, sample_rate=default_sample_rate)
        elif isinstance(audio_source, np.ndarray):
            waveform_tensor = torch.from_numpy(audio_source.astype(np.float32))
            if waveform_tensor.ndim > 1 and waveform_tensor.shape[0] == 1: # Standardize
                waveform_tensor = waveform_tensor.squeeze(0)
            elif waveform_tensor.ndim > 1:
                 warnings.warn(f"Input ndarray has {waveform_tensor.ndim} dimensions. Expected 1D or (1, N) waveform. Using as is.")
            return AudioProcessor(waveform=waveform_tensor, sample_rate=default_sample_rate)
            
        raise TypeError(f"Unsupported audio_source type: {type(audio_source)}. "
                        "Expected path (str), waveform (Tensor/ndarray), or AudioProcessor instance.")

    def __repr__(self):
        transcriber_name = self.transcriber.model_name if self.transcriber and hasattr(self.transcriber, 'model_name') else "N/A"
        diariser_name = self.diariser.model_name if self.diariser and hasattr(self.diariser, 'model_name') else "N/A"
        return (f"Scraibe(whisper_model='{transcriber_name}', diariser_model='{diariser_name}', "
                f"target_device='{self.target_device}')")