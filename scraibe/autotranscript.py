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
                 dia_model: Union[str, Diariser, None] = None, # Configuration for the diarizer
                 target_device: Optional[Union[str, torch.device]] = None,
                 download_root: Optional[str] = None,
                 use_auth_token: Optional[str] = None,
                 verbose: bool = False,
                 save_setup: bool = False,
                 **component_kwargs: Any) -> None:
        """Initializes the Scraibe class.
        Transcriber is loaded immediately. Diarizer is loaded lazily on first use.
        """
        self.verbose = verbose
        self.target_device = torch.device(target_device if target_device is not None else SCRAIBE_TORCH_DEVICE)
        
        # Store arguments needed for lazy loading of diarizer and other components
        self._dia_model_config_arg = dia_model # Store the original arg for diarizer (name, instance, or None)
        self._use_auth_token_for_components = use_auth_token # Token for both transcriber and diarizer
        self._component_kwargs_for_loaders = component_kwargs.copy() # General kwargs for loaders

        if self.verbose:
            print(f"Scraibe __init__: target_device='{self.target_device}', whisper_model='{whisper_model}', "
                  f"dia_model_config='{self._dia_model_config_arg}' (will load on demand)")

        # --- Initialize Transcriber (always loaded immediately) ---
        transcriber_load_kwargs = self._component_kwargs_for_loaders.copy()
        if download_root:
            transcriber_load_kwargs['download_root'] = download_root
        if self._use_auth_token_for_components: # Pass token to transcriber if provided
             transcriber_load_kwargs['use_auth_token'] = self._use_auth_token_for_components

        if isinstance(whisper_model, Transcriber):
            self.transcriber = whisper_model
            if self.verbose: print(f"Using provided Transcriber instance: {self.transcriber}")
        else: 
            effective_whisper_model_name = whisper_model if whisper_model is not None else "medium"
            if self.verbose: 
                print(f"Loading Transcriber: model='{effective_whisper_model_name}', type='{whisper_type}', device='{self.target_device}'")
            self.transcriber = load_transcriber(
                model_name=effective_whisper_model_name,
                whisper_type=whisper_type,
                device=self.target_device,
                **transcriber_load_kwargs
            )

        # --- Diarizer: Mark as not loaded yet ---
        self._diariser_instance: Optional[Diariser] = None
        # If a pre-loaded Diariser instance was passed to __init__, use it directly
        if isinstance(self._dia_model_config_arg, Diariser):
            self._diariser_instance = self._dia_model_config_arg
            if self.verbose: print(f"Using provided Diariser instance directly: {self._diariser_instance}")

        if self.verbose:
            if self._diariser_instance:
                 print(f"Scraibe initialized. Transcriber: {self.transcriber}. Diariser: {self._diariser_instance} (pre-loaded).")
            else:
                 print(f"Scraibe initialized. Transcriber: {self.transcriber}. Diariser will be loaded on demand.")


        self.params = {}
        if save_setup:
            self.params = dict(
                whisper_model=str(whisper_model.model_name) if isinstance(whisper_model, Transcriber) else whisper_model,
                whisper_type=whisper_type,
                dia_model=str(self._dia_model_config_arg.model_name) if isinstance(self._dia_model_config_arg, Diariser) and hasattr(self._dia_model_config_arg, 'model_name') else self._dia_model_config_arg,
                target_device=str(self.target_device),
                download_root=download_root,
                use_auth_token="****" if self._use_auth_token_for_components else None,
                verbose=verbose,
                **self._component_kwargs_for_loaders
            )

    @property
    def diariser(self) -> Diariser:
        """
        Property to lazily load the Diariser instance when first accessed.
        """
        if self._diariser_instance is None:
            if self.verbose: print("Diariser property accessed: attempting to load diarization model...")

            # Prepare kwargs for Diariser.load_model
            # Use a subset of _component_kwargs_for_loaders if necessary, or specific ones
            diariser_load_kwargs = {} # Start fresh or copy from self._component_kwargs_for_loaders and filter
            if self._use_auth_token_for_components:
                diariser_load_kwargs['use_auth_token'] = self._use_auth_token_for_components
            # Add any other specific kwargs from self._component_kwargs_for_loaders if Diariser.load_model expects them.
            # Example: if 'some_diarizer_specific_param' in self._component_kwargs_for_loaders:
            # diariser_load_kwargs['some_diarizer_specific_param'] = self._component_kwargs_for_loaders['some_diarizer_specific_param']

            # _dia_model_config_arg holds the original 'dia_model' argument from __init__
            # It could be a string (model name/path), or None (for default behavior in Diariser.load_model)
            effective_dia_model_identifier = self._dia_model_config_arg

            if self.verbose:
                print(f"Lazy loading Diariser: model_identifier='{effective_dia_model_identifier if effective_dia_model_identifier else 'default (handled by Diariser.load_model)'}', "
                      f"device='{self.target_device}'")
            
            try:
                self._diariser_instance = Diariser.load_model(
                    model=effective_dia_model_identifier, # Pass string, path, or None
                    device=self.target_device,
                    **diariser_load_kwargs
                )
                if self.verbose: print(f"Diariser loaded successfully: {self._diariser_instance}")
            except Exception as e:
                warnings.warn(f"Failed to lazy-load Diariser with model_identifier='{effective_dia_model_identifier}': {e}", RuntimeWarning)
                # Depending on desired behavior, you might:
                # 1. Raise the error to stop execution if diarizer is critical.
                # 2. Set self._diariser_instance to a dummy/non-functional diarizer.
                # 3. Leave self._diariser_instance as None and let calling methods handle it.
                # For now, re-raising is clearest about the failure.
                raise RuntimeError(f"Diariser could not be loaded for identifier '{effective_dia_model_identifier}'.") from e
        
        # This check is essentially for post-successful load, though load_model should raise on failure.
        if self._diariser_instance is None:
            # This state should ideally not be reached if load_model either succeeds or raises.
            raise RuntimeError("Diariser is None after attempting to load; this indicates an unexpected state.")
            
        return self._diariser_instance

    def autotranscribe(self, audio_file: Union[str, torch.Tensor, np.ndarray],
                       remove_original: bool = False,
                       **kwargs: Any) -> Transcript:
        """
        Transcribes an audio file using the whisper model and pyannote diarization model.
        Segments the audio by speaker first, then transcribes each segment.
        """
        current_verbose = kwargs.get("verbose", self.verbose)
        audio_processor: AudioProcessor = self.get_audio_file(audio_file)

        # Accessing self.diariser (the property) will trigger its lazy loading if not already loaded
        if self.verbose: print("Autotranscribe: Accessing diariser...")
        try:
            active_diariser = self.diariser # Trigger load if needed
        except RuntimeError as e:
            warnings.warn(f"Cannot perform autotranscription because Diariser failed to load: {e}. "
                          "Proceeding with full audio transcription without diarization.", RuntimeWarning)
            # Fallback: Transcribe entire audio as a single speaker
            transcription_result = self.transcriber.transcribe(
                audio_processor.waveform, 
                task=kwargs.get("task", "transcribe"), 
                language=kwargs.get("language"),
                verbose=current_verbose
            )
            duration_s = len(audio_processor.waveform) / audio_processor.sr
            final_transcript_data = {
                0: { "speakers": 'SPEAKER_00', "segments": [0.0, duration_s], "text": transcription_result.get("text", "")}
            }
            return Transcript(final_transcript_data)

        # Proceed with diarization
        dia_waveform = audio_processor.waveform.reshape(1, -1)
        if dia_waveform.dtype != torch.float32:
            dia_waveform = dia_waveform.to(dtype=torch.float32)

        dia_audio_input = {
            "waveform": dia_waveform.to(active_diariser.device), # Use the actual device of the loaded diariser
            "sample_rate": audio_processor.sr
        }
        
        if current_verbose: print("Starting diarisation for autotranscribe...")
        
        diarization_params = {k: v for k, v in kwargs.items() if k in active_diariser.diarization.__code__.co_varnames}
        if 'num_speakers' in kwargs and 'num_speakers' not in diarization_params:
            diarization_params['num_speakers'] = kwargs['num_speakers']
        
        diarisation_result = active_diariser.diarization(dia_audio_input, **diarization_params)

        if not diarisation_result.get("segments"):
            if current_verbose:
                print("No speaker segments found by diariser. Transcribing entire audio as a single speaker.")
            transcription_result = self.transcriber.transcribe(
                audio_processor.waveform, 
                task=kwargs.get("task", "transcribe"), 
                language=kwargs.get("language"),
                verbose=current_verbose
            )
            duration_s = len(audio_processor.waveform) / audio_processor.sr
            final_transcript_data = {
                0: { "speakers": 'SPEAKER_01', "segments": [0.0, duration_s], "text": transcription_result.get("text", "")}
            }
            return Transcript(final_transcript_data)

        if current_verbose:
            print(f"Diarisation finished with {len(diarisation_result['segments'])} segments. Starting transcription per segment.")

        final_transcript_data = {}
        segments_data = list(zip(diarisation_result.get("segments", []), diarisation_result.get("speakers", [])))
        
        # Setup tqdm iterator if verbose and segments exist
        segments_iterator = segments_data
        if current_verbose and segments_data:
            segments_iterator = trange(len(segments_data), desc="Transcribing Segments", unit="segment")

        for i, (seg_start_end, speaker_label) in enumerate(segments_data if not (current_verbose and segments_data) else segments_iterator):
            if isinstance(segments_iterator, trange): # If using tqdm, get original data
                 seg_start_end, speaker_label = segments_data[i]

            if not isinstance(seg_start_end, (list, tuple)) or len(seg_start_end) != 2:
                if current_verbose: print(f"Skipping invalid segment data: {seg_start_end}")
                continue
            seg_start, seg_end = seg_start_end

            audio_segment_waveform = audio_processor.cut(seg_start, seg_end)

            segment_transcription_result = self.transcriber.transcribe(
                audio_segment_waveform, 
                task=kwargs.get("task", "transcribe"),
                language=kwargs.get("language"),
                verbose=False # Typically, don't want Whisper's verbose for each small segment
            )
            final_transcript_data[i] = {
                "speakers": speaker_label,
                "segments": [seg_start, seg_end],
                "text": segment_transcription_result.get("text", "")
            }
            if isinstance(segments_iterator, trange): # Update tqdm description
                 segments_iterator.set_description_str(f"Transcribing segment {i+1}/{len(segments_data)}")
        
        if isinstance(segments_iterator, trange): # Close tqdm iterator
            segments_iterator.close()

        if remove_original and isinstance(audio_file, str):
            shred_flag = kwargs.get("shred", False)
            self.remove_audio_file(audio_file, shred=shred_flag)

        return Transcript(final_transcript_data)


    def diarization(self, audio_file: Union[str, torch.Tensor, np.ndarray],
                    **kwargs: Any) -> Dict[str, Any]:
        """Performs speaker diarization on the audio."""
        current_verbose = kwargs.get("verbose", self.verbose)
        audio_processor: AudioProcessor = self.get_audio_file(audio_file)

        # Accessing self.diariser (the property) will trigger its lazy loading
        if self.verbose: print("Diarization method: Accessing diariser...")
        active_diariser = self.diariser # Trigger load

        dia_waveform = audio_processor.waveform.reshape(1, -1)
        if dia_waveform.dtype != torch.float32:
            dia_waveform = dia_waveform.to(dtype=torch.float32)

        dia_audio_input = {
            "waveform": dia_waveform.to(active_diariser.device), # Use device of loaded diariser
            "sample_rate": audio_processor.sr
        }
        if current_verbose: print("Starting diarisation (direct call)...")
        
        diarization_params = {k: v for k, v in kwargs.items() if k in active_diariser.diarization.__code__.co_varnames}
        if 'num_speakers' in kwargs and 'num_speakers' not in diarization_params:
             diarization_params['num_speakers'] = kwargs['num_speakers']

        diarisation_result = active_diariser.diarization(dia_audio_input, **diarization_params)
        return diarisation_result


    def transcribe(self, audio_file: Union[str, torch.Tensor, np.ndarray],
                   **kwargs: Any) -> Dict[str, Any]:
        """Transcribes the entire audio file without speaker diarization.
           This method does NOT load or use the diarizer.
        """
        current_verbose = kwargs.get("verbose", self.verbose)
        audio_processor: AudioProcessor = self.get_audio_file(audio_file)
        if current_verbose: print("Starting transcription (direct call)... Diarizer will not be loaded.")
        
        transcription_result = self.transcriber.transcribe(audio_processor.waveform, **kwargs)
        return transcription_result

    def update_transcriber(self,
                           whisper_model: Union[str, Transcriber, None],
                           whisper_type: Optional[str] = None,
                           download_root: Optional[str] = None,
                           use_auth_token: Optional[str] = None, # Changed from self._use_auth_token_for_components
                           **component_kwargs: Any) -> None: # component_kwargs specific to transcriber
        """Updates or replaces the current transcriber instance."""
        _old_model_name = self.transcriber.model_name if self.transcriber and hasattr(self.transcriber, 'model_name') else "None"
        
        effective_whisper_type = whisper_type
        if effective_whisper_type is None:
            effective_whisper_type = self.params.get("whisper_type", "openai-ipex-llm") if self.params else "openai-ipex-llm"

        transcriber_load_kwargs = component_kwargs.copy()
        # Note: device_option is not set here; load_transcriber's `device` param is used.
        if download_root:
            transcriber_load_kwargs['download_root'] = download_root
        current_use_auth_token = use_auth_token if use_auth_token is not None else self._use_auth_token_for_components
        if current_use_auth_token:
             transcriber_load_kwargs['use_auth_token'] = current_use_auth_token


        if isinstance(whisper_model, Transcriber):
            self.transcriber = whisper_model
            if self.verbose: print(f"Transcriber updated to provided instance: {self.transcriber}")
        elif isinstance(whisper_model, str) or whisper_model is None:
            effective_whisper_model_name = whisper_model if whisper_model is not None else "medium"
            if self.verbose: print(f"Updating Transcriber to: model='{effective_whisper_model_name}', type='{effective_whisper_type}', device='{self.target_device}'")
            self.transcriber = load_transcriber(
                model_name=effective_whisper_model_name,
                whisper_type=effective_whisper_type,
                device=self.target_device, # Pass main target_device
                **transcriber_load_kwargs
            )
        else:
            warnings.warn(
                f"Invalid type for whisper_model: {type(whisper_model)}. "
                f"Expected model name (str), Transcriber instance, or None. "
                f"Transcriber not updated, fallback to old '{_old_model_name}' model.", RuntimeWarning)

        if self.params:
            self.params['whisper_model'] = str(whisper_model.model_name) if isinstance(whisper_model, Transcriber) else whisper_model
            self.params['whisper_type'] = effective_whisper_type
            if download_root: self.params['download_root'] = download_root
            if current_use_auth_token: self.params['use_auth_token'] = "****"


    def update_diariser(self,
                        dia_model: Union[str, Diariser, None], # This is the new config or instance
                        use_auth_token: Optional[str] = None,
                        **component_kwargs: Any) -> None:
        """Updates the diariser configuration. The new diariser will be loaded on its next use."""
        old_config_info = self._dia_model_config_arg
        if isinstance(old_config_info, Diariser) and hasattr(old_config_info, 'model_name'):
            old_config_info = old_config_info.model_name
        elif isinstance(old_config_info, Diariser):
            old_config_info = "Instance"
            
        if self.verbose:
            new_config_info = dia_model
            if isinstance(dia_model, Diariser) and hasattr(dia_model, 'model_name'): new_config_info = dia_model.model_name
            elif isinstance(dia_model, Diariser): new_config_info = "Instance"
            print(f"Updating diariser configuration from '{old_config_info}' to '{new_config_info}'. Will reload on next use.")
        
        self._dia_model_config_arg = dia_model # Store the new config (str, None) or pre-loaded instance
        self._diariser_instance = None          # Reset to force reload with new config on next access
        
        # Update stored token and component_kwargs for diarizer if provided
        if use_auth_token is not None: # Allow specific update of token for diarizer
            self._use_auth_token_for_components = use_auth_token # This might impact transcriber too if not careful.
                                                                # Better to have separate tokens or pass them down.
                                                                # For now, assuming one main token.
                                                                # OR: self._use_auth_token_for_diarizer = use_auth_token (if you add this field)

        # Update component_kwargs for diarizer. Be selective.
        # This example replaces them if new ones are provided, or you could merge.
        self._component_kwargs_for_loaders.update({ 
            k: v for k, v in component_kwargs.items() 
            # if k in relevant_diarizer_component_kwargs # Only update relevant ones
        })


        # If an actual Diariser instance is passed in `dia_model`, use it directly, bypassing lazy load for this update.
        if isinstance(dia_model, Diariser):
            self._diariser_instance = dia_model
            if self.verbose: print(f"Diariser immediately updated to provided instance: {self._diariser_instance}")
        
        # Update self.params if they were saved
        if self.params and 'dia_model' in self.params:
            self.params['dia_model'] = str(dia_model.model_name) if isinstance(dia_model, Diariser) and hasattr(dia_model, 'model_name') else dia_model
            if use_auth_token is not None and 'use_auth_token' in self.params: self.params['use_auth_token'] = "****"

    # Ensure AudioProcessor, Diariser, Transcriber, Transcript, SCRAIBE_TORCH_DEVICE etc. are imported correctly
    # Keep your get_audio_file, remove_audio_file, and __repr__ methods as they were,
    # or adapt __repr__ to show diariser state (loaded/config).

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
                print(f"Shredding {audio_file_path}... This may take some time.")
                run(['shred', '-zvu', '-n', '1', audio_file_path], check=True, capture_output=True)
                print(f"Audio file {audio_file_path} shredded and removed.")
            else:
                os.remove(audio_file_path)
                print(f"Audio file {audio_file_path} removed.")
        except FileNotFoundError: 
             warnings.warn(f"Audio file {audio_file_path} was not found for removal (already deleted).")
        except Exception as e: 
            warnings.warn(f"Error removing/shredding audio file {audio_file_path}: {e}")

    @staticmethod
    def get_audio_file(audio_source: Union[str, torch.Tensor, np.ndarray, AudioProcessor],
                       default_sample_rate: int = 16000) -> AudioProcessor:
        """
        Processes various audio inputs into a consistent AudioProcessor object.
        """
        if isinstance(audio_source, AudioProcessor):
            return audio_source
        if isinstance(audio_source, str):
            return AudioProcessor.from_file(audio_source)
        elif isinstance(audio_source, torch.Tensor):
            waveform = audio_source
            if waveform.ndim > 1 and waveform.shape[0] == 1: 
                waveform = waveform.squeeze(0)
            elif waveform.ndim > 1:
                 warnings.warn(f"Input Tensor has {waveform.ndim} dimensions. Expected 1D or (1, N) waveform.")
            return AudioProcessor(waveform=waveform, sample_rate=default_sample_rate)
        elif isinstance(audio_source, np.ndarray):
            waveform_tensor = torch.from_numpy(audio_source.astype(np.float32))
            if waveform_tensor.ndim > 1 and waveform_tensor.shape[0] == 1: 
                waveform_tensor = waveform_tensor.squeeze(0)
            elif waveform_tensor.ndim > 1:
                 warnings.warn(f"Input ndarray has {waveform_tensor.ndim} dimensions. Expected 1D or (1, N) waveform.")
            return AudioProcessor(waveform=waveform_tensor, sample_rate=default_sample_rate)
            
        raise TypeError(f"Unsupported audio_source type: {type(audio_source)}. "
                        "Expected path (str), waveform (Tensor/ndarray), or AudioProcessor instance.")

    def __repr__(self):
        transcriber_name = self.transcriber.model_name if self.transcriber and hasattr(self.transcriber, 'model_name') else "N/A"
        
        diariser_info = "Not loaded"
        if self._diariser_instance:
            diariser_name = self._diariser_instance.model_name if hasattr(self._diariser_instance, 'model_name') else "Instance"
            diariser_info = f"Loaded({diariser_name})"
        elif self._dia_model_config_arg:
            config_name = self._dia_model_config_arg
            if isinstance(self._dia_model_config_arg, Diariser) and hasattr(self._dia_model_config_arg, 'model_name'): # Should not happen if _diariser_instance is None
                config_name = self._dia_model_config_arg.model_name
            diariser_info = f"Configured({config_name})"
        
        return (f"Scraibe(whisper_model='{transcriber_name}', diariser='{diariser_info}', "
                f"target_device='{self.target_device}')")
                
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