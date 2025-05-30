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

    model = Scraibe()
    transcript = model.autotranscribe("path/to/audiofile.wav")
"""

# Standard Library Imports
import os
from subprocess import run
from typing import TypeVar, Union, Optional, Any, Dict # Added Any

# Third-Party Imports
import torch
from numpy import ndarray
# import intel_extension_for_pytorch as ipex # This import is not used directly here. 
                                          # IPEX-LLM is used in transcriber.py
from tqdm import trange
from glob import iglob

# Application-Specific Imports
from .audio import AudioProcessor
from .diarisation import Diariser
from .transcriber import Transcriber, load_transcriber # Removed 'whisper' import
from .transcript_exporter import Transcript
from .misc import SCRAIBE_TORCH_DEVICE, WHISPER_DEFAULT_PATH # Added WHISPER_DEFAULT_PATH for consistency

DiarisationType = TypeVar('DiarisationType') # For Diariser model instances

class Scraibe:
    """
    Scraibe is a class responsible for managing the transcription and diarization of audio files.
    # ... (rest of docstring)
    """

    def __init__(self,
                 whisper_model_or_name: Union[str, Transcriber, None] = None, # Changed type hint
                 whisper_type: str = "openai-ipex-llm", # Defaulting to our new IPEX-LLM integrated type
                 dia_model_or_name: Union[str, Diariser, None] = None, # Consistent naming and typing
                 device: Optional[Union[str, torch.device]] = None, # Allow device override here
                 **kwargs: Any) -> None:
        """Initializes the Scraibe class.

        Args:
            whisper_model_or_name (Union[str, Transcriber, None], optional): 
                                Whisper model name (e.g., "medium"), an existing Transcriber instance,
                                or None to load the default "medium" model.
            whisper_type (str): Type of whisper model to load if whisper_model_or_name is a string.
                                E.g., "openai-ipex-llm", "faster-whisper".
            dia_model_or_name (Union[str, Diariser, None], optional): 
                                Pyannote diarization model name, an existing Diariser instance,
                                or None to load the default.
            device (Optional[Union[str, torch.device]]): Device to use (e.g. "cpu", "xpu", "cuda").
                                                         Overrides SCRAIBE_TORCH_DEVICE if set.
            **kwargs: Additional keyword arguments for model loading and processing.
                    e.g.:
                    - verbose: If True, the class will print additional information.
                    - save_setup: If True, the initialization parameters will be saved.
                    - IPEX-LLM specific args like `low_bit` for `openai-ipex-llm` whisper_type.
                    - FasterWhisper specific args like `compute_type` for `faster-whisper` type.
        """
        self.verbose = kwargs.get("verbose", False)
        self.target_device = device if device is not None else SCRAIBE_TORCH_DEVICE
        kwargs["device"] = self.target_device # Ensure device is in kwargs for load_transcriber

        # Initialize Transcriber
        if isinstance(whisper_model_or_name, Transcriber):
            self.transcriber = whisper_model_or_name
            if self.verbose: print(f"Using provided Transcriber instance: {self.transcriber}")
        elif isinstance(whisper_model_or_name, str):
            self.transcriber = load_transcriber(
                model_name=whisper_model_or_name, whisper_type=whisper_type, **kwargs)
        elif whisper_model_or_name is None:
            self.transcriber = load_transcriber(
                model_name="medium", whisper_type=whisper_type, **kwargs) # Default model name
        else:
            raise TypeError(f"Unsupported type for whisper_model_or_name: {type(whisper_model_or_name)}. "
                            "Expected model name (str), Transcriber instance, or None.")

        # Initialize Diariser
        if isinstance(dia_model_or_name, Diariser):
            self.diariser = dia_model_or_name
            if self.verbose: print(f"Using provided Diariser instance: {self.diariser}")
        elif isinstance(dia_model_or_name, str):
            self.diariser = Diariser.load_model(dia_model_or_name, device=self.target_device, **kwargs)
        elif dia_model_or_name is None:
            self.diariser = Diariser.load_model(device=self.target_device, **kwargs) # Default diariser model
        else:
            raise TypeError(f"Unsupported type for dia_model_or_name: {type(dia_model_or_name)}. "
                            "Expected model name (str), Diariser instance, or None.")


        if self.verbose:
            print(f"Scraibe initialized. Transcriber: {self.transcriber}, Diariser: {self.diariser} on device '{self.target_device}'")

        self.params = {}
        if kwargs.get('save_setup'):
            self.params = dict(whisper_model_or_name=str(whisper_model_or_name) if isinstance(whisper_model_or_name, Transcriber) else whisper_model_or_name, # Store name if instance
                               whisper_type=whisper_type,
                               dia_model_or_name=str(dia_model_or_name) if isinstance(dia_model_or_name, Diariser) else dia_model_or_name,
                               device=str(self.target_device),
                               **kwargs)
            
    def autotranscribe(self, audio_file: Union[str, torch.Tensor, ndarray],
                       remove_original: bool = False,
                       **kwargs: Any) -> Transcript:
        """
        Transcribes an audio file using the whisper model and pyannote diarization model.
        # ... (rest of docstring)
        """
        current_verbose = kwargs.get("verbose", self.verbose) # Allow overriding verbosity for this call

        audio_file_processed: AudioProcessor = self.get_audio_file(audio_file)

        # Ensure waveform is float32 for diarization, as float16 on CPU can be problematic for some ops
        # Diariser might also move to its own required device/dtype.
        # Assuming self.diariser.device handles this.
        dia_waveform = audio_file_processed.waveform.reshape(1, -1) # Batch dim, ensure it's 1D first
        if dia_waveform.dtype != torch.float32:
            dia_waveform = dia_waveform.to(dtype=torch.float32)

        # Diarization expects waveform on its model's device.
        # Assuming self.diariser.device correctly reflects this.
        dia_audio_input = {
            "waveform": dia_waveform.to(self.diariser.device), 
            "sample_rate": audio_file_processed.sr
        }
        
        if current_verbose:
            print("Starting diarisation...")

        # Pass through relevant diarization kwargs
        diarisation_kwargs = {k:v for k,v in kwargs.items() if k in self.diariser.diarization.__code__.co_varnames}
        diarisation = self.diariser.diarization(dia_audio_input, **diarisation_kwargs)

        if not diarisation.get("segments"): # Check if segments list is empty or key missing
            if current_verbose:
                print("No segments found by diariser. Transcribing entire audio as a single speaker.")

            # Transcribe entire audio, result is a dict {"text": ..., "segments": ...}
            # The transcriber itself will use its model's device.
            transcription_result = self.transcriber.transcribe(
                audio_file_processed.waveform, **kwargs) # Pass all other kwargs to transcriber
            
            # Create a compatible structure for Transcript object
            duration_s = len(audio_file_processed.waveform) / audio_file_processed.sr
            final_transcript_data = {
                0: {
                    "speakers": 'SPEAKER_01', # Default speaker
                    "segments": [0.0, duration_s], # Full duration
                    "text": transcription_result.get("text", "") # Use the text from transcription
                }
            }
            return Transcript(final_transcript_data)

        if current_verbose:
            print(f"Diarisation finished with {len(diarisation['segments'])} segments. Starting transcription per segment.")

        final_transcript_data = {}
        for i in trange(len(diarisation["segments"]), desc="Transcribing Segments", disable=not current_verbose, unit="segment"):
            seg_start, seg_end = diarisation["segments"][i] # Assuming segments are [start, end] tuples
            speaker = diarisation["speakers"][i]

            # AudioProcessor.cut returns a new AudioProcessor object or raw waveform
            audio_segment_waveform = audio_file_processed.cut(seg_start, seg_end) # Returns waveform directly

            # Transcriber expects waveform Tensor or ndarray
            # The transcriber.transcribe method will handle its device placement
            segment_transcription_result = self.transcriber.transcribe(audio_segment_waveform, **kwargs)

            final_transcript_data[i] = {
                "speakers": speaker,
                "segments": [seg_start, seg_end],
                "text": segment_transcription_result.get("text", "")
            }

        if remove_original and isinstance(audio_file, str): # Only remove if original was a file path
            shred_flag = kwargs.get("shred", False)
            self.remove_audio_file(audio_file, shred=shred_flag)

        return Transcript(final_transcript_data)

    def diarization(self, audio_file: Union[str, torch.Tensor, ndarray],
                    **kwargs: Any) -> Dict[str, Any]:
        # ... (implementation as before, ensure dia_audio_input uses self.diariser.device)
        audio_file_processed: AudioProcessor = self.get_audio_file(audio_file)
        dia_waveform = audio_file_processed.waveform.reshape(1, -1)
        if dia_waveform.dtype != torch.float32:
            dia_waveform = dia_waveform.to(dtype=torch.float32)

        dia_audio_input = {
            "waveform": dia_waveform.to(self.diariser.device),
            "sample_rate": audio_file_processed.sr 
        }
        if self.verbose: print("Starting diarisation (direct call)...")
        diarisation_kwargs = {k:v for k,v in kwargs.items() if k in self.diariser.diarization.__code__.co_varnames}
        diarisation_result = self.diariser.diarization(dia_audio_input, **diarisation_kwargs)
        return diarisation_result


    def transcribe(self, audio_file: Union[str, torch.Tensor, ndarray],
                   **kwargs: Any) -> Dict[str, Any]: # Ensure return type matches ABC
        # ... (implementation as before, ensure it returns a dict)
        audio_file_processed: AudioProcessor = self.get_audio_file(audio_file)
        if self.verbose: print("Starting transcription (direct call)...")
        transcription_result = self.transcriber.transcribe(audio_file_processed.waveform, **kwargs)
        return transcription_result # This is already a dict from refactored transcriber

    def update_transcriber(self,
                           whisper_model_or_name: Union[str, Transcriber, None], # Updated type hint
                           whisper_type: Optional[str] = None, # Allow changing type
                           **kwargs: Any) -> None:
        _old_model_name = self.transcriber.model_name if self.transcriber else "None"
        
        effective_whisper_type = whisper_type if whisper_type is not None else \
                                (self.params.get("whisper_type", "openai-ipex-llm") if self.params else "openai-ipex-llm")
        
        kwargs["device"] = self.target_device # Ensure device is passed

        if isinstance(whisper_model_or_name, Transcriber):
            self.transcriber = whisper_model_or_name
            if self.verbose: print(f"Transcriber updated to provided instance: {self.transcriber}")
        elif isinstance(whisper_model_or_name, str):
            self.transcriber = load_transcriber(
                model_name=whisper_model_or_name, whisper_type=effective_whisper_type, **kwargs)
            if self.verbose: print(f"Transcriber updated to loaded model: {self.transcriber}")
        elif whisper_model_or_name is None: # Option to reset to default
             self.transcriber = load_transcriber(
                model_name="medium", whisper_type=effective_whisper_type, **kwargs)
             if self.verbose: print(f"Transcriber reset to default model: {self.transcriber}")
        else:
            warnings.warn(
                f"Invalid type for whisper_model_or_name: {type(whisper_model_or_name)}. "
                f"Expected model name (str), Transcriber instance, or None. "
                f"Transcriber not updated, fallback to old '{_old_model_name}' model.", RuntimeWarning)
        return None

    def update_diariser(self,
                        dia_model_or_name: Union[str, Diariser, None], # Consistent naming and typing
                        **kwargs: Any) -> None:
        _old_diariser_info = str(self.diariser) if self.diariser else "None"
        kwargs["device"] = self.target_device

        if isinstance(dia_model_or_name, Diariser):
            self.diariser = dia_model_or_name
            if self.verbose: print(f"Diariser updated to provided instance: {self.diariser}")
        elif isinstance(dia_model_or_name, str):
            self.diariser = Diariser.load_model(dia_model_or_name, **kwargs)
            if self.verbose: print(f"Diariser updated to loaded model: {self.diariser.model_name if hasattr(self.diariser, 'model_name') else self.diariser}")
        elif dia_model_or_name is None:
            self.diariser = Diariser.load_model(**kwargs) # Load default
            if self.verbose: print(f"Diariser reset to default model: {self.diariser.model_name if hasattr(self.diariser, 'model_name') else self.diariser}")
        else:
            warnings.warn(
                f"Invalid type for dia_model_or_name: {type(dia_model_or_name)}. "
                f"Expected model name (str), Diariser instance, or None. "
                f"Diariser not updated, fallback to old '{_old_diariser_info}'.", RuntimeWarning)
        return None

    # remove_audio_file and get_audio_file can remain largely the same,
    # just ensure AudioProcessor details are consistent.
    # ... (remove_audio_file and get_audio_file methods as in your original, ensure they are robust) ...

    @staticmethod
    def remove_audio_file(audio_file_path: str, shred: bool = False) -> None:
        if not isinstance(audio_file_path, str):
            warnings.warn(f"remove_audio_file expects a path string, got {type(audio_file_path)}. Skipping removal.")
            return
            
        if not os.path.exists(audio_file_path):
            # It might have been a temporary AudioProcessor object, not a file path that still exists
            warnings.warn(f"Audio file {audio_file_path} does not exist or was not a persistent file. Skipping removal.")
            return

        try:
            if shred:
                warnings.warn("Shredding audiofile can take a long time.", RuntimeWarning)
                # Basic shred command for Linux/macOS. May need adjustment for Windows or if `shred` not available.
                # Ensure the file path is secure against command injection if it comes from untrusted input.
                # For simplicity, direct command construction. Be cautious with paths from external sources.
                run(['shred', '-zvu', '-n', '1', audio_file_path], check=True) # Reduced shred passes to 1 for speed
                print(f"Audio file {audio_file_path} shredded and removed.")
            else:
                os.remove(audio_file_path)
                print(f"Audio file {audio_file_path} removed.")
        except FileNotFoundError:
             warnings.warn(f"Audio file {audio_file_path} was not found for removal (perhaps already deleted).")
        except Exception as e:
            warnings.warn(f"Error removing/shredding audio file {audio_file_path}: {e}")


    @staticmethod
    def get_audio_file(audio_source: Union[str, torch.Tensor, ndarray]) -> AudioProcessor:
        """
        Gets an audio file as an AudioProcessor object.
        Assumes if Tensor or ndarray is passed, it's (waveform, sample_rate).
        For a single Tensor input (waveform), assumes a default sample rate or requires it.
        This implementation assumes AudioProcessor can handle these inputs.
        """
        if isinstance(audio_source, str):
            # Assuming AudioProcessor.from_file handles path existence and loading
            return AudioProcessor.from_file(audio_source)
        elif isinstance(audio_source, torch.Tensor):
            # This part of your original logic was ambiguous:
            # audio_file = AudioProcessor(audio_file[0], audio_file[1])
            # If audio_source is just the waveform tensor, what is audio_file[1] (sample_rate)?
            # Assuming if it's a Tensor, it's JUST the waveform, and SR is default (e.g. 16kHz)
            # or must be passed separately.
            # For now, let's assume AudioProcessor can take a raw waveform and assumes/is given SR.
            # This needs to align with your AudioProcessor's constructor.
            # If AudioProcessor expects (waveform_tensor, sr), the caller must provide that structure.
            # Let's assume audio_source is just the waveform here.
            # A more robust AudioProcessor might have from_tensor(tensor, sample_rate)
            if audio_source.ndim > 1 and audio_source.shape[0] == 1 : # (1, num_samples) -> (num_samples)
                audio_source = audio_source.squeeze(0)
            return AudioProcessor(waveform=audio_source, sample_rate=16000) # Assuming 16kHz if SR not given with tensor
        elif isinstance(audio_source, ndarray):
            # Similar ambiguity as Tensor.
            # audio_file = AudioProcessor(torch.Tensor(audio_file[0]), audio_file[1])
            # Assuming ndarray is just the waveform.
            return AudioProcessor(waveform=torch.from_numpy(audio_source.astype(np.float32)), sample_rate=16000) # Assuming 16kHz

        # If audio_source is already an AudioProcessor instance (e.g. from a previous call)
        if isinstance(audio_source, AudioProcessor):
            return audio_source
            
        raise TypeError(f"Unsupported audio_source type: {type(audio_source)}. "
                        "Expected path (str), waveform (Tensor/ndarray), or AudioProcessor.")

    def __repr__(self):
        transcriber_info = str(self.transcriber) if self.transcriber else "None"
        diariser_info = str(self.diariser) if self.diariser else "None"
        return f"Scraibe(transcriber={transcriber_info}, diariser={diariser_info})"