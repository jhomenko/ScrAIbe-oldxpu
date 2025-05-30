"""
Diarisation Class
------------------

This class serves as the heart of the speaker diarization system, responsible for identifying
and segmenting individual speakers from a given audio file. It leverages a pretrained model
from pyannote.audio, providing an accessible interface for audio processing tasks such as
speaker separation, and timestamping.

By encapsulating the complexities of the underlying model, it allows for straightforward
integration into various applications, ranging from transcription services to voice assistants.

Available Classes:
- Diariser: Main class for performing speaker diarization. 
            Includes methods for loading models, processing audio files,
            and formatting the diarization output.

Constants:
- TOKEN_PATH (str): Path to the Pyannote token.
- PYANNOTE_DEFAULT_PATH (str): Default path to Pyannote models.
- PYANNOTE_DEFAULT_CONFIG (str): Default configuration for Pyannote models.

Usage:
    from .diarisation import Diariser

    model = Diariser.load_model(model="path/to/model/config.yaml")
    diarisation_output = model.diarization("path/to/audiofile.wav")
"""

import warnings
import os
import yaml
from pathlib import Path
from typing import TypeVar, Union

from pyannote.audio import Pipeline
from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization
from torch import Tensor
from torch import device as torch_device

from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError

from .misc import PYANNOTE_DEFAULT_PATH, PYANNOTE_DEFAULT_CONFIG, SCRAIBE_TORCH_DEVICE
Annotation = TypeVar('Annotation')

TOKEN_PATH = os.path.join(os.path.dirname(
    os.path.realpath(__file__)), '.pyannotetoken')


class Diariser:
    """
    Handles the diarization process of an audio file using a pretrained model
    from pyannote.audio. Diarization is the task of determining "who spoke when."

    Args:
        model: The pretrained model to use for diarization.
    """

    def __init__(self, model) -> None:

        self.model = model

    def diarization(self, audiofile: Union[str, Tensor, dict],
                    *args, **kwargs) -> Annotation:
        """
        Perform speaker diarization on the provided audio file, 
        effectively separating different speakers
        and providing a timestamp for each segment.

        Args:
            audiofile: The path to the audio file or a torch.Tensor
                        containing the audio data.
            args: Additional arguments for the diarization model.
            kwargs: Additional keyword arguments for the diarization model.

        Returns:
            dict: A dictionary containing speaker names,
                    segments, and other information related
                    to the diarization process.
        """
        kwargs = self._get_diarisation_kwargs(**kwargs)

        diarization = self.model(audiofile, *args, **kwargs)

        out = self.format_diarization_output(diarization)

        return out

    @staticmethod
    def format_diarization_output(dia: Annotation) -> dict:
        """
        Formats the raw diarization output into a more usable structure for this project.

        Args:
            dia: Raw diarization output.

        Returns:
            dict: A structured representation of the diarization, with speaker names
                  as keys and a list of tuples representing segments as values.
        """

        dia_list = list(dia.itertracks(yield_label=True))
        diarization_output = {"speakers": [], "segments": []}

        normalized_output = []
        index_start_speaker = 0
        index_end_speaker = 0
        current_speaker = str()

        ###
        # Sometimes two consecutive speakers are the same
        # This loop removes these duplicates
        ###

        if len(dia_list) == 1:
            normalized_output.append([0, 0, dia_list[0][2]])
        else:

            for i, (_, _, speaker) in enumerate(dia_list):

                if i == 0:
                    current_speaker = speaker

                if speaker != current_speaker:

                    index_end_speaker = i - 1

                    normalized_output.append([index_start_speaker,
                                              index_end_speaker,
                                              current_speaker])

                    index_start_speaker = i
                    current_speaker = speaker

                if i == len(dia_list) - 1:

                    index_end_speaker = i

                    normalized_output.append([index_start_speaker,
                                              index_end_speaker,
                                              current_speaker])

        for outp in normalized_output:
            start = dia_list[outp[0]][0].start
            end = dia_list[outp[1]][0].end

            diarization_output["segments"].append([start, end])
            diarization_output["speakers"].append(outp[2])
        return diarization_output

    @staticmethod
    def _get_token():
        """
        Retrieves the Huggingface token from a local file. This token is required
        for accessing certain online resources.

        Raises:
            ValueError: If the token is not found.

        Returns:
            str: The Huggingface token.
        """

        if os.path.exists(TOKEN_PATH):
            with open(TOKEN_PATH, 'r', encoding="utf-8") as file:
                token = file.read()
        else:
            raise ValueError('No token found.'
                             'Please create a token at https://huggingface.co/settings/token'
                             f'and save it in a file called {TOKEN_PATH}')
        return token

    @staticmethod
    def _save_token(token):
        """
        Saves the provided Huggingface token to a local file. This facilitates future
        access to online resources without needing to repeatedly authenticate.

        Args:
            token: The Huggingface token to save.
        """
        with open(TOKEN_PATH, 'w', encoding="utf-8") as file:
            file.write(token)

    @classmethod
    def load_model(cls,
                   model: Optional[str] = PYANNOTE_DEFAULT_CONFIG, # Allow model to be explicitly None
                   use_auth_token: str = None,
                   cache_token: bool = False,
                   cache_dir: Union[Path, str] = PYANNOTE_DEFAULT_PATH,
                   hparams_file: Union[str, Path] = None,
                   device: str = SCRAIBE_TORCH_DEVICE,
                   ) -> 'Diariser': # Type hint for returning an instance of the class
        """
        Loads a pretrained model from pyannote.audio, 
        either from a local cache or some online repository.

        Args:
            model: Path to a local config.yaml, a Hugging Face model ID (e.g., 'pyannote/speaker-diarization-3.1'),
                   a special tuple for fallback, or None to use PYANNOTE_DEFAULT_CONFIG.
            use_auth_token: Optional HUGGINGFACE_TOKEN for authenticated access. (Note: docstring said 'token')
            cache_token: Whether to cache the token locally for future use.
            cache_dir: Directory for caching models.
            hparams_file: Path to a YAML file containing hyperparameters.
            device: Device to load the model on.
        
        Returns:
            Diariser: An instance of the Diariser class, encapsulating the loaded pyannote.audio Pipeline.
        """
        
        # If 'model' is explicitly passed as None (e.g., from Scraibe if no diarization model is specified),
        # then use the default configuration path/ID defined in this class.
        current_model_identifier = model
        if current_model_identifier is None:
            current_model_identifier = PYANNOTE_DEFAULT_CONFIG
            if current_model_identifier is None: # Should not happen if PYANNOTE_DEFAULT_CONFIG is properly set
                raise ValueError("Diariser model identifier is None and PYANNOTE_DEFAULT_CONFIG is also not set.")

        # --- Your original logic for handling local paths and tuples ---
        # This section might modify 'current_model_identifier' or 'use_auth_token'
        
        if isinstance(current_model_identifier, str) and os.path.exists(current_model_identifier):
            # This block handles the case where 'current_model_identifier' is a path to a local config.yaml
            # It will attempt to find the associated .bin file and may modify the config.yaml in place.
            # 'current_model_identifier' will remain the path to the (potentially modified) config.yaml.
            
            # Your existing local path handling code:
            with open(current_model_identifier, 'r') as file:
                config = yaml.safe_load(file)

            path_to_model_bin = config['pipeline']['params']['segmentation']

            if not os.path.exists(path_to_model_bin):
                warnings.warn(f"Model binary not found at {path_to_model_bin} (specified in {current_model_identifier}). "
                              "Trying to find it nearby the config file.")

                pwd = os.path.dirname(current_model_identifier) # Get directory of the config file
                potential_path_to_model_bin = os.path.join(pwd, "pytorch_model.bin")

                if not os.path.exists(potential_path_to_model_bin):
                    warnings.warn(f"Model binary also not found at {potential_path_to_model_bin}. "
                                  "Trying to find any .bin file in the same directory.")
                    bin_files = [f for f in os.listdir(pwd) if f.endswith(".bin")]
                    if len(bin_files) == 1:
                        potential_path_to_model_bin = os.path.join(pwd, bin_files[0])
                        # Fall through to update config with potential_path_to_model_bin
                    elif len(bin_files) > 1:
                        warnings.warn(f"Found more than one .bin file in {pwd}. Cannot automatically select. "
                                      "Please ensure the 'segmentation' path in your config.yaml is correct.")
                        # Proceed with original path_to_model_bin from config, Pipeline.from_pretrained might handle or fail.
                        potential_path_to_model_bin = None # Flag that we didn't find a unique alternative
                    else:
                        warnings.warn(f"Found no .bin files in {pwd}. "
                                      "Model loading will rely on the original 'segmentation' path in config or fail.")
                        potential_path_to_model_bin = None # Flag that we didn't find an alternative
                
                if potential_path_to_model_bin and os.path.exists(potential_path_to_model_bin):
                     warnings.warn(
                        f"Found model binary at {potential_path_to_model_bin}. Overwriting 'segmentation' path in loaded config for {current_model_identifier}.")
                     config['pipeline']['params']['segmentation'] = str(Path(potential_path_to_model_bin).resolve()) # Use absolute path
                     try:
                        with open(current_model_identifier, 'w') as file: # Attempt to update the config file
                            yaml.dump(config, file)
                        warnings.warn(f"Config file {current_model_identifier} updated with new model binary path.")
                     except Exception as e:
                        warnings.warn(f"Could not write updated config to {current_model_identifier}: {e}. Using in-memory modified config.")
            # 'current_model_identifier' (the path to config.yaml) is used by Pipeline.from_pretrained
            # which will read the (potentially modified in-memory or on-disk) 'segmentation' path.

        elif isinstance(current_model_identifier, tuple):
            # Your existing tuple handling code (for trying two HF model IDs)
            # This block should set 'current_model_identifier' to the chosen string ID
            # and potentially update 'use_auth_token'.
            try:
                primary_model_id = current_model_identifier[0]
                HfApi().model_info(primary_model_id) # Check existence
                current_model_identifier = primary_model_id
                # use_auth_token = None # If primary is public, token might not be needed for it
            except RepositoryNotFoundError:
                print(f"Primary model '{current_model_identifier[0]}' not found on Huggingface. Trying fallback '{current_model_identifier[1]}'.")
                fallback_model_id = current_model_identifier[1]
                try:
                    HfApi().model_info(fallback_model_id) # Check existence of fallback
                    current_model_identifier = fallback_model_id
                    # Token logic for fallback from your original code:
                    if use_auth_token is None: # If no token was provided initially for the fallback
                        try:
                            use_auth_token = cls._get_token() # Try to get a saved token
                        except ValueError: # No token found/saved
                            warnings.warn(f"No Hugging Face token provided or found for fallback model {fallback_model_id}. Public access will be attempted.")
                            use_auth_token = None # Ensure it's None for Pipeline.from_pretrained
                    # If use_auth_token was already provided, it will be used for the fallback.
                    if cache_token and use_auth_token is not None: # Cache if specified and token exists
                        cls._save_token(use_auth_token)
                except RepositoryNotFoundError:
                    raise FileNotFoundError(f"Neither primary '{current_model_identifier[0]}' nor fallback '{fallback_model_id}' model found on Huggingface.")
            except Exception as e:
                 raise ValueError(f"Error validating model tuple {current_model_identifier} with Hugging Face Hub: {e}")
        
        # If current_model_identifier is a string (original, from default, or from tuple logic)
        # and not a local existing path handled above, it's assumed to be a Hugging Face ID
        # or other identifier that Pipeline.from_pretrained can handle.
        # The original 'else: raise FileNotFoundError' is removed here to allow HF IDs to pass through.
        # Pipeline.from_pretrained will raise its own error if the ID is invalid.
        if not isinstance(current_model_identifier, str):
             raise TypeError(f"Processed model identifier is not a string: {current_model_identifier} (type {type(current_model_identifier)}). Expected path or Hugging Face ID.")


        # Call Pipeline.from_pretrained with the resolved current_model_identifier
        pipeline_instance = Pipeline.from_pretrained(
            current_model_identifier,
            use_auth_token=use_auth_token, # This might have been updated by the tuple logic
            cache_dir=cache_dir,
            hparams_file=hparams_file,
        )
        
        if pipeline_instance is None: # Should typically raise error in from_pretrained, but as a safeguard
            raise ValueError(f"Unable to load model for '{current_model_identifier}'. "
                             "Please check your model identifier, token, or local path integrity.")

        pipeline_instance = pipeline_instance.to(torch_device(device))
        return cls(pipeline_instance)    

    @staticmethod
    def _get_diarisation_kwargs(**kwargs) -> dict:
        """
        Validates and extracts the keyword arguments for the pyannote diarization model.

        Ensures that the provided keyword arguments match the expected parameters,
        filtering out any invalid or unnecessary arguments.

        Returns:
            dict: A dictionary containing the validated keyword arguments.
        """
        _possible_kwargs = SpeakerDiarization.apply.__code__.co_varnames

        diarisation_kwargs = {k: v for k,
                              v in kwargs.items() if k in _possible_kwargs}

        return diarisation_kwargs

    def __repr__(self):
        return f"Diarisation(model={self.model})"
