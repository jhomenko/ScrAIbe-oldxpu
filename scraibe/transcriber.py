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

Usage:
    >>> from your_package import Transcriber
    >>> transcriber = Transcriber.load_model(model="medium")
    >>> transcript = transcriber.transcribe(audio="path/to/audio.wav")
    >>> transcriber.save_transcript(transcript, "path/to/save.txt")
"""

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
from transformers.utils import ModelOutput
import transformers
from packaging import version

import torch.xpu.amp
import torch
import warnings
import intel_extension_for_pytorch as ipex
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
        if isinstance(audio, Tensor):
            audio = audio.to(self.model.device).to(torch.float16) # Use self.model.device for consistency


        kwargs = self._get_whisper_kwargs(**kwargs)

        if not kwargs.get("verbose"):
            kwargs["verbose"] = None

        with torch.xpu.amp.autocast(enabled=True, dtype=torch.float16):
            result = self.whisper_generate(audio, *args, **kwargs)

        # Assuming whisper_generate returns a dictionary with a 'sequences' key containing the token IDs
        # We need to convert these token IDs back to text. This might require using the model's tokenizer.
        # The exact method depends on the structure of `result` from whisper_generate and the model's tokenizer.
        # For now, let's assume `result` is a dictionary and the output we want is in 'sequences'
        # and we can use the model's tokenizer's decode method.
        # If whisper_generate returns something else, this part will need adjustment.

        if isinstance(result, dict) and 'sequences' in result:

            # Assuming self.model has a tokenizer attribute
            if hasattr(self.model, 'tokenizer'):
                # Decode the token IDs. Assuming sequences is a tensor of token IDs.
                # This might need adjustment based on the shape and content of result['sequences']
                text = self.model.tokenizer.batch_decode(result['sequences'], skip_special_tokens=True)
            # If it's a list of sequences, join them or handle appropriately
            if isinstance(text, list):
                text = " ".join(text) # Example: join multiple sequences
            else:
 warnings.warn("Expected 'sequences' in whisper_generate result, but the structure is different. Returning raw result.")
 text = str(result) # Return string representation as a fallback
 else:
 warnings.warn("Model does not have a tokenizer. Cannot decode sequences. Returning raw result.")
 text = str(result) # Return string representation as a fallback
 else:
 warnings.warn("Expected a dictionary result from whisper_generate. Returning raw result.")
            text = str(result) # Return string representation as a fallback

        return text

    # Helper functions and whisper_generate copied from utils.py
 def _extract_past_from_model_output(
 self, outputs: ModelOutput, standardize_cache_format: bool = False
 ):
 past_key_values = None
 cache_name = "past_key_values"
 # To use torch.jit.trace, the output is no longer a Dict. outputs[1] corresponds to "past_key_values"
 if hasattr(self, "trace_graph"):
 past_key_values = outputs[1]
 if "past_key_values" in outputs:
 past_key_values = outputs.past_key_values
 elif "mems" in outputs:
 past_key_values = outputs.mems
 elif "past_buckets_states" in outputs:
 past_key_values = outputs.past_buckets_states
 elif "cache_params" in outputs:
 past_key_values = outputs.cache_params
 cache_name = "cache_params"

 # Bloom fix: standardizes the cache format when requested
 if standardize_cache_format and hasattr(self, "_convert_to_standard_cache"):
 batch_size = outputs.logits.shape[0]
 past_key_values = self._convert_to_standard_cache(
 past_key_values, batch_size=batch_size
 )
 if version.parse(transformers.__version__) < version.parse("4.42.0"):
 return past_key_values
 return cache_name, past_key_values
 
 def _update_model_kwargs_for_generation(
 self,
 outputs: ModelOutput,
 model_kwargs: Dict[str, Any],
 is_encoder_decoder: bool = False,
 standardize_cache_format: bool = False,
 num_new_tokens: int = 1,
 ) -> Dict[str, Any]:

 cross_attention_mask_prev = model_kwargs.get("cross_attention_mask", None)
 # add cross-attn mask for new token
 if cross_attention_mask_prev is not None:
 model_kwargs["cross_attention_mask"] = torch.cat(
 [cross_attention_mask_prev, cross_attention_mask_prev[:, -1:, ...]], dim=1
 )

 try:
 # update past_key_values keeping its naming used in model code
 cache_name, cache = self._extract_past_from_model_output(
 outputs, standardize_cache_format=standardize_cache_format
 )
 model_kwargs[cache_name] = cache
 except ValueError:
 # update past_key_values
 model_kwargs["past_key_values"] = self._extract_past_from_model_output(
 outputs, standardize_cache_format=standardize_cache_format
 )
 if getattr(outputs, "state", None) is not None:
 model_kwargs["state"] = outputs.state

 # update token_type_ids with last value
 if "token_type_ids" in model_kwargs:
 token_type_ids = model_kwargs["token_type_ids"]
 model_kwargs["token_type_ids"] = torch.cat(
 [token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1
 )

 if not is_encoder_decoder:
 # update attention mask
 if "attention_mask" in model_kwargs:
 attention_mask = model_kwargs["attention_mask"]
 model_kwargs["attention_mask"] = torch.cat(
 [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))],
 dim=-1,
 )
 else:
 # update decoder attention mask
 if "decoder_attention_mask" in model_kwargs:
 decoder_attention_mask = model_kwargs["decoder_attention_mask"]
 model_kwargs["decoder_attention_mask"] = torch.cat(
 [
 decoder_attention_mask,
 decoder_attention_mask.new_ones(
 (decoder_attention_mask.shape[0], 1)
 ),
 ],
 dim=-1,
 )

 if (
 model_kwargs.get("use_cache", True)
 and "cache_position" in model_kwargs
 and model_kwargs["cache_position"] is not None
 ):
 model_kwargs["cache_position"] = (
 model_kwargs["cache_position"][-1:] + num_new_tokens
 )

 return model_kwargs
 
 def _get_attr_from_logit_processors(
 logits_processor, logit_processor_class, attribute_name
 ):
 logit_processor = next(
 (cls for cls in logits_processor if isinstance(cls, logit_processor_class)),
 None,
 )
 if logit_processor:
 return getattr(logit_processor, attribute_name, None)
 return None
 
 def _pad_to_max_length(
 current_segments,
 pad_token_id,
 device,
 padding_side="right",
 padding="longest",
 bos_token_tensor=None,
 cut_off_length=None,
 return_token_timestamps=False,
 force_unique_generate_call=False,
 ):
 max_total_length = 0
 sequences = []
 token_timestamps_list = []

 if padding_side not in ["right", "left"]:
 raise ValueError(
 f"`padding_side` must be either 'right' or 'left', not {padding_side}"
 )

 if padding not in ["longest", "max_length"]:
 raise ValueError(
 f"`padding` must be either 'longest' or 'max_length', not {padding}"
 )
 elif padding == "max_length" and cut_off_length is None:
 raise ValueError(
 "`cut_off_length` must be specified when `padding='max_length'`"
 )

 if force_unique_generate_call:
 sequences_list = []
 timestamps_list = []
 for segments in current_segments:
 result = segments[0]["result"]
 sequences_list.append(
 result if isinstance(result, torch.Tensor) else result["sequences"]
 )
 if return_token_timestamps:
 timestamps_list.append(result["token_timestamps"])

 sequences = torch.stack(sequences_list, dim=0)
 if return_token_timestamps:
 token_timestamps = torch.stack(timestamps_list, dim=0)
 return sequences, token_timestamps
 return sequences

 for current_segment_list in current_segments:
 if (
 current_segment_list is not None
 and len([d["tokens"] for d in current_segment_list]) > 0
 ):
 sequence = torch.cat([d["tokens"] for d in current_segment_list], dim=-1)
 if return_token_timestamps:
 token_timestamps = torch.cat(
 [
 d["result"]["token_timestamps"][d["idxs"][0] : d["idxs"][1]]
 for d in current_segment_list
 ],
 dim=-1,
 )

 if cut_off_length is not None:
 sequence = sequence[-cut_off_length:]
 if return_token_timestamps:
 token_timestamps = token_timestamps[-cut_off_length:]

 if bos_token_tensor is not None:
 sequence = torch.cat([bos_token_tensor, sequence])
 if return_token_timestamps:
 token_timestamps = torch.cat(
 [
 torch.ones_like(bos_token_tensor, device=device) * 0.0,
 token_timestamps,
 ]
 )
 sequences.append(sequence)
 if return_token_timestamps:
 token_timestamps_list.append(token_timestamps)
 max_total_length = max(max_total_length, len(sequences[-1]))
 elif bos_token_tensor is not None:
 sequences.append(bos_token_tensor)
 if return_token_timestamps:
 token_timestamps_list.append(
 torch.ones_like(bos_token_tensor, device=device) * 0.0
 )
 else:
 sequences.append(torch.tensor([], device=device))
 if return_token_timestamps:
 token_timestamps_list.append(torch.tensor([], device=device))

 max_total_length = (
 cut_off_length + 1 if padding == "max_length" else max_total_length
 )
 for i in range(len(current_segments)):
 pad_length = max_total_length - len(sequences[i])
 pad = (0, pad_length) if padding_side == "right" else (pad_length, 0)

 sequences[i] = F.pad(sequences[i], pad=pad, value=pad_token_id)
 if return_token_timestamps:
 token_timestamps_list[i] = F.pad(
 token_timestamps_list[i],
 pad=pad,
 value=(
 token_timestamps_list[i][-1]
 if len(token_timestamps_list[i]) > 0
 else 0.0
 ),
 )

 sequences = torch.stack(sequences, dim=0)

 if return_token_timestamps:
 token_timestamps = torch.stack(token_timestamps_list, dim=0)
 return sequences, token_timestamps
 else:
 return sequences
 
 def whisper_generate(
 self,
 input_features: Optional[torch.Tensor] = None,
 generation_config=None,
 logits_processor=None,
 stopping_criteria=None,
 prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
 synced_gpus: bool = False,
 return_timestamps: Optional[bool] = None,
 task: Optional[str] = None,
 language: Optional[Union[str, List[str]]] = None,
 is_multilingual: Optional[bool] = None,
 prompt_ids: Optional[torch.Tensor] = None,
 prompt_condition_type: Optional[str] = None, # first-segment, all-segments
 condition_on_prev_tokens: Optional[bool] = None,
 temperature: Optional[Union[float, Tuple[float, ...]]] = None,
 compression_ratio_threshold: Optional[float] = None,
 logprob_threshold: Optional[float] = None,
 no_speech_threshold: Optional[float] = None,
 num_segment_frames: Optional[int] = None,
 attention_mask: Optional[torch.Tensor] = None,
 time_precision: float = 0.02,
 time_precision_features: float = 0.01,
 return_token_timestamps: Optional[bool] = None,
 return_segments: bool = False,
 return_dict_in_generate: Optional[bool] = None,
 force_unique_generate_call: Optional[bool] = None,
 **kwargs,
 ):
 # 0. deprecate old inputs
 if "inputs" in kwargs:
 input_features = kwargs.pop("inputs")
 warnings.warn(
 "The input name `inputs` is deprecated. Please make sure to use `input_features` instead.",
 FutureWarning,
 )

 # 1. prepare generation config
 generation_config, kwargs = self._prepare_generation_config(
 generation_config, **kwargs
 )

 # 2. set global generate variables
 input_stride = (
 self.model.encoder.conv1.stride[0] * self.model.encoder.conv2.stride[0]
 )
 num_segment_frames = input_stride * self.config.max_source_positions
 batch_size, total_input_frames = self._retrieve_total_input_frames(
 input_features=input_features, input_stride=input_stride, kwargs=kwargs
 )
 is_shortform = total_input_frames <= num_segment_frames

 # 3. Make sure generation config is correctly set
 # Make sure the generation config is correctly set depending on whether timestamps are to be returned or not
 return_dict_in_generate = self._set_return_outputs(
 return_dict_in_generate=return_dict_in_generate,
 return_token_timestamps=return_token_timestamps,
 logprob_threshold=logprob_threshold,
 generation_config=generation_config,
 )
 timestamp_begin = self._set_return_timestamps(
 return_timestamps=return_timestamps,
 is_shortform=is_shortform,
 generation_config=generation_config,
 )
 self._set_language_and_task(
 language=language,
 task=task,
 is_multilingual=is_multilingual,
 generation_config=generation_config,
 )
 self._set_num_frames(
 return_token_timestamps=return_token_timestamps,
 generation_config=generation_config,
 kwargs=kwargs,
 )
 self._set_thresholds_and_condition(
 generation_config=generation_config,
 logprob_threshold=logprob_threshold,
 compression_ratio_threshold=compression_ratio_threshold,
 no_speech_threshold=no_speech_threshold,
 condition_on_prev_tokens=condition_on_prev_tokens,
 )
 self._set_prompt_condition_type(
 generation_config=generation_config,
 prompt_condition_type=prompt_condition_type,
 )

 # pass self.config for backward compatibility
 init_tokens = self._retrieve_init_tokens(
 input_features,
 batch_size=batch_size,
 generation_config=generation_config,
 config=self.config,
 num_segment_frames=num_segment_frames,
 kwargs=kwargs,
 )
 # passing `decoder_input_ids` is deprecated - the only exception is for assisted generation
 # where the input ids are handled explicitly by the generate method
 self._check_decoder_input_ids(kwargs=kwargs)

 # 3. Retrieve logits processors
 device = (
 kwargs["encoder_outputs"][0].device
 if "encoder_outputs" in kwargs
 else input_features.device
 )
 begin_index = init_tokens.shape[1]
 num_beams = kwargs.get(
 "num_beams",
 (
 generation_config.num_beams
 if hasattr(generation_config, "num_beams")
 and generation_config.num_beams is not None
 else 1
 ),
 )
 if "assistant_model" in kwargs:
 # speculative decoding: the model should be able to return eos token
 generation_config.begin_suppress_tokens = None
 logits_processor = self._retrieve_logit_processors(
 generation_config=generation_config,
 logits_processor=logits_processor,
 begin_index=begin_index, # begin index is index of first generated decoder token
 num_beams=num_beams,
 device=device,
 )

 # 4 Set and retrieve global generation variables
 self._set_condition_on_prev_tokens(
 condition_on_prev_tokens=condition_on_prev_tokens,
 generation_config=generation_config,
 )

 temperatures = (
 [temperature] if not isinstance(temperature, (list, tuple)) else temperature
 )
 temperature = temperatures[0]

 max_frames, seek = self._retrieve_max_frames_and_seek(
 batch_size=batch_size,
 attention_mask=attention_mask,
 total_input_frames=total_input_frames,
 is_shortform=is_shortform,
 )

 # 5 Prepare running variables, list for generation
 num_return_sequences = generation_config.num_return_sequences
 (
 batch_idx_map,
 cur_bsz,
 input_features,
 seek,
 max_frames,
 init_tokens,
 do_condition_on_prev_tokens,
 ) = self._expand_variables_for_generation(
 input_features=input_features,
 seek=seek,
 max_frames=max_frames,
 init_tokens=init_tokens,
 batch_size=batch_size,
 condition_on_prev_tokens=condition_on_prev_tokens,
 generation_config=generation_config,
 )

 current_segments = self._prepare_segments(
 prompt_ids=prompt_ids,
 batch_size=cur_bsz,
 generation_config=generation_config,
 )
 # 5bis speculative decoding: ensure the assistant model does only one call to generate
 # and therefore returns decoder input token ids and eos token id
 # we set a flag in the generation config to force the model to make only one call to generate
 # and return the decoder input token ids and eos token id
 if "assistant_model" in kwargs:
 assistant_model = kwargs["assistant_model"]
 assistant_model.generation_config.force_unique_generate_call = True

 if force_unique_generate_call is None:
 if hasattr(generation_config, "force_unique_generate_call"):
 force_unique_generate_call = generation_config.force_unique_generate_call
 elif hasattr(self.generation_config, "force_unique_generate_call"):
 force_unique_generate_call = (
 self.generation_config.force_unique_generate_call
 )
 else:
 force_unique_generate_call = False
 # 6 Transcribe audio until we reach the end of all input audios
 while (seek < max_frames).any():
 # 6.1 NOTE: When in longform transcription mode and batch size > 1 we need to dynamically
 # reduce the batch size during the loop in case one audio finished earlier than another one.
 # Thus, we need to keep a table of "previous-index-2-current-index" in order
 # to know which original audio is being decoded
 # Set updated index map, duration of previously decoded chunks and number of max frames of current decoding chunk
 input_features, cur_bsz, batch_idx_map = self._maybe_reduce_batch(
 input_features=input_features,
 seek=seek,
 max_frames=max_frames,
 cur_bsz=cur_bsz,
 batch_idx_map=batch_idx_map,
 )
 time_offset = (
 seek.to(torch.float32 if device.type == "mps" else torch.float64)
 * time_precision
 / input_stride
 )
 seek_num_frames = (max_frames - seek).clamp(max=num_segment_frames)

 # 6.2 cut out next 30s segment from input features
 segment_input = self._get_input_segment(
 input_features=input_features,
 seek=seek,
 seek_num_frames=seek_num_frames,
 num_segment_frames=num_segment_frames,
 cur_bsz=cur_bsz,
 batch_idx_map=batch_idx_map,
 )

 # 6.3 prepare decoder input ids
 suppress_tokens = self._get_attr_from_logit_processors(
 logits_processor,
 transformers.generation.logits_process.SuppressTokensLogitsProcessor,
 "suppress_tokens",
 )
 extra_kwargs = {}
 if version.parse(transformers.__version__) >= version.parse("4.47.0"):
 extra_kwargs["timestamp_begin"] = timestamp_begin

 decoder_input_ids, kwargs = self._prepare_decoder_input_ids(
 cur_bsz=cur_bsz,
 init_tokens=init_tokens,
 current_segments=current_segments,
 batch_idx_map=batch_idx_map,
 do_condition_on_prev_tokens=do_condition_on_prev_tokens,
 prompt_ids=prompt_ids,
 generation_config=generation_config,
 config=self.config,
 device=init_tokens.device,
 suppress_tokens=suppress_tokens,
 **extra_kwargs,
 kwargs=kwargs,
 )

 # 6.4 set max new tokens or max length
 self._set_max_new_tokens_and_length(
 config=self.config,
 decoder_input_ids=decoder_input_ids,
 generation_config=generation_config,
 )

 # 6.5 Set current `begin_index` for all logit processors
 if logits_processor is not None:
 for proc in logits_processor:
 if hasattr(proc, "set_begin_index"):
 proc.set_begin_index(decoder_input_ids.shape[-1])

 # 6.6 Run generate with fallback
 (
 seek_sequences,
 seek_outputs,
 should_skip,
 do_condition_on_prev_tokens,
 model_output_type,
 ) = self.generate_with_fallback(
 segment_input=segment_input,
 decoder_input_ids=decoder_input_ids,
 cur_bsz=cur_bsz,
 batch_idx_map=batch_idx_map,
 seek=seek,
 num_segment_frames=num_segment_frames,
 max_frames=max_frames,
 temperatures=temperatures,
 generation_config=generation_config,
 logits_processor=logits_processor,
 stopping_criteria=stopping_criteria,
 prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
 synced_gpus=synced_gpus,
 return_token_timestamps=return_token_timestamps,
 do_condition_on_prev_tokens=do_condition_on_prev_tokens,
 is_shortform=is_shortform,
 batch_size=batch_size,
 attention_mask=attention_mask,
 kwargs=kwargs,
 )

 # 6.7 In every generated sequence, split by timestamp tokens and extract segments
 for i, seek_sequence in enumerate(seek_sequences):
 prev_i = batch_idx_map[i]

 if should_skip[i]:
 seek[prev_i] += seek_num_frames[prev_i]
 continue
 extra_kwargs = {}
 if version.parse(transformers.__version__) >= version.parse("4.48.0"):
 extra_kwargs["decoder_input_ids"] = decoder_input_ids
 if version.parse(transformers.__version__) >= version.parse("4.47.0"):
 extra_kwargs["time_precision_features"] = time_precision_features
 segments, segment_offset = self._retrieve_segment(
 seek_sequence=seek_sequence,
 seek_outputs=seek_outputs,
 time_offset=time_offset,
 timestamp_begin=timestamp_begin,
 seek_num_frames=seek_num_frames,
 time_precision=time_precision,
 input_stride=input_stride,
 prev_idx=prev_i,
 idx=i,
 return_token_timestamps=return_token_timestamps,
 **extra_kwargs,
 )
 seek[prev_i] += segment_offset
 current_segments[prev_i] += segments

 if force_unique_generate_call:
 break

 # 7. Once all segments are added to the list of all segments, called `current_segments`, we extract the predicted
 # output tokens from the list of dicts. If we use batch size > 1, we make sure to pad the output
 final_segments = (
 [x[1:] for x in current_segments]
 if (
 prompt_ids is not None
 and generation_config.prompt_condition_type == "first-segment"
 )
 else current_segments
 )

 # if return_dict_in_generate=True and we forced a unique call to generate or return_timestamps=False,\n
 # meaning we are sure only one call to generate has been made,\n
 # -> we can return a ModelOutput\n
 # otherwise, return_dict_in_generate is applied in the \'result\' of each segment in final_segments\n
 if (
 return_dict_in_generate
 and generation_config.return_dict_in_generate
 and (force_unique_generate_call or not return_timestamps)
 ):
 # only one call to generate_with_fallback, we can return a ModelOutput
 outputs = self._stack_split_outputs(
 seek_outputs, model_output_type, self.device, kwargs
 )
 if num_return_sequences > 1:
 if (
 hasattr(outputs, "encoder_attentions")
 and outputs.encoder_attentions is not None
 ):
 outputs.encoder_attentions = tuple(
 outputs.encoder_attentions[i][::num_return_sequences]
 for i in range(len(outputs.encoder_attentions))
 )
 if (
 hasattr(outputs, "encoder_hidden_states")
 and outputs.encoder_hidden_states is not None
 ):
 outputs.encoder_hidden_states = tuple(
 outputs.encoder_hidden_states[i][::num_return_sequences]
 for i in range(len(outputs.encoder_hidden_states))
 )
 return outputs

 padded_outputs = self._pad_to_max_length(
 current_segments=final_segments,
 pad_token_id=generation_config.pad_token_id,
 device=self.device,
 padding_side="right",
 return_token_timestamps=return_token_timestamps,
 force_unique_generate_call=force_unique_generate_call,
 )

 if return_dict_in_generate and generation_config.return_dict_in_generate:
 return_segments = True
 elif not return_segments and not return_token_timestamps:
 if hasattr(self.config, "token_latency") and self.config.token_latency:
 return (padded_outputs, seek_outputs[0])
 return padded_outputs

 if return_token_timestamps:
 sequences, token_timestamps = padded_outputs
 outputs = {
 "sequences": sequences,
 "token_timestamps": token_timestamps,
 }
 elif hasattr(self.config, "token_latency") and self.config.token_latency:
 outputs = (sequences, seek_outputs[0])
 else:
 sequences = padded_outputs
 outputs = {
 "sequences": sequences,
 }

 if return_segments:
 outputs["segments"] = final_segments

 return outputs

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

        _model = whisper_load_model(model, download_root=download_root, device=device, in_memory=in_memory)
        _model = ipex.optimize(_model.eval(), dtype=torch.float16)
        _model.to(torch.float16)

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

        if not isinstance(device, str):
            device = str(device)
            
        compute_type = kwargs.get('compute_type', 'float16')
        if device == 'cpu' and compute_type == 'float16':
            warnings.warn(f'Compute type {compute_type} not compatible with device {device}! Changing compute type to int8.')
            compute_type = 'int8'
        _model = FasterWhisperModel(model, download_root=download_root,
                                    device=device, compute_type=compute_type, 
                                    cpu_threads=SCRAIBE_NUM_THREADS)

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



def load_transcriber(model: str = "medium",
                     whisper_type: str = 'whisper',
                     download_root: str = WHISPER_DEFAULT_PATH,
                     device: Optional[Union[str, device]] = SCRAIBE_TORCH_DEVICE,
                     in_memory: bool = False,
                     *args, **kwargs
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
        kwargs: Additional keyword arguments only to avoid errors.

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
            model, download_root, device, *args, **kwargs)
        return _model
    else:
        raise ValueError(f'Model type not recognized, exptected "whisper" '
                         f'or "faster-whisper", got {whisper_type}.')
