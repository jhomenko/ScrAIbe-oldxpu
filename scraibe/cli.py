"""
Command-Line Interface (CLI) for the Scraibe class,
allowing for user interaction to transcribe and diarize audio files.
The function includes arguments for specifying the audio files, model paths,
output formats, and other options necessary for transcription.
"""
import os
import json
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE
from torch.cuda import is_available
from .autotranscript import Scraibe
from .misc import set_threads

def cli():
    """
    Command-Line Interface (CLI) for the Scraibe class, allowing for user interaction to transcribe
    and diarize audio files. The function includes arguments for specifying the audio files, model paths,
    output formats, and other options necessary for transcription.

    This function can be executed from the command line to perform transcription tasks, providing a
    user-friendly way to access the Scraibe class functionalities.
    """

    def str2bool(string):
        str2val = {"True": True, "False": False}
        if string in str2val:
            return str2val[string]
        else:
            raise ValueError(
                f"Expected one of {set(str2val.keys())}, got {string}")

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("-f", "--audio-files", nargs="+", type=str, default=None,
                        help="List of audio files to transcribe.")

    parser.add_argument("--whisper-type", type=str, default="whisper",
                        choices=["whisper", "faster-whisper"],
                        help="Type of Whisper model to use ('whisper' or 'faster-whisper').")

    parser.add_argument("--whisper-model-name", default="medium",
                        help="Name of the Whisper model to use.")

    parser.add_argument("--whisper-model-directory", type=str, default=None,
                        help="Path to save Whisper model files; defaults to system's default cache.")

    parser.add_argument("--diarization-directory", type=str, default=None,
                        help="Path to the diarization model directory or name of a preset model.")

    parser.add_argument("--hf-token", default=None, type=str,
                        help="HuggingFace token for private model download.")

    parser.add_argument("--inference-device",
                        default="cuda" if is_available() else "cpu",
                        help="Device to use for PyTorch inference (e.g., 'cpu', 'cuda', 'xpu').")

    parser.add_argument("--num-threads", type=int, default=None,
                        help="Number of threads used by torch for CPU inference; "
                             "overrides MKL_NUM_THREADS/OMP_NUM_THREADS.")

    parser.add_argument("--output-directory", "-o", type=str, default=".",
                        help="Directory to save the transcription outputs.")

    parser.add_argument("--output-format", "-of", type=str, default="txt",
                        choices=["txt", "json", "md", "html"], # Assuming Scraibe output object has .save(path, format=...)
                        help="Format of the output file; defaults to txt.")

    parser.add_argument("--verbose-output", type=str2bool, default=True,
                        help="Enable or disable verbose progress and debug messages for transcription methods.")

    parser.add_argument("--task", type=str, default='autotranscribe',
                        choices=["autotranscribe", "diarization",
                                 "autotranscribe+translate", "translate", 'transcribe'],
                        help="Choose to perform transcription, diarization, or translation. "
                             "If set to translate, the output will be translated to English.")

    parser.add_argument("--language", type=str, default=None,
                        choices=sorted(
                            LANGUAGES.keys()) + sorted([k.title() for k in TO_LANGUAGE_CODE.keys()]),
                        help="Language spoken in the audio. Specify None to perform language detection.")
    parser.add_argument("--num-speakers", type=int, default=None, # Default to None, let autotranscribe handle default if not set
                        help="Number of speakers in the audio (used by autotranscribe).")

    args = parser.parse_args()
    arg_dict = vars(args)

    # --- Prepare arguments for Scraibe class constructor ---
    class_kwargs = {}
    class_kwargs['whisper_model'] = arg_dict.pop("whisper_model_name")
    class_kwargs['whisper_type'] = arg_dict.pop("whisper_type")
    class_kwargs['target_device'] = arg_dict.pop("inference_device")

    if arg_dict.get("diarization_directory") is not None:
        class_kwargs['dia_model'] = arg_dict.pop("diarization_directory")
    else:
        arg_dict.pop("diarization_directory", None) # Remove from dict even if None

    if arg_dict.get("hf_token") is not None:
        class_kwargs['use_auth_token'] = arg_dict.pop("hf_token")
    else:
        arg_dict.pop("hf_token", None)

    if arg_dict.get("whisper_model_directory") is not None:
        class_kwargs["download_root"] = arg_dict.pop("whisper_model_directory")
    else:
        arg_dict.pop("whisper_model_directory", None)

    # --- General setup from remaining args ---
    out_folder = arg_dict.pop("output_directory")
    os.makedirs(out_folder, exist_ok=True)

    output_file_format = arg_dict.pop("output_format") # Renamed to avoid conflict
    task_to_perform = arg_dict.pop("task")
    
    set_threads(arg_dict.pop("num_threads"))

    # --- Initialize Scraibe ---
    # At this point, class_kwargs contains all necessary args for Scraibe.__init__
    # arg_dict contains args for Scraibe's methods:
    # 'audio_files', 'verbose_output', 'language', 'num_speakers'
    model = Scraibe(**class_kwargs)

    # --- Perform tasks ---
    audio_files_to_process = arg_dict.pop("audio_files")
    # Ensure out_folder and output_file_format are defined before this loop from arg_dict
    # out_folder = arg_dict.pop("output_directory")
    # os.makedirs(out_folder, exist_ok=True)
    # output_file_format = arg_dict.pop("output_format")


    if audio_files_to_process:
        language_arg = arg_dict.pop("language")
        verbose_arg = arg_dict.pop("verbose_output")
        num_speakers_arg = arg_dict.pop("num_speakers")

        for audio_file_path in audio_files_to_process:
            base_filename = os.path.splitext(os.path.basename(audio_file_path))[0]
            # Ensure 'out_folder' is defined from popped 'output_directory' before this loop
            output_path_base = os.path.join(out_folder, base_filename)

            print(f"\nProcessing: {audio_file_path}")

            if task_to_perform == "autotranscribe" or task_to_perform == "autotranscribe+translate":
                current_task_for_method = "translate" if task_to_perform == "autotranscribe+translate" else "transcribe"
                
                autotranscribe_params = {
                    'task': current_task_for_method,
                    'language': language_arg,
                    'verbose': verbose_arg,
                }
                if num_speakers_arg is not None:
                    autotranscribe_params['num_speakers'] = num_speakers_arg

                transcript_obj = model.autotranscribe(audio_file_path, **autotranscribe_params) # Renamed to transcript_obj
                
                save_path_with_extension = f"{output_path_base}.{output_file_format}"
                print(f'Attempting to save autotranscribe output to {save_path_with_extension}')

                saved_successfully = False
                if hasattr(transcript_obj, 'save'): # Ideal: Transcript object has a save method
                    try:
                        transcript_obj.save(save_path_with_extension) # Assumes .save() handles format by extension or internal logic
                        saved_successfully = True
                    except Exception as e:
                        print(f"Warning: transcript_obj.save() failed: {e}")
                elif hasattr(transcript_obj, 'save_to_file'): # Alternative save method signature
                     try:
                        transcript_obj.save_to_file(output_path_base, format=output_file_format)
                        saved_successfully = True
                     except Exception as e:
                        print(f"Warning: transcript_obj.save_to_file() failed: {e}")
                
                if not saved_successfully: # Fallback to manual formatting if no general save method worked
                    if output_file_format == 'txt':
                        try:
                            text_content = str(transcript_obj) # Relies on __str__ or a .to_text() method
                            if hasattr(transcript_obj, 'to_text'): text_content = transcript_obj.to_text()
                            with open(save_path_with_extension, "w", encoding='utf-8') as f:
                                f.write(text_content)
                            saved_successfully = True
                        except Exception as e:
                            print(f"Warning: Could not convert autotranscribe output to TXT or save: {e}")
                    elif output_file_format == 'json':
                        try:
                            if hasattr(transcript_obj, 'to_dict'):
                                dict_content = transcript_obj.to_dict()
                                with open(save_path_with_extension, "w", encoding='utf-8') as f:
                                    json.dump(dict_content, f, indent=2)
                                saved_successfully = True
                            else:
                                print(f"Warning: Transcript object does not have a to_dict() method for JSON export.")
                        except Exception as e:
                            print(f"Warning: Could not convert autotranscribe output to JSON or save: {e}")
                
                if not saved_successfully:
                    print(f"Warning: Failed to save output for {audio_file_path} in format {output_file_format}.")


            elif task_to_perform == "diarization":
                if verbose_arg:
                    print("Performing diarization...")
                
                diarization_result_dict = model.diarization(audio_file_path, num_speakers=num_speakers_arg)
                
                save_path = f"{output_path_base}.json" # Diarization defaults to JSON
                print(f'Saving diarization result to {save_path}')
                try:
                    with open(save_path, "w", encoding='utf-8') as f:
                        json.dump(diarization_result_dict, f, indent=2)
                except Exception as e:
                    print(f"Error saving diarization JSON for {audio_file_path}: {e}")


            elif task_to_perform == "transcribe" or task_to_perform == "translate":
                # CRITICAL FIX: Assign to 'transcription_output_dict' (or any name, e.g. 'out')
                transcription_output_dict = model.transcribe(
                    audio_file_path, 
                    task=task_to_perform,
                    language=language_arg,
                    verbose=verbose_arg
                )
                
                save_path = f"{output_path_base}.txt" # Default to .txt
                
                # Optional: Handle other formats if model.transcribe output (a dict) supports them
                if output_file_format == 'json':
                    save_path = f"{output_path_base}.json"
                    print(f'Saving full transcription dictionary to {save_path}')
                    try:
                        with open(save_path, "w", encoding='utf-8') as f:
                            json.dump(transcription_output_dict, f, indent=2) # Save the whole dict
                    except Exception as e:
                        print(f"Error saving transcription JSON for {audio_file_path}: {e}")
                else: # Default to saving only text to .txt
                    if output_file_format != 'txt':
                        print(f"Warning: Transcription task with output format '{output_file_format}' will save text to .txt. "
                              f"For full dictionary, use --output-format json.")
                    
                    print(f'Saving transcription text to {save_path}')
                    try:
                        with open(save_path, "w", encoding='utf-8') as f:
                            # Use the SAME variable name here that holds the dictionary
                            f.write(transcription_output_dict.get("text", "")) 
                    except Exception as e:
                        print(f"Error saving transcription text for {audio_file_path}: {e}")
    else:
        print("No audio files provided. Use the -f or --audio-files flag.")

if __name__ == "__main__":
    cli()