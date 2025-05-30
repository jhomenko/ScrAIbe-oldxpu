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

    if audio_files_to_process:
        # Pop method-specific arguments once before the loop
        language_arg = arg_dict.pop("language")
        verbose_arg = arg_dict.pop("verbose_output")
        num_speakers_arg = arg_dict.pop("num_speakers")

        for audio_file_path in audio_files_to_process:
            base_filename = os.path.splitext(os.path.basename(audio_file_path))[0]
            output_path_base = os.path.join(out_folder, base_filename)

            print(f"\nProcessing: {audio_file_path}")

            if task_to_perform == "autotranscribe" or task_to_perform == "autotranscribe+translate":
                current_task_for_method = "translate" if task_to_perform == "autotranscribe+translate" else "transcribe"
                
                autotranscribe_params = {
                    'task': current_task_for_method,
                    'language': language_arg,
                    'verbose': verbose_arg,
                }
                if num_speakers_arg is not None: # Only pass num_speakers if provided
                    autotranscribe_params['num_speakers'] = num_speakers_arg

                out = model.autotranscribe(audio_file_path, **autotranscribe_params)
                
                # Assuming 'out' is an object with a save method like: out.save(path, format=output_file_format)
                # If 'out' is just the text for .txt, and dict for .json etc., saving needs to adapt.
                # For now, let's assume a versatile save method or specific handling.
                print(f'Saving to {output_path_base} with format {output_file_format}')
                if hasattr(out, 'save_to_file'): # Ideal
                     out.save_to_file(output_path_base, format=output_file_format)
                elif output_file_format == 'txt' and isinstance(out, str): # Simple text case
                     with open(f"{output_path_base}.txt", "w", encoding='utf-8') as f:
                        f.write(out)
                elif output_file_format == 'json' and isinstance(out, dict): # Simple dict case for JSON
                     with open(f"{output_path_base}.json", "w", encoding='utf-8') as f:
                        json.dump(out, f, indent=2)
                else: # Fallback or more complex save logic from Scraibe output object
                    # This part needs to align with how your Scraibe output objects work.
                    # The original code had out.save(path_with_extension)
                    # Let's assume the output object `out` from autotranscribe has a method `save(filepath_with_extension)`
                    # And it infers format from extension or you need to pass it.
                    # For simplicity, if your original `out.save()` worked, that's fine.
                    # This example makes the output filename include the format.
                    save_path = f"{output_path_base}.{output_file_format}"
                    if hasattr(out, 'save'):
                        out.save(save_path) # Assuming out.save handles the format based on extension or internally
                    else:
                        print(f"Warning: Output object from autotranscribe doesn't have a 'save' or 'save_to_file' method. Cannot save {output_file_format}.")


            elif task_to_perform == "diarization":
                if verbose_arg: # Check the verbose_arg captured before loop
                    print("Performing diarization...") # Simple verbose message
                
                # Assuming model.diarization returns a JSON serializable dict/list
                diarization_result = model.diarization(audio_file_path, num_speakers=num_speakers_arg) # Pass num_speakers if diarizer uses it
                
                save_path = f"{output_path_base}.json" # Diarization often saved as JSON
                print(f'Saving diarization result to {save_path}')
                with open(save_path, "w", encoding='utf-8') as f:
                    json.dump(diarization_result, f, indent=2)


            elif task_to_perform == "transcribe" or task_to_perform == "translate":
                # Assuming model.transcribe returns the transcribed text directly as a string
                transcribed_text = model.transcribe(
                    audio_file_path, 
                    task=task_to_perform, # 'transcribe' or 'translate'
                    language=language_arg,
                    verbose=verbose_arg
                )
                save_path = f"{output_path_base}.txt" # Default to .txt for direct transcription
                if output_file_format != 'txt':
                    print(f"Warning: Direct transcribe/translate task defaults to .txt output. Requested format '{output_file_format}' may not be suitable if output is plain text.")
                
                print(f'Saving transcription to {save_path}')
                with open(save_path, "w", encoding='utf-8') as f:
                    f.write(transcribed_text)
    else:
        print("No audio files provided. Use the -f or --audio-files flag.")

if __name__ == "__main__":
    cli()