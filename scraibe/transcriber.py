# --- Your if __name__ == '__main__': block (ensure it passes verbose if you test here) ---
# ...

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
        openai_low_bit = 'bf16' if openai_device == "xpu" else None # BF16 on XPU, no low-bit quant on CPU for this example

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
        # transcriber_openai.save_transcript(result_openai, "openai_ipex_transcript.txt")

    except Exception as e:
        print(f"Error testing OpenAIWhisperIPEXLLMTranscriber: {e}")
        import traceback
        traceback.print_exc()

    # --- Test FasterWhisper ---
    print("\n--- Testing FasterWhisperTranscriber ---")
    try:
        fw_device = "cuda" if torch.cuda.is_available() else "cpu"
        fw_compute = "float16" if fw_device == "cuda" else "int8"
        
        transcriber_faster = load_transcriber(
            model_name="tiny",
            whisper_type="faster-whisper",
            device=fw_device,
            compute_type=fw_compute # Specific kwarg for FasterWhisper
        )
        print(f"Loaded FasterWhisper transcriber on {fw_device} with compute_type='{fw_compute}'")

        result_faster = transcriber_faster.transcribe(audio_source, language="en") # Pass language kwarg
        print(f"FasterWhisper Text: '{result_faster.get('text', 'N/A')}'")
        if result_faster.get("segments"):
            print("Segments (first 3):")
            for seg in result_faster["segments"][:3]:
                 print(f"  ID {seg.get('id')}: [{seg.get('start',0):.2f}s -> {seg.get('end',0):.2f}s] {seg.get('text','N/A')}")
        # transcriber_faster.save_transcript(result_faster, "faster_whisper_transcript.txt")

    except Exception as e:
        print(f"Error testing FasterWhisperTranscriber: {e}")
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