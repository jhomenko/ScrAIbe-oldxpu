import torch
import numpy as np
from scraibe.transcriber import load_transcriber
from scraibe.audio import AudioProcessor

def main():
    # Load the transcriber with IPEX-LLM optimization
    print("Loading transcriber...")
    transcriber = load_transcriber(
        model_name="medium",
        whisper_type="openai-ipex-llm",
        device="cpu",  # Try with XPU first, will fall back to CPU if needed
        #low_bit="bf16", # Using bf16 which is a valid value
        #verbose=True
    )
    
    # Load audio file
    print("Loading audio file...")
    audio_processor = AudioProcessor.from_file("/mnt/data/GreenParty.m4a")
    
    # Ensure waveform is float32
    waveform = audio_processor.waveform.to(dtype=torch.float32)
    
    # Transcribe audio with batching
    print("Transcribing audio with batching...")
    result = transcriber.transcribe(
        audio=waveform,
        language="en",
        verbose=True,
        batch_size=4  # Process 4 chunks at a time
    )
    
    # Print result
    print("\nTranscription result:")
    print(result["text"])
    
    # Print segments
    print("\nSegments:")
    for segment in result["segments"]:
        print(f"[{segment['start']:.2f}s -> {segment['end']:.2f}s] {segment['text']}")

if __name__ == "__main__":
    main()
