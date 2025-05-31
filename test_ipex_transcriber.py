import torch
import numpy as np
from scraibe.transcriber import load_transcriber
from scraibe.audio import AudioProcessor

def main():
    # Load the transcriber with IPEX-LLM optimization
    print("Loading transcriber...")
    transcriber = load_transcriber(
        model_name="tiny",
        whisper_type="openai-ipex-llm",
        device="xpu",
        low_bit="bf16",
        #verbose=True
    )
    
    # Load audio file
    print("Loading audio file...")
    audio_processor = AudioProcessor.from_file("tests/audio_test_1.mp4")
    
    # Transcribe audio
    print("Transcribing audio...")
    result = transcriber.transcribe(
        audio=audio_processor.waveform,
        language="en",
        verbose=True
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
