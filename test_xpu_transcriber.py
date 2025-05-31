#!/usr/bin/env python3
"""
Test script to verify FasterWhisper with XPU acceleration.
"""
import os
import time
import torch
import numpy as np
from scraibe.transcriber import load_transcriber

def main():
    # Check if XPU is available
    xpu_available = hasattr(torch, 'xpu') and torch.xpu.is_available()
    device = "xpu" if xpu_available else "cpu"
    print(f"Using device: {device}")
    
    # Create a dummy audio file (10 seconds of silence at 16kHz)
    sample_rate = 16000
    duration = 10
    audio = np.zeros(int(sample_rate * duration), dtype=np.float32)
    
    # Add some sine waves to make it non-silent
    for i in range(5):
        freq = 440 * (i + 1)  # A4 and harmonics
        t = np.arange(0, duration, 1/sample_rate)
        audio += 0.1 * np.sin(2 * np.pi * freq * t) * np.exp(-0.5 * i)
    
    # Test FasterWhisper with XPU acceleration
    print("\n--- Testing FasterWhisper with XPU acceleration ---")
    start_time = time.time()
    
    # Check if OPENVINO_DEVICE is set
    print(f"OPENVINO_DEVICE environment variable: {os.environ.get('OPENVINO_DEVICE', 'Not set')}")
    
    # Load FasterWhisper transcriber
    transcriber = load_transcriber(
        model_name="tiny",  # Use tiny model for quick testing
        whisper_type="faster-whisper",
        device=device,
        compute_type="int8",  # Good default for OpenVINO
        verbose=True
    )
    
    # Transcribe audio
    result = transcriber.transcribe(
        audio,
        language="en",
        verbose=True,
        use_chunking=True,  # Enable chunking
        chunk_size_sec=5.0,  # Use smaller chunks for testing
        overlap_sec=0.5,     # Small overlap for testing
        num_workers=2        # Use 2 workers for parallel processing
    )
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"\nTranscription result: '{result.get('text', 'N/A')}'")
    print(f"Transcription took {elapsed_time:.2f} seconds")
    
    # Print segments
    if result.get("segments"):
        print("\nSegments:")
        for seg in result["segments"]:
            print(f"  ID {seg.get('id')}: [{seg.get('start',0):.2f}s -> {seg.get('end',0):.2f}s] {seg.get('text','N/A')}")
    
    # Test OpenAI Whisper with IPEX-LLM for comparison
    print("\n--- Testing OpenAI Whisper with IPEX-LLM for comparison ---")
    start_time = time.time()
    
    # Load OpenAI Whisper transcriber
    transcriber_openai = load_transcriber(
        model_name="tiny",
        whisper_type="openai-ipex-llm",
        device=device,
        low_bit="bf16" if device == "xpu" else "fp32",
        verbose=True
    )
    
    # Transcribe audio
    result_openai = transcriber_openai.transcribe(
        audio,
        language="en",
        verbose=True,
        use_chunking=True,
        chunk_size_sec=5.0,
        overlap_sec=0.5,
        num_workers=2
    )
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"\nTranscription result: '{result_openai.get('text', 'N/A')}'")
    print(f"Transcription took {elapsed_time:.2f} seconds")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main()
