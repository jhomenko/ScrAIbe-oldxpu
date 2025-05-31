#!/usr/bin/env python3
"""
Test script to verify OpenAI Whisper with IPEX-LLM acceleration.
This script focuses on the non-chunked approach which was working in the original implementation.
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
    
    # Test OpenAI Whisper with IPEX-LLM
    print("\n--- Testing OpenAI Whisper with IPEX-LLM ---")
    start_time = time.time()
    
    # Load OpenAI Whisper transcriber with IPEX-LLM
    transcriber = load_transcriber(
        model_name="tiny",  # Use tiny model for quick testing
        whisper_type="openai-ipex-llm",
        device=device,
        low_bit="bf16" if device == "xpu" else "fp32",
        use_ipex_llm=True,
        verbose=True
    )
    
    # Transcribe audio WITHOUT chunking first (this was working in the original implementation)
    print("\n--- Testing WITHOUT chunking ---")
    result = transcriber.transcribe(
        audio,
        language="en",
        verbose=True,
        use_chunking=False  # Disable chunking
    )
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"\nTranscription result: '{result.get('text', 'N/A')}'")
    print(f"Transcription took {elapsed_time:.2f} seconds")
    
    # Print segments
    if result and result.get("segments"):
        print("\nSegments:")
        for seg in result["segments"]:
            print(f"  ID {seg.get('id')}: [{seg.get('start',0):.2f}s -> {seg.get('end',0):.2f}s] {seg.get('text','N/A')}")
    
    # Now test WITH chunking (this was failing)
    print("\n--- Testing WITH chunking ---")
    start_time = time.time()
    
    result_chunked = transcriber.transcribe(
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
    
    print(f"\nChunked transcription result: '{result_chunked.get('text', 'N/A') if result_chunked else 'Failed'}'")
    print(f"Chunked transcription took {elapsed_time:.2f} seconds")
    
    # Print segments
    if result_chunked and result_chunked.get("segments"):
        print("\nChunked segments:")
        for seg in result_chunked["segments"]:
            print(f"  ID {seg.get('id')}: [{seg.get('start',0):.2f}s -> {seg.get('end',0):.2f}s] {seg.get('text','N/A')}")
    
    print("\nTest completed!")

if __name__ == "__main__":
    main()
