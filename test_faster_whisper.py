#!/usr/bin/env python3
"""
Test script to verify FasterWhisper with Intel XPU acceleration.
FasterWhisper uses CTranslate2 backend which might avoid the Conv1D issue.
"""
import os
import time
import torch
import numpy as np
from faster_whisper import WhisperModel

# Constants
SAMPLE_RATE = 16000

def main():
    # Check if XPU is available
    xpu_available = hasattr(torch, 'xpu') and torch.xpu.is_available()
    print(f"XPU available: {xpu_available}")
    
    # For FasterWhisper on XPU, we typically use device="cpu" with compute_type="int8"
    # This leverages OpenVINO backend which can use the XPU
    device = "cpu"  # Use "cpu" even for XPU with FasterWhisper
    compute_type = "int8"  # Good default for OpenVINO backend
    
    if xpu_available:
        print(f"Using device='cpu' with compute_type='{compute_type}' for FasterWhisper with XPU acceleration via OpenVINO")
    else:
        print(f"Using device='{device}' with compute_type='{compute_type}' for FasterWhisper")
    
    # Create a dummy audio file (10 seconds of silence at 16kHz)
    sample_rate = 16000
    duration = 10
    audio = np.zeros(int(sample_rate * duration), dtype=np.float32)
    
    # Add some sine waves to make it non-silent
    for i in range(5):
        freq = 440 * (i + 1)  # A4 and harmonics
        t = np.arange(0, duration, 1/sample_rate)
        audio += 0.1 * np.sin(2 * np.pi * freq * t) * np.exp(-0.5 * i)
    
    # Test FasterWhisper
    print("\n--- Testing FasterWhisper ---")
    start_time = time.time()
    
    # Load model directly
    model_name = "tiny"
    print(f"Loading model: {model_name}")
    
    # Load the model with appropriate settings for XPU
    model = WhisperModel(
        model_size_or_path=model_name,
        device=device,
        compute_type=compute_type,
        cpu_threads=4,  # Adjust based on your system
        num_workers=1   # Adjust based on your system
    )
    
    # Transcribe options
    transcribe_options = {
        "language": "en",
        "task": "transcribe",
        "beam_size": 1,
        "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],  # FasterWhisper can use temperature list
        "vad_filter": True,  # Enable Voice Activity Detection
        "vad_parameters": {"threshold": 0.5},  # Adjust VAD parameters
        "word_timestamps": True  # Get word-level timestamps
    }
    
    print(f"Transcribing audio with options: {transcribe_options}")
    
    # Transcribe using FasterWhisper
    try:
        # FasterWhisper's transcribe method returns a generator of segments and info
        segments, info = model.transcribe(audio, **transcribe_options)
        
        # Convert generator to list to get all segments
        segments_list = list(segments)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Combine all segment texts to get the full transcription
        full_text = " ".join([segment.text for segment in segments_list]).strip()
        
        print(f"\nTranscription result: '{full_text}'")
        print(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")
        print(f"Transcription took {elapsed_time:.2f} seconds")
        
        # Print segments
        if segments_list:
            print("\nSegments:")
            for i, segment in enumerate(segments_list):
                print(f"  ID {i}: [{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
                if hasattr(segment, 'words') and segment.words:
                    print("    Words:")
                    for word in segment.words:
                        print(f"      [{word.start:.2f}s -> {word.end:.2f}s] {word.word}")
        
    except Exception as e:
        print(f"Error during transcription: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nTest completed!")

if __name__ == "__main__":
    main()
