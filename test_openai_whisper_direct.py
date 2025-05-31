#!/usr/bin/env python3
"""
Test script to verify OpenAI Whisper (original implementation) with IPEX acceleration.
This script uses the original OpenAI Whisper model directly, not the Hugging Face version.
"""
import os
import time
import torch
import numpy as np
import whisper  # Original OpenAI Whisper

try:
    import intel_extension_for_pytorch as ipex
    print("INFO: Intel Extension for PyTorch (IPEX) library found.")
except ImportError:
    ipex = None
    print("WARNING: Intel Extension for PyTorch (IPEX) library not found.")

# Constants
SAMPLE_RATE = 16000

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
    
    # Test OpenAI Whisper (original implementation)
    print("\n--- Testing Original OpenAI Whisper with IPEX ---")
    start_time = time.time()
    
    # Load model directly
    model_name = "tiny"
    print(f"Loading model: {model_name}")
    
    # Load the model
    model = whisper.load_model(model_name)
    
    # Move model to device and optimize with IPEX if available
    if device == "xpu" and ipex is not None:
        print(f"Moving model to {device} and applying IPEX optimization")
        model = model.to(device)
        # Try to optimize with IPEX
        try:
            model_dtype = torch.bfloat16
            model = ipex.optimize(model.eval(), dtype=model_dtype, inplace=True, weights_prepack=False)
            print(f"IPEX optimization applied with dtype={model_dtype}")
        except Exception as e:
            print(f"IPEX optimization failed: {e}. Using unoptimized model.")
    else:
        model = model.to(device)
    
    # Convert audio to tensor
    audio_tensor = torch.from_numpy(audio).to(torch.float32)
    
    # Transcribe options
    transcribe_options = {
        "language": "en",
        "task": "transcribe",
        "temperature": 0.0,
        "beam_size": 1,
        "verbose": True
    }
    
    print(f"Transcribing audio with options: {transcribe_options}")
    
    # Transcribe using the original OpenAI Whisper model
    try:
        # The original OpenAI Whisper model's transcribe method handles the audio processing internally
        result = model.transcribe(audio, **transcribe_options)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print(f"\nTranscription result: '{result['text']}'")
        print(f"Transcription took {elapsed_time:.2f} seconds")
        
        # Print segments
        if "segments" in result and result["segments"]:
            print("\nSegments:")
            for i, seg in enumerate(result["segments"]):
                print(f"  ID {i}: [{seg.get('start', 0):.2f}s -> {seg.get('end', 0):.2f}s] {seg.get('text', '')}")
        
    except Exception as e:
        print(f"Error during transcription: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nTest completed!")

if __name__ == "__main__":
    main()
