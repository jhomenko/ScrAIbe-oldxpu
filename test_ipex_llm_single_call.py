#!/usr/bin/env python3
"""
Test script to verify OpenAI Whisper with IPEX-LLM acceleration using a single call approach.
This script follows the approach from the working implementation in 'scraibe/transcriber working old.py'.
"""
import os
import time
import torch
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import intel_extension_for_pytorch as ipex

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
    
    # Convert to tensor
    audio_tensor = torch.from_numpy(audio).to(torch.float32)
    
    # Test OpenAI Whisper with IPEX-LLM
    print("\n--- Testing OpenAI Whisper with IPEX-LLM (Single Call Approach) ---")
    start_time = time.time()
    
    # Load processor and model directly (without using the Transcriber class)
    model_name = "openai/whisper-tiny"
    print(f"Loading model: {model_name}")
    
    # Load processor
    processor = WhisperProcessor.from_pretrained(model_name)
    
    # Load model with BF16 precision for XPU
    print(f"Loading model with BF16 precision for {device}")
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    # Apply IPEX optimization
    print("Applying IPEX optimization")
    model = ipex.optimize(model.eval(), dtype=torch.bfloat16, inplace=True, weights_prepack=False)
    model = model.to(device)
    
    # Process audio
    print("Processing audio")
    inputs = processor(
        audio_tensor.cpu().numpy(), 
        sampling_rate=SAMPLE_RATE, 
        return_tensors="pt",
        return_attention_mask=True,
        truncation=False
    )
    input_features = inputs.input_features.to(device).to(torch.bfloat16)
    attention_mask = inputs.attention_mask.to(device) if hasattr(inputs, "attention_mask") else None
    
    # Prepare generate options
    generate_options = {
        'language': 'en',
        'task': 'transcribe',
        'return_timestamps': True,
        'return_dict_in_generate': True,
        'condition_on_prev_tokens': True,
        'temperature': 0.0,  # Use a single float value instead of a tuple
        'num_beams': 1,
        'do_sample': False,
        'use_cache': True
    }
    
    print(f"Calling model.generate() with options: {generate_options}")
    
    # Generate transcription
    with torch.no_grad():
        try:
            output = model.generate(
                input_features=input_features,
                attention_mask=attention_mask,
                **generate_options
            )
            
            if device == "xpu":
                torch.xpu.synchronize()
            
            # Decode output
            predicted_ids = output.sequences if hasattr(output, "sequences") else output
            full_text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
            
            # Extract segments if available
            segments_data = []
            raw_segments = None
            if hasattr(output, "segments") and output.segments is not None:
                raw_segments = output.segments
            elif hasattr(output, "chunks") and output.chunks is not None:
                raw_segments = output.chunks
            
            if raw_segments and isinstance(raw_segments, list):
                for i, seg_data in enumerate(raw_segments):
                    text = seg_data.get("text", "").strip()
                    ts = seg_data.get("timestamp", (0.0, 0.0))
                    start_time_seg = ts[0] if isinstance(ts, (list, tuple)) and len(ts) > 0 else 0.0
                    end_time_seg = ts[1] if isinstance(ts, (list, tuple)) and len(ts) > 1 and ts[1] is not None else start_time_seg
                    segments_data.append({
                        "id": i,
                        "start": round(float(start_time_seg), 3),
                        "end": round(float(end_time_seg), 3),
                        "text": text
                    })
            elif full_text:
                segments_data.append({
                    "id": 0,
                    "start": 0.0,
                    "end": round(duration, 3),
                    "text": full_text
                })
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            print(f"\nTranscription result: '{full_text}'")
            print(f"Transcription took {elapsed_time:.2f} seconds")
            
            # Print segments
            if segments_data:
                print("\nSegments:")
                for seg in segments_data:
                    print(f"  ID {seg['id']}: [{seg['start']:.2f}s -> {seg['end']:.2f}s] {seg['text']}")
            
        except Exception as e:
            print(f"Error during transcription: {e}")
            import traceback
            traceback.print_exc()
    
    print("\nTest completed!")

if __name__ == "__main__":
    main()
