#!/usr/bin/env python3
"""
Test script for the improved transcriber implementation.
This script tests both OpenAIWhisperIPEXLLMTranscriber and FasterWhisperTranscriber
with chunking enabled to handle long audio files efficiently.
"""

import os
import sys
import time
import argparse
import torch
import numpy as np

# Add the project root to the path so we can import scraibe modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scraibe.transcriber import load_transcriber

def main():
    parser = argparse.ArgumentParser(description="Test the improved transcriber implementation")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file")
    parser.add_argument("--model", type=str, default="tiny", help="Model name (tiny, base, small, medium, large)")
    parser.add_argument("--type", type=str, default="openai-ipex-llm", 
                        choices=["openai-ipex-llm", "faster-whisper"],
                        help="Transcriber type")
    parser.add_argument("--device", type=str, default=None, 
                        help="Device to use (cpu, cuda, xpu). If not specified, uses system default.")
    parser.add_argument("--low-bit", type=str, default="bf16", 
                        help="Low-bit format for IPEX-LLM (bf16, fp16, int4, etc.)")
    parser.add_argument("--compute-type", type=str, default="default", 
                        help="Compute type for FasterWhisper (int8, float16, etc.)")
    parser.add_argument("--chunk-size", type=float, default=30.0, 
                        help="Chunk size in seconds for long audio processing")
    parser.add_argument("--overlap", type=float, default=1.0, 
                        help="Overlap between chunks in seconds")
    parser.add_argument("--num-workers", type=int, default=4, 
                        help="Number of worker threads for parallel processing")
    parser.add_argument("--language", type=str, default=None, 
                        help="Language code (e.g., 'en', 'fr'). If not specified, auto-detect.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--output", type=str, default=None, 
                        help="Output file path. If not specified, prints to console.")
    
    args = parser.parse_args()
    
    # Load the appropriate transcriber
    print(f"Loading {args.type} transcriber with model '{args.model}' on device '{args.device or 'default'}'...")
    
    transcriber_kwargs = {
        "verbose": args.verbose
    }
    
    if args.type == "openai-ipex-llm":
        transcriber_kwargs["low_bit"] = args.low_bit
        transcriber_kwargs["use_ipex_llm"] = True
    elif args.type == "faster-whisper":
        transcriber_kwargs["compute_type"] = args.compute_type
    
    start_time = time.time()
    
    transcriber = load_transcriber(
        model_name=args.model,
        whisper_type=args.type,
        device=args.device,
        **transcriber_kwargs
    )
    
    load_time = time.time() - start_time
    print(f"Transcriber loaded in {load_time:.2f} seconds")
    
    # Load audio file
    print(f"Processing audio file: {args.audio}")
    
    # Load audio using AudioProcessor from scraibe
    try:
        from scraibe.audio import AudioProcessor
        audio_processor = AudioProcessor.from_file(args.audio)
        audio_waveform = audio_processor.waveform
        sample_rate = audio_processor.sr
        print(f"Loaded audio: {audio_waveform.shape}, sample rate: {sample_rate}Hz")
    except Exception as e:
        print(f"Error loading audio with AudioProcessor: {e}")
        print("Falling back to soundfile...")
        try:
            import soundfile as sf
            audio_data, sample_rate = sf.read(args.audio, dtype='float32')
            audio_waveform = torch.from_numpy(audio_data)
            if audio_waveform.ndim > 1:
                # Convert stereo to mono by averaging channels
                audio_waveform = audio_waveform.mean(dim=1)
            print(f"Loaded audio with soundfile: {audio_waveform.shape}, sample rate: {sample_rate}Hz")
        except Exception as e2:
            print(f"Error loading audio with soundfile: {e2}")
            sys.exit(1)
    
    # Transcribe with chunking enabled
    transcribe_kwargs = {
        "verbose": args.verbose,
        "use_chunking": True,
        "chunk_size_sec": args.chunk_size,
        "overlap_sec": args.overlap,
        "num_workers": args.num_workers
    }
    
    if args.language:
        transcribe_kwargs["language"] = args.language
    
    start_time = time.time()
    
    result = transcriber.transcribe(
        audio_waveform,
        **transcribe_kwargs
    )
    
    transcribe_time = time.time() - start_time
    audio_duration = None
    
    # Try to get audio duration for reporting
    try:
        import soundfile as sf
        info = sf.info(args.audio)
        audio_duration = info.duration
    except Exception as e:
        print(f"Could not determine audio duration: {e}")
    
    # Print results
    print("\n--- Transcription Results ---")
    print(f"Detected language: {result.get('language', 'unknown')}")
    if audio_duration:
        print(f"Audio duration: {audio_duration:.2f} seconds")
        print(f"Transcription time: {transcribe_time:.2f} seconds ({transcribe_time/audio_duration:.2f}x real-time)")
    else:
        print(f"Transcription time: {transcribe_time:.2f} seconds")
    
    print(f"Number of segments: {len(result.get('segments', []))}")
    print("\nTranscribed text:")
    print(result.get("text", ""))
    
    # Save to file if specified
    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(result.get("text", ""))
            print(f"\nTranscription saved to: {args.output}")
        except Exception as e:
            print(f"Error saving transcription to file: {e}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
