import torch
from scraibe.transcriber import load_transcriber

def main():
    device = "xpu" if torch.xpu.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load InsanelyFastWhisperTranscriber explicitly
    transcriber = load_transcriber(
        model_name="tiny",
        whisper_type="insanely-fast-whisper",
        device=device,
        compute_type="int8",
        verbose=True
    )

    # Create dummy audio (10 seconds of silence at 16kHz)
    sample_rate = 16000
    duration = 10
    audio = torch.zeros(sample_rate * duration, dtype=torch.float32)

    print("Starting transcription with InsanelyFastWhisperTranscriber...")
    result = transcriber.transcribe(audio, verbose=True)

    print("Transcription result:")
    print(f"Text: {result.get('text', '')}")
    print(f"Segments (first 3):")
    for seg in result.get("segments", [])[:3]:
        print(f"  ID {seg.get('id')}: [{seg.get('start', 0):.2f}s -> {seg.get('end', 0):.2f}s] {seg.get('text', '')}")

    print(f"Elapsed time: {result.get('elapsed_time', 'N/A')} seconds")

if __name__ == "__main__":
    main()
