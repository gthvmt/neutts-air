import os
import soundfile as sf
import torch
import numpy as np
import time
from neuttsair.neutts import NeuTTSAir


def _read_if_path(value: str) -> str:
    """If the given string is a file path, read and return its contents."""
    return open(value, "r", encoding="utf-8").read().strip() if os.path.exists(value) else value


def main(input_text, ref_codes_path, ref_text, backbone, codec, output_path):
    # Initialize NeuTTSAir
    tts = NeuTTSAir(
        backbone_repo=backbone,
        backbone_device="cpu",
        codec_repo=codec,
        codec_device="cpu",
    )

    input_text = _read_if_path(input_text)
    ref_text = _read_if_path(ref_text)

    ref_codes = None
    if ref_codes_path and os.path.exists(ref_codes_path):
        ref_codes = torch.load(ref_codes_path)

    # Streaming diagnostics
    word_count = len(input_text.split())
    chunk_samples = tts.streaming_stride_samples
    chunk_ms = chunk_samples / tts.sample_rate * 1000
    print(f"Input text: {word_count} words / {len(input_text)} chars")
    if ref_codes is not None:
        shape = getattr(ref_codes, "shape", None)
        ref_summary = "x".join(str(dim) for dim in shape) if shape is not None else str(len(ref_codes))
        print(f"Reference codes shape: {ref_summary}")
    print(f"Streaming frames per chunk: {tts.streaming_frames_per_chunk}, hop length: {tts.hop_length}")
    print(f"Each chunk ~{chunk_ms:.1f} ms of audio")

    # Generate audio chunks
    print(f"Generating audio for input text: {input_text}")
    total_audio_samples = 0
    total_lm_time = 0.0
    chunk_count = 0
    last_yield_time = None
    start_time = time.perf_counter()

    print("Streaming...")
    print("-" * 80)

    with sf.SoundFile(output_path, mode='w', samplerate=tts.sample_rate, channels=1, subtype='PCM_16') as f:
        for chunk in tts.infer_to_stream(input_text, ref_codes, ref_text):
            chunk_count += 1
            now = time.perf_counter()
            lm_duration = None
            if last_yield_time is not None:
                lm_duration = now - last_yield_time
                total_lm_time += lm_duration
            last_yield_time = now

            # Convert float32 audio to int16
            audio = (chunk * 32767).astype(np.int16)
            f.write(audio)
            total_audio_samples += audio.shape[0]

            # Per-chunk timing log for latency info
            chunk_ms_actual = audio.shape[0] / tts.sample_rate * 1000
            lm_ms = f"{lm_duration * 1000:6.1f}ms" if lm_duration is not None else "  n/a "
            rt_percent = (
                (lm_duration / (chunk_ms_actual / 1000) * 100)
                if lm_duration is not None else 0.0
            )
            print(
                f"Chunk {chunk_count:2d}: "
                f"LM={lm_ms} │ Audio={chunk_ms_actual:5.1f}ms │ {rt_percent:5.1f}% RT"
            )

        # Add a short tail pad
        tail_pad = np.zeros(int(0.25 * tts.sample_rate), dtype=np.int16)
        f.write(tail_pad)

        print(f"\n✅ Audio written to: {output_path}")

    total_time = time.perf_counter() - start_time
    total_audio_seconds = total_audio_samples / tts.sample_rate if total_audio_samples else 0.0

    # Print stats
    print("-" * 80)
    print(f"Streaming complete. Generated {total_audio_seconds:.2f}s of audio in {total_time:.2f}s.")
    if chunk_count:
        print(f"  → Average Speech LM time per chunk: {(total_lm_time / chunk_count) * 1000:.1f}ms")
        if total_audio_seconds:
            rtf = total_time / total_audio_seconds
            print(f"  → Real-Time Factor (RTF): {rtf:.2f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NeuTTSAir Example")
    parser.add_argument("--input_text", type=str, required=True, help="Input text to be converted to speech")
    parser.add_argument("--ref_codes", type=str, default="./samples/dave.pt", help="Path to pre-encoded reference audio")
    parser.add_argument("--ref_text", type=str, default="./samples/dave.txt", help="Reference text corresponding to the reference audio")
    parser.add_argument("--output_path", type=str, default="output.wav", help="Path to save the output audio")
    parser.add_argument("--backbone", type=str, default="neuphonic/neutts-air-q8-gguf", help="Huggingface repo containing the backbone checkpoint. Must be GGUF.")
    parser.add_argument("--codec", type=str, default="neuphonic/neucodec-onnx-decoder", help="Huggingface repo containing the codec checkpoint. Must be ONNX.")
    args = parser.parse_args()

    main(
        input_text=args.input_text,
        ref_codes_path=args.ref_codes,
        ref_text=args.ref_text,
        backbone=args.backbone,
        codec=args.codec,
        output_path=args.output_path
    )
