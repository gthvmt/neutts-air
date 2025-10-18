from typing import Generator
from pathlib import Path
import os
import re

import librosa
import numpy as np
import torch

from neucodec import NeuCodec, DistillNeuCodec
from phonemizer.backend import EspeakBackend
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import AsyncGenerator

# Optional: for intelligent sentence splitting
try:
    import nltk

    NLTK_AVAILABLE = True
    # Try to use punkt tokenizer, download if needed
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        print("Downloading NLTK punkt tokenizer (one-time setup)...")
        nltk.download("punkt_tab", quiet=True)
except ImportError:
    NLTK_AVAILABLE = False


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def smooth_audio_overlap(frames: list[np.ndarray], stride: int) -> np.ndarray:
    """
    Combine overlapping audio frames smoothly to avoid clicking sounds.

    When generating audio in chunks, we need to blend the overlapping parts
    so transitions sound natural. This uses a triangular weighting function
    to smoothly fade between frames.

    Args:
        frames: List of audio segments (numpy arrays)
        stride: Number of samples to advance between frames

    Returns:
        Single combined audio array

    Original implementation from:
    https://github.com/facebookresearch/encodec/blob/main/encodec/utils.py
    """
    if not frames:
        raise ValueError("Cannot combine empty list of frames")

    # Get audio properties from first frame
    dtype = frames[0].dtype
    shape = frames[0].shape[:-1]  # All dimensions except the last (time)

    # Calculate total length needed for combined audio
    total_size = 0
    for i, frame in enumerate(frames):
        frame_end = stride * i + frame.shape[-1]
        total_size = max(total_size, frame_end)

    # Initialize arrays for weighted sum and weight tracking
    sum_weight = np.zeros(total_size, dtype=dtype)
    combined_audio = np.zeros(*shape, total_size, dtype=dtype)

    # Blend each frame into the output
    offset = 0
    for frame in frames:
        frame_length = frame.shape[-1]

        # Create triangular weight: starts at 0, peaks at 0.5, ends at 0
        # This creates smooth transitions at frame boundaries
        t = np.linspace(0, 1, frame_length + 2, dtype=dtype)[1:-1]
        weight = np.abs(0.5 - (t - 0.5))

        # Add weighted frame to output
        combined_audio[..., offset : offset + frame_length] += weight * frame
        sum_weight[offset : offset + frame_length] += weight
        offset += stride

    # Normalize by total weight (ensures consistent volume)
    assert sum_weight.min() > 0, "Some samples have zero weight"
    return combined_audio / sum_weight


# ============================================================================
# MAIN TTS CLASS
# ============================================================================


class NeuTTSAir:
    """
    Neural Text-to-Speech System

    This class handles the complete pipeline for converting text to speech:
    - Loading pre-trained models (language model + audio codec)
    - Converting text to phonemes (pronunciation)
    - Generating speech tokens from text
    - Decoding tokens to audio waveforms

    Key Components:
    - Backbone: The "brain" - a language model that predicts speech patterns
    - Codec: The "voice" - converts patterns into actual audio
    - Phonemizer: Converts written text to pronunciation
    """

    # ========================================================================
    # INITIALIZATION
    # ========================================================================

    def __init__(
        self,
        backbone_repo="neuphonic/neutts-air",
        backbone_device="cpu",
        codec_repo="neuphonic/neucodec",
        codec_device="cpu",
        local_files_only=False,
    ):
        """
        Initialize the TTS system with models.

        Args:
            backbone_repo: Path or HuggingFace repo for the language model
            backbone_device: Where to run backbone ("cpu" or "cuda" for GPU)
            codec_repo: Path or HuggingFace repo for the audio codec
            codec_device: Where to run codec ("cpu" or "cuda" for GPU)
            local_files_only: If True, only load from disk (no downloads)
        """
        print("Initializing NeuTTS Air...")

        # Model configuration flags
        self._is_quantized_model = False  # Quantized = compressed for speed
        self._is_onnx_codec = False  # ONNX = optimized format
        self.local_files_only = local_files_only

        # Audio settings (don't change these - they're trained into the model)
        self.sample_rate = 24_000  # Audio quality: 24kHz
        self.hop_length = 480  # Samples per speech token

        # Model context limits
        self.max_context = 2048  # Maximum tokens the model can process

        # Estimate tokens per second of audio (conservative estimate)
        # The model uses ~68 tokens per second of audio
        self.tokens_per_second = 68

        # Reserve tokens for prompt overhead (reference audio + formatting)
        self.prompt_overhead_tokens = 200  # Conservative estimate

        # Streaming settings (for real-time generation)
        self.streaming_overlap_frames = 1  # Frames to overlap for smooth blending
        self.streaming_frames_per_chunk = 25  # Tokens to generate per chunk
        self.streaming_lookforward = 5  # Future context for better quality
        self.streaming_lookback = 50  # Past context for consistency
        self.streaming_stride_samples = (
            self.streaming_frames_per_chunk * self.hop_length
        )
        self.text_buffer = ""  # Buffer for streaming text input

        # Tokenizer for text processing (may remain None for quantized models)
        self.tokenizer = None

        # Load the components
        print("Models in local models directory:", os.listdir("./models"))
        print("Samples in local sample directory:", os.listdir("./samples"))

        self._load_phonemizer()
        self._load_backbone(backbone_repo, backbone_device)
        self._load_codec(codec_repo, codec_device)

        print("✓ NeuTTS Air ready!")

    # ========================================================================
    # MODEL LOADING
    # ========================================================================

    def _load_phonemizer(self):
        """
        Load the phonemizer (text → pronunciation converter).

        Example: "hello" → "həˈloʊ"
        This helps the model understand how words should sound.
        """
        print("Loading phonemizer...")
        self.phonemizer = EspeakBackend(
            language="en-us",
            preserve_punctuation=True,  # Keep commas, periods, etc.
            with_stress=True,  # Mark which syllables to emphasize
        )

    def _is_local_path(self, path: str) -> bool:
        """Check if a path points to a local file/directory."""
        return os.path.exists(path) or path.endswith(".gguf")

    def _load_backbone(self, backbone_repo: str, backbone_device: str):
        """
        Load the language model "backbone" - the AI brain of the system.

        This model learns the relationship between text and speech patterns.
        It can be loaded in two formats:
        - GGUF: Quantized (compressed) format, runs faster, slightly lower quality
        - PyTorch: Full precision format, higher quality, slower

        Args:
            backbone_repo: Where to load the model from
            backbone_device: cpu or cuda (GPU)
        """
        print(f"Loading backbone from: {backbone_repo} on {backbone_device}...")

        is_gguf = backbone_repo.endswith("gguf") or (
            self._is_local_path(backbone_repo) and Path(backbone_repo).suffix == ".gguf"
        )

        if is_gguf:
            self._load_gguf_backbone(backbone_repo, backbone_device)
        else:
            self._load_pytorch_backbone(backbone_repo, backbone_device)

    def _load_gguf_backbone(self, backbone_repo: str, backbone_device: str):
        """Load quantized (GGUF) model using llama-cpp."""
        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise ImportError(
                "GGUF models require llama-cpp-python. Install with:\n"
                "    pip install llama-cpp-python"
            ) from e

        # Determine GPU layers (-1 = all layers on GPU)
        n_gpu_layers = -1 if backbone_device == "gpu" else 0

        common_params = {
            "verbose": False,
            "n_gpu_layers": n_gpu_layers,
            "n_ctx": self.max_context,
            "mlock": True,  # Lock model in RAM to prevent swapping
            "flash_attn": backbone_device == "gpu",  # Fast attention on GPU
        }

        if os.path.isfile(backbone_repo):
            # Load from local file
            self.backbone = Llama(model_path=backbone_repo, **common_params)
        else:
            # Download from HuggingFace
            self.backbone = Llama.from_pretrained(
                repo_id=backbone_repo, filename="*.gguf", **common_params
            )

        self._is_quantized_model = True

    def _load_pytorch_backbone(self, backbone_repo: str, backbone_device: str):
        """Load full-precision PyTorch model."""
        load_kwargs = {}
        if self.local_files_only or self._is_local_path(backbone_repo):
            load_kwargs["local_files_only"] = True

        # Load tokenizer (converts text to numbers)
        self.tokenizer = AutoTokenizer.from_pretrained(backbone_repo, **load_kwargs)

        # Load model and move to device (CPU or GPU)
        self.backbone = AutoModelForCausalLM.from_pretrained(
            backbone_repo, **load_kwargs
        ).to(torch.device(backbone_device))

    def _load_codec(self, codec_repo: str, codec_device: str):
        """
        Load the audio codec - converts between speech tokens and audio.

        Think of it like an audio translator:
        - Encoder: audio → compact tokens
        - Decoder: tokens → audio

        We mainly use the decoder here to generate speech.

        Args:
            codec_repo: Where to load codec from
            codec_device: cpu or cuda (GPU)
        """
        print(f"Loading codec from: {codec_repo} on {codec_device}...")

        # Prepare loading arguments
        load_kwargs = {}
        if self.local_files_only or self._is_local_path(codec_repo):
            load_kwargs["local_files_only"] = True

        # Determine codec type from path
        codec_path = Path(codec_repo) if self._is_local_path(codec_repo) else None
        codec_name = codec_path.name if codec_path else codec_repo.split("/")[-1]

        # Load appropriate codec type
        if "neucodec-onnx-decoder" in codec_name:
            self._load_onnx_codec(codec_repo, codec_device, load_kwargs, codec_path)
        elif "distill-neucodec" in codec_name:
            self._load_distilled_codec(codec_repo, codec_device, load_kwargs)
        elif "neucodec" in codec_name:
            self._load_standard_codec(codec_repo, codec_device, load_kwargs)
        else:
            raise ValueError(
                f"Unknown codec type: {codec_repo}\n"
                "Path must contain 'neucodec', 'distill-neucodec', "
                "or 'neucodec-onnx-decoder'"
            )

    def _load_onnx_codec(self, codec_repo, codec_device, load_kwargs, codec_path):
        """Load ONNX optimized codec (CPU only)."""
        if codec_device != "cpu":
            raise ValueError("ONNX decoder only runs on CPU")

        try:
            from neucodec import NeuCodecOnnxDecoder
        except ImportError as e:
            raise ImportError(
                "ONNX decoder requires:\n" "    pip install onnxruntime neucodec>=0.0.4"
            ) from e

        if codec_path:
            self.codec = NeuCodecOnnxDecoder(codec_path)
        else:
            self.codec = NeuCodecOnnxDecoder.from_pretrained(codec_repo, **load_kwargs)
        self._is_onnx_codec = True

    def _load_distilled_codec(self, codec_repo, codec_device, load_kwargs):
        """Load distilled (compressed) codec."""
        self.codec = DistillNeuCodec.from_pretrained(codec_repo, **load_kwargs)
        self.codec.eval().to(codec_device)

    def _load_standard_codec(self, codec_repo, codec_device, load_kwargs):
        """Load standard full-size codec."""
        self.codec = NeuCodec.from_pretrained(codec_repo, **load_kwargs)
        self.codec.eval().to(codec_device)

    # ========================================================================
    # CORE INFERENCE METHODS
    # ========================================================================

    def infer(
        self,
        text: str,
        ref_codes: np.ndarray | torch.Tensor,
        ref_text: str,
        auto_split: bool = True,
    ) -> np.ndarray:
        """
        Generate speech from text (non-streaming version).

        This is the main method for generating speech. It needs:
        1. The text you want spoken
        2. Reference audio (encoded as codes) to clone the voice
        3. What the reference audio says (for context)

        Process:
        text + reference → language model → speech tokens → codec → audio

        NEW: Automatically splits long text into chunks if it exceeds context window!

        Args:
            text: What you want the system to say
            ref_codes: Encoded reference audio (defines the voice style)
            ref_text: Transcript of the reference audio
            auto_split: If True, automatically chunk long text (default: True)

        Returns:
            Audio waveform as numpy array (play with soundfile, scipy, etc.)
        """
        # Check if text needs chunking
        if auto_split and self._should_split_text(text, ref_codes):
            return self._infer_with_chunking(text, ref_codes, ref_text)

        # Standard single-chunk inference
        if self._is_quantized_model:
            output_str = self._generate_tokens_ggml(ref_codes, ref_text, text)
        else:
            prompt_ids = self._build_prompt(ref_codes, ref_text, text)
            output_str = self._generate_tokens_pytorch(prompt_ids)

        audio_waveform = self._decode_tokens_to_audio(output_str)
        return audio_waveform

    def infer_to_stream(
        self,
        text: str,
        ref_codes: np.ndarray | torch.Tensor,
        ref_text: str,
        auto_split: bool = True,
    ) -> Generator[np.ndarray, None, None]:
        """
        Generate speech from text in real-time (streaming version).

        Instead of waiting for all audio to generate, this yields chunks
        as they're ready. Great for interactive applications!

        Currently only works with quantized (GGUF) models.

        Args:
            text: What you want the system to say
            ref_codes: Encoded reference audio (defines the voice style)
            ref_text: Transcript of the reference audio
            auto_split: If True, automatically chunk long text (default: True)

        Yields:
            Audio chunks as they're generated
        """
        if not self._is_quantized_model:
            raise NotImplementedError(
                "Streaming only available for quantized (GGUF) models"
            )

        # Check if text needs chunking
        if auto_split and self._should_split_text(text, ref_codes):
            yield from self._infer_stream_with_chunking(text, ref_codes, ref_text)
        else:
            # Single chunk streaming
            yield from self._generate_streaming_ggml(ref_codes, ref_text, text)

    async def infer_from_stream(
    self, 
    text_stream: AsyncGenerator[str], 
    ref_codes: np.ndarray | torch.Tensor, 
    ref_text: str
) -> AsyncGenerator[np.ndarray]:
        if not self._is_quantized_model:
            raise NotImplementedError(
                "Streaming only available for quantized (GGUF) models"
            )
        async for short_chunk in text_stream:
            # TODO: check if single chunk needs splitting
            if self._should_split_text(self.text_buffer + short_chunk, ref_codes):
                print("Text buffer full - generating chunk from:\n", self.text_buffer)
                long_chunks = self._group_sentences_into_chunks(self._split_text_into_sentences(self.text_buffer), ref_codes)
                long_chunk = long_chunks[0]
                print("Generated chunk:\n", long_chunk)
                for audio in self._generate_streaming_ggml(
                    ref_codes, ref_text, long_chunk
                ):
                    yield audio
                self.text_buffer = "".join(long_chunks[1:]) + short_chunk
            else:
                self.text_buffer = self.text_buffer + short_chunk
        if self.text_buffer:
            print ("Final text buffer - generating chunk from:\n", self.text_buffer)
            for audio in self._generate_streaming_ggml(
                ref_codes, ref_text, self.text_buffer
            ):
                yield audio
            self.text_buffer = ""


    def encode_reference(self, ref_audio_path: str | Path) -> np.ndarray:
        """
        Encode reference audio into tokens.

        This converts a reference audio file into the compact representation
        used by the model. The encoded audio captures the voice characteristics
        (pitch, tone, speaking style) that will be cloned.

        Args:
            ref_audio_path: Path to reference audio file (WAV, MP3, etc.)

        Returns:
            Encoded audio as array of integers (speech codes)
        """
        # Load audio file at 16kHz (codec's expected rate)
        audio_waveform, _ = librosa.load(ref_audio_path, sr=16000, mono=True)

        # Prepare tensor: [batch, channels, time]
        audio_tensor = (
            torch.from_numpy(audio_waveform)
            .float()
            .unsqueeze(0)  # Add batch dimension
            .unsqueeze(0)  # Add channel dimension
        )

        # Encode to discrete codes
        with torch.no_grad():
            ref_codes = (
                self.codec.encode_code(audio_or_path=audio_tensor)
                .squeeze(0)  # Remove batch
                .squeeze(0)  # Remove channel
            )

        return ref_codes

    # ========================================================================
    # TEXT CHUNKING FOR LONG INPUTS
    # ========================================================================

    def _should_split_text(
        self, text: str, ref_codes: np.ndarray | torch.Tensor
    ) -> bool:
        """
        Determine if text needs to be split due to context window limits.

        The real issue: We need to estimate OUTPUT tokens (speech codes),
        not just input tokens! The model generates ~68 speech tokens per
        second of audio, and text-to-speech ratio varies significantly.

        Args:
            text: Input text to check
            ref_codes: Reference audio codes

        Returns:
            True if text should be chunked
        """
        # The key insight: we need to estimate GENERATED speech tokens, not input tokens!
        # Speech generation uses far more tokens than the input text

        # Rough estimation: ~150 words per minute of speech
        # That's ~2.5 words per second
        word_count = len(text.split())
        estimated_speech_seconds = word_count / 2.5

        # Model generates ~68 speech tokens per second of audio
        estimated_speech_tokens = int(estimated_speech_seconds * self.tokens_per_second)

        # Add input tokens (text phonemes + reference codes + overhead)
        input_phonemes = int(word_count * 1.2)
        ref_tokens = len(ref_codes) if hasattr(ref_codes, "__len__") else 100
        total_input_tokens = input_phonemes + ref_tokens + self.prompt_overhead_tokens

        # Total context usage = input tokens + output speech tokens
        total_estimated = total_input_tokens + estimated_speech_tokens

        # Use a more aggressive safety margin since estimation is imperfect
        # Keep it under 70% of max context to be safe
        safe_limit = int(self.max_context * 0.7)

        return total_estimated > safe_limit

    def _split_text_into_sentences(self, text: str) -> list[str]:
        """
        Split text into sentences intelligently.

        Uses NLTK if available for smart splitting (handles abbreviations, etc.)
        Falls back to simple punctuation splitting otherwise.

        Args:
            text: Text to split

        Returns:
            List of sentences
        """
        if NLTK_AVAILABLE:
            # NLTK's sentence tokenizer is smart about abbreviations
            sentences = nltk.sent_tokenize(text)
        else:
            # Fallback: simple regex split on sentence-ending punctuation
            # This is less sophisticated but works reasonably well
            import re

            sentences = re.split(r"(?<=[.!?])\s+", text)
            sentences = [s.strip() for s in sentences if s.strip()]

        return sentences

    def _group_sentences_into_chunks(
        self, sentences: list[str], ref_codes: np.ndarray | torch.Tensor
    ) -> list[str]:
        """
        Group sentences into chunks that fit within context window.

        Tries to keep sentences together when possible for natural phrasing.
        Each chunk stays safely within the token limit.

        IMPORTANT: Must account for OUTPUT speech tokens, not just input!

        Args:
            sentences: List of individual sentences
            ref_codes: Reference codes (for token counting)

        Returns:
            List of text chunks, each safe to process
        """
        ref_tokens = len(ref_codes) if hasattr(ref_codes, "__len__") else 100

        # Calculate max tokens available per chunk
        # Use 60% of context for speech output tokens
        safe_limit = int(self.max_context * 0.6)

        chunks = []
        current_chunk = []
        current_words = 0

        for sentence in sentences:
            sentence_words = len(sentence.split())

            # Estimate speech tokens this sentence would generate
            # ~2.5 words per second, ~68 tokens per second
            estimated_speech_tokens = int(
                (sentence_words / 2.5) * self.tokens_per_second
            )

            # Check if adding this sentence would exceed limit
            test_words = current_words + sentence_words
            test_speech_tokens = int((test_words / 2.5) * self.tokens_per_second)
            test_input_tokens = (
                int(test_words * 1.2) + ref_tokens + self.prompt_overhead_tokens
            )
            test_total = test_speech_tokens + test_input_tokens

            if test_total > safe_limit and current_chunk:
                # Finalize current chunk
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_words = sentence_words
            elif estimated_speech_tokens > safe_limit:
                # Single sentence too long - need to split it
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_words = 0

                sub_chunks = self._split_long_sentence(sentence, safe_limit)
                chunks.extend(sub_chunks)
            else:
                # Add to current chunk
                current_chunk.append(sentence)
                current_words += sentence_words

        # Add final chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _split_long_sentence(self, sentence: str, max_tokens: int) -> list[str]:
        """
        Split an overly long sentence into smaller parts.

        Tries to split on natural boundaries (commas, semicolons, conjunctions)
        to maintain readability.

        Args:
            sentence: Long sentence to split
            max_tokens: Maximum OUTPUT speech tokens per part

        Returns:
            List of sentence fragments
        """
        # Convert max_tokens (speech tokens) to approximate word count
        # ~68 tokens per second, ~2.5 words per second
        # So roughly: max_tokens / 68 * 2.5 = max_words
        max_words = int(max_tokens / self.tokens_per_second * 2.5)

        # Try splitting on commas and semicolons first
        parts = re.split(r"([,;])", sentence)

        # Reconstruct with punctuation
        fragments = []
        current = ""
        for i, part in enumerate(parts):
            if part in [",", ";"]:
                current += part
            else:
                test = current + part
                test_words = len(test.split())

                if test_words > max_words and current:
                    fragments.append(current.strip())
                    current = part
                else:
                    current = test

        if current:
            fragments.append(current.strip())

        # If still too long, split by words as last resort
        final_fragments = []
        for fragment in fragments:
            fragment_words = len(fragment.split())
            if fragment_words > max_words:
                words = fragment.split()
                for i in range(0, len(words), max_words):
                    final_fragments.append(" ".join(words[i : i + max_words]))
            else:
                final_fragments.append(fragment)

        return final_fragments

    def _infer_with_chunking(
        self,
        text: str,
        ref_codes: np.ndarray | torch.Tensor,
        ref_text: str,
    ) -> np.ndarray:
        """
        Generate speech for long text by processing in chunks.

        This is the magic that enables unlimited-length generation!

        Process:
        1. Split text into sentence-aware chunks
        2. Generate audio for each chunk
        3. Join chunks with optional pauses or crossfades
        4. Return seamless combined audio

        Args:
            text: Long text to synthesize
            ref_codes: Reference audio codes
            ref_text: Reference text

        Returns:
            Combined audio waveform
        """
        print(f"Text is long - splitting into chunks for processing...")

        # Step 1: Split into sentences
        sentences = self._split_text_into_sentences(text)
        print(f"  Split into {len(sentences)} sentences")

        # Step 2: Group sentences into processable chunks
        chunks = self._group_sentences_into_chunks(sentences, ref_codes)
        print(f"  Grouped into {len(chunks)} chunks")

        # Step 3: Generate audio for each chunk
        audio_chunks = []
        for i, chunk in enumerate(chunks):
            print(f"  Processing chunk {i+1}/{len(chunks)}...")

            # Generate audio for this chunk
            if self._is_quantized_model:
                output_str = self._generate_tokens_ggml(ref_codes, ref_text, chunk)
            else:
                prompt_ids = self._build_prompt(ref_codes, ref_text, chunk)
                output_str = self._generate_tokens_pytorch(prompt_ids)

            audio = self._decode_tokens_to_audio(output_str)
            audio_chunks.append(audio)

        # Step 4: Join chunks
        print("  Joining chunks...")
        combined_audio = np.concatenate(audio_chunks)

        print("✓ Chunked generation complete!")
        return combined_audio

    def _crossfade_audio_chunks(
        self, audio_chunks: list[np.ndarray], fade_seconds: float
    ) -> np.ndarray:
        """
        Blend multiple audio chunks with smooth cross-fades.

        Args:
            audio_chunks: List of audio arrays to combine
            fade_seconds: Duration of cross-fade between chunks

        Returns:
            Crossfaded audio
        """
        if len(audio_chunks) == 1:
            return audio_chunks[0]

        # Calculate overlap in samples
        overlap_samples = int(fade_seconds * self.sample_rate)

        # Ensure overlap isn't longer than shortest chunk
        min_chunk_length = min(len(chunk) for chunk in audio_chunks)
        overlap_samples = min(overlap_samples, min_chunk_length // 2)

        # Build combined audio with cross-fades
        result = audio_chunks[0].copy()

        for i in range(1, len(audio_chunks)):
            current_chunk = audio_chunks[i]

            if overlap_samples > 0:
                # Create cross-fade weights (linear fade)
                fade_out = np.linspace(1.0, 0.0, overlap_samples)
                fade_in = np.linspace(0.0, 1.0, overlap_samples)

                # Apply cross-fade to overlapping region
                overlap_end = result[-overlap_samples:]
                overlap_start = current_chunk[:overlap_samples]

                blended_overlap = overlap_end * fade_out + overlap_start * fade_in

                # Combine: previous audio (minus overlap) + blended region + new audio
                result = np.concatenate(
                    [
                        result[:-overlap_samples],
                        blended_overlap,
                        current_chunk[overlap_samples:],
                    ]
                )
            else:
                # No overlap - simple concatenation
                result = np.concatenate([result, current_chunk])

        return result

    def _infer_stream_with_chunking(
        self, text: str, ref_codes: np.ndarray | torch.Tensor, ref_text: str
    ) -> Generator[np.ndarray, None, None]:
        """
        Stream audio for long text by processing in chunks.

        This combines chunking with streaming for the best of both worlds:
        - Handles unlimited text length
        - Returns audio progressively (no waiting for completion)

        Process:
        1. Split text into chunks
        2. Stream each chunk
        3. Add pauses or crossfades between chunks
        4. Yield audio as it's generated

        Args:
            text: Long text to synthesize
            ref_codes: Reference audio codes
            ref_text: Reference text

        Yields:
            Audio chunks as they're generated
        """
        print(f"Text is long - splitting into chunks for streaming...")

        # Step 1: Split into processable chunks
        sentences = self._split_text_into_sentences(text)
        chunks = self._group_sentences_into_chunks(sentences, ref_codes)
        print(f"  Processing {len(chunks)} chunks in streaming mode")

        # Step 2: Stream each chunk
        for chunk in chunks:
            yield from self._generate_streaming_ggml(ref_codes, ref_text, chunk)

    # ========================================================================
    # TEXT PROCESSING
    # ========================================================================

    def _text_to_phonemes(self, text: str) -> str:
        """
        Convert written text to phonetic representation.

        Example: "hello world" → "həˈloʊ wɜːld"

        This helps the model understand pronunciation, especially for:
        - Ambiguous words (read vs read)
        - Proper names
        - Made-up words

        Args:
            text: Written text

        Returns:
            Phonetic representation with spaces between phonemes
        """
        # preprocess text to better be compatible with phonemizer
        text = text.strip()
        text = text.replace('—', '-')
        text = text.replace('–', '-')
        text = ' '.join(text.split())
        if text and text[-1] not in '.!?;:':
            text += '.'
        print("Phonemizing text:", text)

        phoneme_list = self.phonemizer.phonemize([text])
        phonemes = phoneme_list[0].split()
        print("Phonemes:", phonemes)
        return " ".join(phonemes)

    # ========================================================================
    # TOKEN GENERATION (PyTorch Models)
    # ========================================================================

    def _build_prompt(
        self, ref_codes: list[int], ref_text: str, input_text: str
    ) -> list[int]:
        """
        Build the prompt for PyTorch models.

        The prompt structure tells the model:
        1. What text to convert (phonemes)
        2. What voice to use (reference codes)
        3. Where to start generating

        The format follows a chat-like structure:
        "user: Convert text to speech: [text]
         assistant: [reference codes]"

        Then the model continues generating speech codes.
        """
        # Convert both texts to phonemes
        combined_phonemes = (
            self._text_to_phonemes(ref_text) + " " + self._text_to_phonemes(input_text)
        )

        # Get special token IDs
        speech_replace = self.tokenizer.convert_tokens_to_ids("<|SPEECH_REPLACE|>")
        speech_gen_start = self.tokenizer.convert_tokens_to_ids(
            "<|SPEECH_GENERATION_START|>"
        )
        text_replace = self.tokenizer.convert_tokens_to_ids("<|TEXT_REPLACE|>")
        text_prompt_start = self.tokenizer.convert_tokens_to_ids(
            "<|TEXT_PROMPT_START|>"
        )
        text_prompt_end = self.tokenizer.convert_tokens_to_ids("<|TEXT_PROMPT_END|>")

        # Encode phonemes to token IDs
        phoneme_ids = self.tokenizer.encode(combined_phonemes, add_special_tokens=False)

        # Build chat template
        chat_template = (
            "user: Convert the text to speech:<|TEXT_REPLACE|>\n"
            "assistant:<|SPEECH_REPLACE|>"
        )
        template_ids = self.tokenizer.encode(chat_template)

        # Insert text prompt
        text_replace_idx = template_ids.index(text_replace)
        template_ids = (
            template_ids[:text_replace_idx]
            + [text_prompt_start]
            + phoneme_ids
            + [text_prompt_end]
            + template_ids[text_replace_idx + 1 :]
        )

        # Insert reference speech codes
        speech_replace_idx = template_ids.index(speech_replace)
        codes_str = "".join([f"<|speech_{i}|>" for i in ref_codes])
        codes = self.tokenizer.encode(codes_str, add_special_tokens=False)
        template_ids = (
            template_ids[:speech_replace_idx] + [speech_gen_start] + list(codes)
        )

        return template_ids

    def _generate_tokens_pytorch(self, prompt_ids: list[int]) -> str:
        """
        Generate speech tokens using PyTorch model.

        This runs the language model to predict speech tokens autoregressively
        (one token at a time, using previous tokens as context).
        """
        # Convert to tensor and move to model device
        prompt_tensor = (
            torch.tensor(prompt_ids)
            .unsqueeze(0)  # Add batch dimension
            .to(self.backbone.device)
        )

        # Get stop token
        speech_end_id = self.tokenizer.convert_tokens_to_ids(
            "<|SPEECH_GENERATION_END|>"
        )

        # Generate tokens
        with torch.no_grad():
            output_tokens = self.backbone.generate(
                prompt_tensor,
                max_length=self.max_context,
                eos_token_id=speech_end_id,  # Stop at this token
                do_sample=True,  # Use sampling (not greedy)
                temperature=1.0,  # Sampling randomness
                top_k=50,  # Only sample from top 50 tokens
                use_cache=True,  # Cache for speed
                min_new_tokens=50,  # Generate at least this many
            )

        # Decode only the newly generated tokens
        input_length = prompt_tensor.shape[-1]
        output_str = self.tokenizer.decode(
            output_tokens[0, input_length:].cpu().numpy().tolist(),
            add_special_tokens=False,
        )

        return output_str

    # ========================================================================
    # TOKEN GENERATION (GGUF/Quantized Models)
    # ========================================================================

    def _generate_tokens_ggml(
        self, ref_codes: list[int], ref_text: str, input_text: str
    ) -> str:
        """
        Generate speech tokens using quantized (GGUF) model.

        Similar to PyTorch version but uses llama-cpp interface.
        """
        # Convert texts to phonemes
        ref_phonemes = self._text_to_phonemes(ref_text)
        input_phonemes = self._text_to_phonemes(input_text)

        # Build prompt string
        codes_str = "".join([f"<|speech_{idx}|>" for idx in ref_codes])
        prompt = (
            f"user: Convert the text to speech:"
            f"<|TEXT_PROMPT_START|>{ref_phonemes} {input_phonemes}"
            f"<|TEXT_PROMPT_END|>\n"
            f"assistant:<|SPEECH_GENERATION_START|>{codes_str}"
        )

        # Generate
        output = self.backbone(
            prompt,
            max_tokens=self.max_context,
            temperature=1.0,
            top_k=50,
            stop=["<|SPEECH_GENERATION_END|>"],
        )

        return output["choices"][0]["text"]

    def _generate_streaming_ggml(
        self, ref_codes: np.ndarray | torch.Tensor, ref_text: str, input_text: str
    ) -> Generator[np.ndarray, None, None]:
        """
        Generate speech tokens in streaming mode (real-time).

        This is more complex because we need to:
        1. Generate tokens one at a time
        2. Decode audio chunks as we have enough tokens
        3. Yield audio progressively
        """
        # Convert texts to phonemes
        ref_phonemes = self._text_to_phonemes(ref_text)
        input_phonemes = self._text_to_phonemes(input_text)

        # Build prompt
        codes_str = "".join([f"<|speech_{idx}|>" for idx in ref_codes])
        prompt = (
            f"user: Convert the text to speech:"
            f"<|TEXT_PROMPT_START|>{ref_phonemes} {input_phonemes}"
            f"<|TEXT_PROMPT_END|>\n"
            f"assistant:<|SPEECH_GENERATION_START|>{codes_str}"
        )

        # Initialize caches
        audio_cache: list[np.ndarray] = []
        token_cache: list[str] = [f"<|speech_{idx}|>" for idx in ref_codes]
        n_decoded_samples = 0  # Track how many audio samples we've yielded
        n_decoded_tokens = len(ref_codes)  # Track how many tokens we've decoded

        # Stream tokens from model
        for item in self.backbone(
            prompt,
            max_tokens=self.max_context,
            temperature=1.0,
            top_k=50,
            stop=["<|SPEECH_GENERATION_END|>"],
            stream=True,  # Enable streaming mode
        ):
            output_token = item["choices"][0]["text"]
            token_cache.append(output_token)

            # Check if we have enough tokens to decode a chunk
            new_tokens = len(token_cache[n_decoded_tokens:])
            min_tokens_needed = (
                self.streaming_frames_per_chunk + self.streaming_lookforward
            )

            if new_tokens >= min_tokens_needed:
                # Decode chunk with context for better quality
                tokens_start = max(
                    n_decoded_tokens
                    - self.streaming_lookback
                    - self.streaming_overlap_frames,
                    0,
                )
                tokens_end = (
                    n_decoded_tokens
                    + self.streaming_frames_per_chunk
                    + self.streaming_lookforward
                    + self.streaming_overlap_frames
                )

                # Calculate which part of decoded audio is new
                sample_start = (n_decoded_tokens - tokens_start) * self.hop_length
                sample_end = (
                    sample_start
                    + (
                        self.streaming_frames_per_chunk
                        + 2 * self.streaming_overlap_frames
                    )
                    * self.hop_length
                )

                # Decode chunk
                curr_codes = token_cache[tokens_start:tokens_end]
                audio_chunk = self._decode_tokens_to_audio("".join(curr_codes))
                audio_chunk = audio_chunk[sample_start:sample_end]
                audio_cache.append(audio_chunk)

                # Blend overlapping chunks
                blended_audio = smooth_audio_overlap(
                    audio_cache, stride=self.streaming_stride_samples
                )

                # Yield only new samples
                new_samples_end = len(audio_cache) * self.streaming_stride_samples
                new_audio = blended_audio[n_decoded_samples:new_samples_end]
                n_decoded_samples = new_samples_end
                n_decoded_tokens += self.streaming_frames_per_chunk

                yield new_audio

        # Handle remaining tokens (last chunk may be smaller)
        remaining_tokens = len(token_cache) - n_decoded_tokens
        if remaining_tokens > 0:
            tokens_start = max(
                len(token_cache)
                - (
                    self.streaming_lookback
                    + self.streaming_overlap_frames
                    + remaining_tokens
                ),
                0,
            )
            sample_start = (
                len(token_cache)
                - tokens_start
                - remaining_tokens
                - self.streaming_overlap_frames
            ) * self.hop_length

            # Decode final chunk
            curr_codes = token_cache[tokens_start:]
            audio_chunk = self._decode_tokens_to_audio("".join(curr_codes))
            audio_chunk = audio_chunk[sample_start:]
            audio_cache.append(audio_chunk)

            # Final blending
            blended_audio = smooth_audio_overlap(
                audio_cache, stride=self.streaming_stride_samples
            )
            final_audio = blended_audio[n_decoded_samples:]

            yield final_audio

    # ========================================================================
    # AUDIO DECODING
    # ========================================================================

    def _decode_tokens_to_audio(self, token_string: str) -> np.ndarray:
        """
        Convert speech tokens to audio waveform.

        The model generates tokens like "<|speech_42|><|speech_17|>..."
        This extracts the numbers and feeds them to the codec to get audio.

        Args:
            token_string: String containing speech tokens

        Returns:
            Audio waveform as 1D numpy array

        Raises:
            ValueError: If no valid speech tokens found
        """
        # Extract token IDs using regex
        # Matches patterns like "<|speech_123|>" and extracts 123
        speech_ids = [
            int(num) for num in re.findall(r"<\|speech_(\d+)\|>", token_string)
        ]

        if len(speech_ids) == 0:
            raise ValueError("No valid speech tokens found in output")

        # Decode using appropriate codec
        if self._is_onnx_codec:
            # ONNX codec expects numpy array: [batch, codebooks, time]
            codes = np.array(speech_ids, dtype=np.int32)[np.newaxis, np.newaxis, :]
            audio = self.codec.decode_code(codes)
        else:
            # PyTorch codec expects tensor: [batch, codebooks, time]
            with torch.no_grad():
                codes = torch.tensor(speech_ids, dtype=torch.long)[
                    None, None, :
                ].to(  # Add batch and codebook dimensions
                    self.codec.device
                )
                audio = self.codec.decode_code(codes).cpu().numpy()

        # Return as 1D array (remove batch and channel dimensions)
        return audio[0, 0, :]
