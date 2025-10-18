import argparse
import asyncio
import logging
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, Iterable, List, Optional, AsyncGenerator

import numpy as np
import torch

from neuttsair.neutts import NeuTTSAir
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.error import Error
from wyoming.event import Event
from wyoming.info import Attribution, Describe, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncEventHandler, AsyncServer
from wyoming.tts import Synthesize, SynthesizeVoice, SynthesizeStart, SynthesizeChunk, SynthesizeStop

LOGGER = logging.getLogger(__name__)


@dataclass
class VoiceProfile:
    """Represents a cloned voice that can be served over Wyoming."""

    name: str
    ref_codes: np.ndarray
    ref_text: str
    description: Optional[str] = None
    language: str = "en"

    def to_tts_voice(self, attribution: Attribution) -> TtsVoice:
        return TtsVoice(
            name=self.name,
            description=self.description or f"NeuTTSAir voice '{self.name}'",
            attribution=attribution,
            installed=True,
            version=None,
            languages=[self.language],
        )


class NeuTTSAirEventHandler(AsyncEventHandler):
    """Handles incoming Wyoming events for NeuTTSAir."""

    def __init__(
        self,
        tts: NeuTTSAir,
        voice_map: Dict[str, VoiceProfile],
        default_voice: str,
        info_event: Event,
        chunk_size: int,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._tts = tts
        self._voices = voice_map
        self._default_voice = default_voice
        self._info_event = info_event
        self._chunk_size = max(1, chunk_size)
        self._sample_rate = self._tts.sample_rate

        # Streaming state
        self._stream_voice = None
        self._stream_text_queue = None
        self._stream_task = None

    async def handle_event(self, event: Event) -> bool:
        if event is None:
            return False

        # Handle info request
        if event.type == "describe":
            await self.write_event(self._info_event)
            LOGGER.debug("Provided Wyoming info to client")
            return True

        # Handle regular synthesis
        if event.type == "synthesize":
            synthesize = Synthesize.from_event(event)
            voice = self._select_voice(synthesize.voice)
            if voice is None:
                await self.write_event(
                    Error(
                        text="Requested voice not available",
                        code="voice-not-found",
                    ).event()
                )
                return True

            LOGGER.info("Synthesizing text with voice '%s'", voice.name)
            await self._synthesize_and_stream(synthesize.text, voice)
            return True

        # Handle streaming synthesis start
        if event.type == "synthesize-start":
            await self._handle_synthesize_start(event)
            return True

        # Handle streaming text chunks
        if event.type == "synthesize-chunk":
            await self._handle_synthesize_chunk(event)
            return True

        # Handle streaming synthesis stop
        if event.type == "synthesize-stop":
            await self._handle_synthesize_stop()
            return True

        LOGGER.debug("Ignoring unsupported event type: %s", event.type)
        return True

    async def _handle_synthesize_start(self, event: Event) -> None:
        """Initialize streaming synthesis session."""
        if self._stream_text_queue:
            await self.write_event(
                Error(
                    text="Stream text queue is already initialized",
                    code="streaming-active",
                ).event()
            )
            return

        synthesize_start = SynthesizeStart.from_event(event)
        voice = self._select_voice(synthesize_start.voice)
        
        if voice is None:
            await self.write_event(
                Error(
                    text="Requested voice not available",
                    code="voice-not-found",
                ).event()
            )
            return

        self._stream_voice = voice
        self._stream_text_queue = asyncio.Queue()
        
        # Start synthesis in background task
        self._stream_task = asyncio.create_task(
            self._run_streaming_synthesis(voice)
        )
        
        LOGGER.info("Started streaming synthesis with voice '%s'", voice.name)

    async def _run_streaming_synthesis(self, voice: VoiceProfile) -> None:
        """Background task that runs the streaming synthesis."""
        try:
            # Send audio start event
            await self.write_event(
                AudioStart(rate=self._sample_rate, width=2, channels=1).event()
            )
            LOGGER.info("Sent AudioStart event")
            
            # Process the text stream and generate audio
            async for audio_chunk in self._tts.infer_from_stream(
                self._get_text_stream(),
                voice.ref_codes,
                voice.ref_text,
            ):
                audio_bytes = _to_pcm16_bytes(audio_chunk)
                await self.write_event(
                    AudioChunk(
                        rate=self._sample_rate,
                        width=2,
                        channels=1,
                        audio=audio_bytes,
                    ).event()
                )
            
            # Send audio stop event
            await self.write_event(AudioStop().event())
            LOGGER.info("Streaming synthesis complete")
            
        except Exception as err:
            LOGGER.exception("Error in streaming synthesis")
            await self.write_event(
                Error(
                    text=str(err),
                    code=type(err).__name__,
                ).event()
            )

    async def _handle_synthesize_chunk(self, event: Event) -> None:
        """Accumulate text chunks for streaming synthesis."""
        if not self._stream_text_queue:
            await self.write_event(
                Error(
                    text="Stream text queue has not been initialized",
                    code="streaming-not-started",
                ).event()
            )
            return

        synthesize_chunk = SynthesizeChunk.from_event(event)
        await self._stream_text_queue.put(synthesize_chunk.text)

    async def _handle_synthesize_stop(self) -> None:
        """Finalize and synthesize accumulated text."""
        if not self._stream_text_queue:
            await self.write_event(
                Error(
                    text="Stream text queue has not been initialized",
                    code="streaming-not-started",
                ).event()
            )
            return

        # Signal end of text stream
        await self._stream_text_queue.put(None)
        
        # Wait for synthesis task to complete
        if self._stream_task:
            try:
                await self._stream_task
            except Exception as err:
                LOGGER.exception("Error waiting for synthesis task")

        # Reset streaming state
        self._stream_voice = None
        self._stream_text_queue = None
        self._stream_task = None
        LOGGER.info("Stopped streaming synthesis")

    async def _get_text_stream(self) -> AsyncGenerator[str]:
        """Asynchronously yield text chunks from the queue."""
        try:
            while True:
                if self._stream_text_queue is None:
                    LOGGER.warning("Text stream queue is None, ending stream")
                    break
                text_chunk = await self._stream_text_queue.get()
                if text_chunk is None:
                    LOGGER.info("Text stream ended (received None)")
                    break
                yield text_chunk
        except asyncio.CancelledError:
            LOGGER.info("Text stream cancelled")
            raise

    async def _synthesize_and_stream(self, text: str, voice) -> None:
        """Synthesize text and stream audio back to client."""
        try:
            # Check if streaming inference is available
            if hasattr(self._tts, 'infer_to_stream'):
                await self._synthesize_streaming(text, voice)
            else:
                await self._synthesize_non_streaming(text, voice)
        except Exception as err:
            LOGGER.exception("Failed to synthesize audio")
            await self.write_event(
                Error(
                    text=str(err),
                    code=type(err).__name__,
                ).event()
            )

    async def _synthesize_streaming(self, text: str, voice) -> None:
        """Use streaming inference for real-time audio generation."""
        LOGGER.info("Using streaming synthesis")
        
        # Send audio start event
        await self.write_event(
            AudioStart(rate=self._sample_rate, width=2, channels=1).event()
        )

        # Stream audio chunks as they're generated
        for audio_chunk in self._tts.infer_to_stream(text, voice.ref_codes, voice.ref_text):
            audio_bytes = _to_pcm16_bytes(audio_chunk)
            
            await self.write_event(
                AudioChunk(
                    rate=self._sample_rate,
                    width=2,
                    channels=1,
                    audio=audio_bytes,
                ).event()
            )
            LOGGER.debug("Streamed audio chunk of %d bytes", len(audio_bytes))

        await self.write_event(AudioStop().event())
        LOGGER.info("Streaming synthesis complete")

    async def _synthesize_non_streaming(self, text: str, voice) -> None:
        """Fallback to non-streaming synthesis."""
        LOGGER.info("Using non-streaming synthesis")
        
        wav = await asyncio.to_thread(
            self._tts.infer,
            text,
            voice.ref_codes,
            voice.ref_text,
        )
        
        audio_bytes = _to_pcm16_bytes(wav)
        bytes_per_sample = 2
        chunk_bytes = self._chunk_size * bytes_per_sample

        await self.write_event(
            AudioStart(rate=self._sample_rate, width=2, channels=1).event()
        )

        for i in range(0, len(audio_bytes), chunk_bytes):
            chunk = audio_bytes[i:i + chunk_bytes]
            await self.write_event(
                AudioChunk(
                    rate=self._sample_rate,
                    width=2,
                    channels=1,
                    audio=chunk,
                ).event()
            )
            LOGGER.debug("Streaming audio chunk of %d bytes", len(chunk))

        await self.write_event(AudioStop().event())

    def _select_voice(self, request_voice: Optional[SynthesizeVoice]) -> Optional[VoiceProfile]:
        if request_voice is not None:
            if request_voice.name and request_voice.name in self._voices:
                return self._voices[request_voice.name]

            if request_voice.language:
                for voice in self._voices.values():
                    if voice.language == request_voice.language:
                        return voice

        return self._voices.get(self._default_voice)


def _to_pcm16_bytes(audio: np.ndarray | torch.Tensor) -> bytes:
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()

    if not isinstance(audio, np.ndarray):
        audio = np.asarray(audio)

    audio = audio.flatten()
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * np.iinfo(np.int16).max).astype("<i2")
    return pcm.tobytes()


def _parse_voice_arg(raw: str) -> Dict[str, str]:
    parts = {}
    for section in raw.split(","):
        section = section.strip()
        if not section:
            continue
        if "=" not in section:
            message = (
                "Voice definition '{raw}' must use key=value pairs separated by commas"
            ).format(raw=raw)
            raise ValueError(message)
        key, value = section.split("=", 1)
        parts[key.strip()] = value.strip()

    required = {"name", "ref_text"}
    if not required.issubset(parts):
        missing = ", ".join(sorted(required - set(parts)))
        message = f"Voice definition missing required fields: {missing}"
        raise ValueError(message)

    if ("ref_audio" not in parts) and ("ref_codes" not in parts):
        message = "Voice definition must include either 'ref_audio' or 'ref_codes'"
        raise ValueError(message)

    return parts


def _flatten_codes(
    codes: np.ndarray | torch.Tensor | Iterable[int],
) -> np.ndarray:
    if isinstance(codes, torch.Tensor):
        codes = codes.detach().cpu().view(-1).tolist()
    elif isinstance(codes, np.ndarray):
        codes = codes.reshape(-1).tolist()
    else:
        codes = list(codes)

    return np.array([int(code) for code in codes], dtype=np.int32)


def _load_ref_codes(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix in {".pt", ".pth"}:
        data = torch.load(path, map_location="cpu")
    elif suffix == ".npy":
        data = np.load(path)
    else:
        message = (
            "Unsupported reference code format '{suffix}'. Expected .pt, .pth, or .npy"
        ).format(suffix=path.suffix)
        raise ValueError(message)

    return _flatten_codes(data)


def _load_voice_profiles(
    tts: NeuTTSAir,
    voice_args: List[str],
) -> List[VoiceProfile]:
    voices: List[VoiceProfile] = []

    for raw_voice in voice_args:
        config = _parse_voice_arg(raw_voice)
        name = config["name"]

        ref_text_value = config["ref_text"]
        ref_text_path = Path(ref_text_value)
        if ref_text_path.is_file():
            text_data = ref_text_path.read_text(encoding="utf-8")
            ref_text = text_data.strip()
        else:
            ref_text = ref_text_value

        if "ref_codes" in config:
            ref_codes_path = Path(config["ref_codes"])
            if not ref_codes_path.is_file():
                message = f"Reference codes not found: {ref_codes_path}"
                raise FileNotFoundError(message)

            ref_codes = _load_ref_codes(ref_codes_path)
        else:
            ref_audio_path = Path(config["ref_audio"])
            if not ref_audio_path.is_file():
                message = f"Reference audio not found: {ref_audio_path}"
                raise FileNotFoundError(message)

            audio_path = str(ref_audio_path)
            ref_codes_tensor = tts.encode_reference(audio_path)
            ref_codes = _flatten_codes(ref_codes_tensor)

        description = config.get("description")
        language = config.get("language", "en")

        voices.append(
            VoiceProfile(
                name=name,
                ref_codes=ref_codes,
                ref_text=ref_text,
                description=description,
                language=language,
            )
        )

        LOGGER.info("Loaded voice '%s'", name)

    return voices


async def _run_server(args: argparse.Namespace) -> None:
    tts = NeuTTSAir(
        backbone_repo=args.backbone,
        backbone_device=args.backbone_device,
        codec_repo=args.codec,
        codec_device=args.codec_device,
    )

    voices = _load_voice_profiles(tts, args.voice)
    if not voices:
        raise ValueError("At least one voice definition is required")

    voice_map = {voice.name: voice for voice in voices}
    default_voice = voices[0].name

    program_attribution = Attribution(
        name="Neuphonic",
        url="https://neuphonic.com/",
    )
    tts_program_voices: List[TtsVoice] = []
    for voice in voices:
        program_voice = voice.to_tts_voice(program_attribution)
        tts_program_voices.append(program_voice)

    wyoming_info = Info(
        tts=[
            TtsProgram(
                name="neutts-air",
                description="NeuTTSAir neural text to speech",
                attribution=program_attribution,
                installed=True,
                voices=tts_program_voices,
                version=None,
                supports_synthesize_streaming=True,
            )
        ]
    )

    server = AsyncServer.from_uri(args.uri)
    log_message = "NeuTTSAir Wyoming server listening on %s"
    LOGGER.info(log_message, args.uri)
    await server.run(
        partial(
            NeuTTSAirEventHandler,
            tts,
            voice_map,
            default_voice,
            wyoming_info.event(),
            args.chunk_size,
        )
    )


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Expose NeuTTSAir as a Wyoming TTS service")
    parser.add_argument(
        "--uri",
        default="tcp://0.0.0.0:10200",
        help="Wyoming URI",
    )
    parser.add_argument(
        "--backbone",
        default="neuphonic/neutts-air",
        help="Backbone model repository or GGUF path",
    )
    parser.add_argument(
        "--backbone-device",
        default="cpu",
        help="Device for the backbone model (e.g. cpu, cuda)",
    )
    parser.add_argument(
        "--codec",
        default="neuphonic/neucodec",
        help="Codec repository identifier",
    )
    parser.add_argument(
        "--codec-device",
        default="cpu",
        help="Device for the codec (cpu, cuda)",
    )
    parser.add_argument(
        "--voice",
        action="append",
        required=True,
        help=(
            "Voice definition as comma separated key=value pairs. "
            "Required keys: name, ref_text, and one of ref_audio/ref_codes. "
            "Optional keys: description, language."
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2400,
        help="Number of samples per Wyoming audio chunk",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging output")
    return parser


def main() -> None:
    parser = _build_argument_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    try:
        asyncio.run(_run_server(args))
    except KeyboardInterrupt:
        LOGGER.info("Shutting down")


if __name__ == "__main__":
    main()