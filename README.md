# last


# Jmj


import asyncio
from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketState
from contextlib import asynccontextmanager
import io
from scipy.io.wavfile import write
from scipy import signal
from groq import Groq
import json
import os
from dotenv import load_dotenv
from gtts import gTTS
import numpy as np
from pydub import AudioSegment
import sounddevice as sd
import numpy as np
import threading
import queue
import base64
import wave
import subprocess
import time
import atexit
from pathlib import Path
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. VAD will use simple energy-based detection.")

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    raise ValueError("Missing GROQ_API_KEY in .env")

ROOT_DIR = Path(__file__).resolve().parent
ASOUNDRC_SCRIPT_PATH = ROOT_DIR / "scripts" / "configure_asoundrc.sh"

VIRTUAL_SINK_NAME = "virtual_sink"
VIRTUAL_TTS_NAME = "virtual_tts"
VIRTUAL_SOURCE_NAME = "virtual_source"
VIRTUAL_CAPTURE_DEVICE = "virtual_sink_monitor"
PULSE_CAPTURE_MONITOR = f"{VIRTUAL_SINK_NAME}.monitor"
DEFAULT_TTS_ALSA_DEVICE = VIRTUAL_TTS_NAME
CAPTURE_DEVICE_CANDIDATES = [
    dev for dev in [VIRTUAL_CAPTURE_DEVICE, PULSE_CAPTURE_MONITOR, f"{VIRTUAL_SINK_NAME}_monitor"] if dev
]

# Store original audio settings and module IDs for restoration
original_audio_settings = {
    "default_sink": None,
    "loopback_module_id": None,
    "virtual_sink_module_id": None,
    "virtual_tts_module_id": None,
    "virtual_source_module_id": None
}


def run_pactl(args: list[str], *, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(["pactl", *args], capture_output=True, text=True, check=check)


def find_module_id(keyword: str) -> str | None:
    """Find a PulseAudio module ID that contains the provided keyword."""
    try:
        result = run_pactl(["list", "short", "modules"], check=True)
        for line in result.stdout.splitlines():
            if keyword in line:
                return line.split()[0]
    except Exception as e:
        print(f"Failed to query modules for '{keyword}': {e}")
    return None


def unload_modules(keyword: str):
    """Unload all PulseAudio modules containing keyword."""
    try:
        result = run_pactl(["list", "short", "modules"], check=True)
        for line in result.stdout.splitlines():
            if keyword in line:
                module_id = line.split()[0]
                run_pactl(["unload-module", module_id])
                print(f"Unloaded module {module_id} matching '{keyword}'")
    except Exception as e:
        print(f"Unable to unload modules for '{keyword}': {e}")


ASOUNDRC_FALLBACK_BLOCK = """
# Dhvani virtual audio devices (fallback)
pcm.virtual_tts {
    type pulse
    device "virtual_tts"
}

ctl.virtual_tts {
    type pulse
    device "virtual_tts"
}

pcm.virtual_sink_monitor {
    type pulse
    device "virtual_sink.monitor"
}

ctl.virtual_sink_monitor {
    type pulse
    device "virtual_sink.monitor"
}
# End Dhvani virtual audio devices (fallback)
""".strip() + "\n"


def ensure_asoundrc_configuration():
    """Ensure ~/.asoundrc exposes the virtual PulseAudio devices to ALSA."""
    if ASOUNDRC_SCRIPT_PATH.exists():
        try:
            subprocess.run(["/bin/bash", str(ASOUNDRC_SCRIPT_PATH)], check=True)
            return
        except Exception as exc:
            print(f"Warning: {ASOUNDRC_SCRIPT_PATH} failed: {exc}. Falling back to inline update.")
    try:
        asoundrc_path = Path.home() / ".asoundrc"
        marker = "# Dhvani virtual audio devices (fallback)"
        if asoundrc_path.exists():
            content = asoundrc_path.read_text()
            if marker in content:
                return
            with asoundrc_path.open("a") as f:
                if not content.endswith("\n"):
                    f.write("\n")
                f.write("\n" + ASOUNDRC_FALLBACK_BLOCK)
        else:
            asoundrc_path.write_text(ASOUNDRC_FALLBACK_BLOCK)
        print(f"Updated ALSA configuration at {asoundrc_path}")
    except Exception as e:
        print(f"Warning: Unable to update ~/.asoundrc for virtual devices: {e}")


def setup_audio_routing():
    """Set up PulseAudio routing for Teams integration."""
    print("Setting up audio routing...")

    try:
        subprocess.run(["pulseaudio", "--check"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("WARNING: PulseAudio not running. Audio routing may not work.")
        return

    try:
        ensure_asoundrc_configuration()

        result = run_pactl(["info"], check=True)
        for line in result.stdout.splitlines():
            if "Default Sink:" in line:
                original_audio_settings["default_sink"] = line.split(":")[1].strip()
                break

        unload_modules(f"sink_name={VIRTUAL_SINK_NAME}")
        result = run_pactl(
            [
                "load-module",
                "module-null-sink",
                f"sink_name={VIRTUAL_SINK_NAME}",
                f"sink_properties=device.description={VIRTUAL_SINK_NAME}",
            ]
        )
        if result.returncode == 0 and result.stdout.strip():
            original_audio_settings["virtual_sink_module_id"] = result.stdout.strip()
        else:
            original_audio_settings["virtual_sink_module_id"] = find_module_id(VIRTUAL_SINK_NAME)
        print(f"{VIRTUAL_SINK_NAME} module ID: {original_audio_settings['virtual_sink_module_id']}")

        unload_modules(f"sink_name={VIRTUAL_TTS_NAME}")
        result = run_pactl(
            [
                "load-module",
                "module-null-sink",
                f"sink_name={VIRTUAL_TTS_NAME}",
                f"sink_properties=device.description={VIRTUAL_TTS_NAME}",
            ]
        )
        if result.returncode == 0 and result.stdout.strip():
            original_audio_settings["virtual_tts_module_id"] = result.stdout.strip()
        else:
            original_audio_settings["virtual_tts_module_id"] = find_module_id(VIRTUAL_TTS_NAME)
        print(f"{VIRTUAL_TTS_NAME} module ID: {original_audio_settings['virtual_tts_module_id']}")

        unload_modules(f"source_name={VIRTUAL_SOURCE_NAME}")
        result = run_pactl(
            [
                "load-module",
                "module-remap-source",
                f"source_name={VIRTUAL_SOURCE_NAME}",
                f"master={VIRTUAL_TTS_NAME}.monitor",
                f"source_properties=device.description={VIRTUAL_SOURCE_NAME}",
            ]
        )
        if result.returncode == 0 and result.stdout.strip():
            original_audio_settings["virtual_source_module_id"] = result.stdout.strip()
        else:
            original_audio_settings["virtual_source_module_id"] = find_module_id(VIRTUAL_SOURCE_NAME)
        print(f"{VIRTUAL_SOURCE_NAME} module ID: {original_audio_settings['virtual_source_module_id']}")

        result = run_pactl(["list", "short", "sinks"], check=True)
        physical_speaker = None
        for line in result.stdout.splitlines():
            if VIRTUAL_SINK_NAME not in line and "RUNNING" in line:
                physical_speaker = line.split()[1]
                break
        if not physical_speaker:
            for line in result.stdout.splitlines():
                if VIRTUAL_SINK_NAME not in line and line.strip():
                    physical_speaker = line.split()[1]
                    break

        if physical_speaker:
            result = run_pactl(["list", "short", "modules"], check=True)
            for line in result.stdout.splitlines():
                if "module-loopback" in line and f"source={VIRTUAL_SINK_NAME}.monitor" in line:
                    module_id = line.split()[0]
                    run_pactl(["unload-module", module_id])

            result = run_pactl(
                [
                    "load-module",
                    "module-loopback",
                    f"source={VIRTUAL_SINK_NAME}.monitor",
                    f"sink={physical_speaker}",
                    "latency_msec=1",
                ]
            )
            if result.returncode == 0 and result.stdout.strip():
                original_audio_settings["loopback_module_id"] = result.stdout.strip()
            else:
                original_audio_settings["loopback_module_id"] = find_module_id(
                    f"source={VIRTUAL_SINK_NAME}.monitor"
                )
            print(f"Loopback module ID: {original_audio_settings['loopback_module_id']}")
        else:
            print("WARNING: Could not determine physical speaker sink. Teams audio may not be audible.")

        print("Audio routing setup complete!")
        print(f"  - {VIRTUAL_SINK_NAME}: Teams output sink with loopback")
        print(f"  - {VIRTUAL_TTS_NAME}: TTS injection sink")
        print(f"  - {VIRTUAL_SOURCE_NAME}: Teams microphone mapped to {VIRTUAL_TTS_NAME}.monitor")

    except Exception as e:
        print(f"Error setting up audio routing: {e}")
        import traceback

        traceback.print_exc()


def restore_audio_settings():
    """Restore original audio settings"""
    print("Restoring original audio settings...")
    
    try:
        # Unload loopback module
        if original_audio_settings["loopback_module_id"]:
            run_pactl(["unload-module", original_audio_settings["loopback_module_id"]])
            print(f"Removed loopback (module ID: {original_audio_settings['loopback_module_id']})")
            original_audio_settings["loopback_module_id"] = None
        
        # Unload remap source module
        if original_audio_settings["virtual_source_module_id"]:
            run_pactl(["unload-module", original_audio_settings["virtual_source_module_id"]])
            print(f"Removed {VIRTUAL_SOURCE_NAME} (module ID: {original_audio_settings['virtual_source_module_id']})")
            original_audio_settings["virtual_source_module_id"] = None
        
        # Unload virtual_tts sink
        if original_audio_settings["virtual_tts_module_id"]:
            run_pactl(["unload-module", original_audio_settings["virtual_tts_module_id"]])
            print(f"Removed {VIRTUAL_TTS_NAME} (module ID: {original_audio_settings['virtual_tts_module_id']})")
            original_audio_settings["virtual_tts_module_id"] = None
        
        # Unload virtual_sink
        if original_audio_settings["virtual_sink_module_id"]:
            run_pactl(["unload-module", original_audio_settings["virtual_sink_module_id"]])
            print(f"Removed {VIRTUAL_SINK_NAME} (module ID: {original_audio_settings['virtual_sink_module_id']})")
            original_audio_settings["virtual_sink_module_id"] = None
        
        # Restore original default sink if it was changed
        if original_audio_settings["default_sink"]:
            run_pactl(["set-default-sink", original_audio_settings["default_sink"]])
            print(f"Restored default sink to: {original_audio_settings['default_sink']}")
        
        print("Audio settings restored!")
        
    except Exception as e:
        print(f"Error restoring audio settings: {e}")
        import traceback
        traceback.print_exc()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    setup_audio_routing()
    # Start client transcript service automatically - DISABLED (User controlled now)
    # ensure_client_transcript_service()
    yield
    # Shutdown
    if client_transcript_service:
        client_transcript_service.stop()
    restore_audio_settings()


app = FastAPI(
    title="Dhvani API",
    description="Japanese-English Translation API with TTS",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global variables for virtual audio capture
virtual_capture_active = False
virtual_capture_queue = queue.Queue()
virtual_capture_thread = None
virtual_audio_buffer = []
websocket_clients = []  
websocket_lock = threading.Lock()

# Client transcript streaming state
client_transcript_subscribers = []
client_transcript_lock = threading.Lock()
client_transcript_service = None
client_transcript_service_lock = threading.Lock()

# VAD state
silero_vad_model = None
silero_get_speech_ts = None
silero_vad_lock = threading.Lock()

# TTS concurrency guard (serialize server-side TTS playback)
tts_semaphore = asyncio.Semaphore(1)


def resolve_capture_device():
    """
    Determine capture device preference order and return (device_index, device_name).
    Tries configured ALSA alias first, then falls back to the Pulse monitor name.
    """
    for candidate in CAPTURE_DEVICE_CANDIDATES:
        idx = get_device_index(candidate)
        if idx is not None:
            print(f"Using capture device '{candidate}' (index {idx})")
            return idx, candidate

    if CAPTURE_DEVICE_CANDIDATES:
        print(
            f"Capture devices {CAPTURE_DEVICE_CANDIDATES} not detected; "
            f"attempting to open '{CAPTURE_DEVICE_CANDIDATES[0]}' directly."
        )
        return None, CAPTURE_DEVICE_CANDIDATES[0]

    raise RuntimeError("No capture devices configured")


def load_silero_vad():
    """
    Load the Silero VAD model once and reuse it for all requests.
    
    Silero VAD is an industry-standard neural network-based VAD used by companies
    like NVIDIA. It's required to prevent false positives (hallucinated audio).
    """
    global silero_vad_model, silero_get_speech_ts
    if not TORCH_AVAILABLE:
        return None, None
    with silero_vad_lock:
        if silero_vad_model is None or silero_get_speech_ts is None:
            try:
                print("Loading Silero VAD model (industry-standard, prevents false positives)...")
                model, utils = torch.hub.load(
                    repo_or_dir="snakers4/silero-vad",
                    model="silero_vad",
                    force_reload=False,
                    trust_repo=True,
                )
                (get_speech_ts, *_rest) = utils
                silero_vad_model = model.to(torch.device("cpu"))
                silero_vad_model.eval()
                silero_get_speech_ts = get_speech_ts
                print("✓ Silero VAD model loaded successfully")
            except Exception as exc:
                print(f"ERROR: Failed to load Silero VAD model: {exc}")
                print("  This is required to prevent false positives (hallucinated audio).")
                print("  Please ensure you have internet connection and PyTorch installed.")
                import traceback
                traceback.print_exc()
                return None, None
    return silero_vad_model, silero_get_speech_ts


class SileroRealtimeSentenceProcessor:
    """
    Sentence segmenter powered by Silero VAD for streaming audio.
    
    Silero VAD is an industry-standard, neural network-based VAD used by companies like NVIDIA.
    It's trained on 6,000+ languages and significantly outperforms energy-based methods,
    reducing false positives (hallucinated audio) by using deep learning to distinguish
    speech from noise, music, and other non-speech audio.
    
    This implementation requires Silero VAD to function - no fallback to prevent hallucinations.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        silence_duration: float = 0.8,  # Optimized: 800ms silence for better sentence boundaries
        min_speech_duration: float = 0.25,  # Optimized: 250ms minimum speech (Silero recommended)
        pre_roll_seconds: float = 0.3,  # Optimized: 300ms pre-roll to capture speech onset
        speech_pad_ms: int = 30,  # Optimized: 30ms padding (Silero default, prevents clipping)
    ):
        self.sample_rate = sample_rate
        self.guard_samples = max(1, int(silence_duration * sample_rate))
        self.min_speech_duration = max(0.15, min_speech_duration)
        self.min_sentence_samples = max(1, int(self.sample_rate * self.min_speech_duration))
        self.min_speech_ms = int(self.min_speech_duration * 1000)
        self.pre_roll_samples = max(0, int(pre_roll_seconds * sample_rate))
        self.speech_pad_ms = max(0, speech_pad_ms)
        self.buffer = np.array([], dtype=np.float32)
        
        # Silero VAD is REQUIRED - no fallback to prevent hallucinations
        if not TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch is required for Silero VAD. "
                "Install with: pip install torch"
            )
        
        self.model, self.get_speech_ts = load_silero_vad()
        if self.model is None or self.get_speech_ts is None:
            raise RuntimeError(
                "Silero VAD model failed to load. "
                "This is required to prevent false positives (hallucinated audio). "
                "Please check your internet connection and PyTorch installation."
            )
        
        print(f"✓ Silero VAD initialized successfully:")
        print(f"  - Sample rate: {sample_rate} Hz")
        print(f"  - Silence duration: {silence_duration}s (sentence boundary)")
        print(f"  - Min speech duration: {min_speech_duration}s")
        print(f"  - Pre-roll: {pre_roll_seconds}s (captures speech onset)")
        print(f"  - Speech padding: {speech_pad_ms}ms (prevents clipping)")

    def add_audio_chunk(self, audio_chunk: np.ndarray) -> list[bytes]:
        if audio_chunk.size == 0:
            return []
        if self.buffer.size == 0:
            self.buffer = audio_chunk.copy()
        else:
            self.buffer = np.concatenate((self.buffer, audio_chunk))
        return self._extract_sentences(flush=False)

    def flush(self) -> list[bytes]:
        return self._extract_sentences(flush=True)

    def _extract_sentences(self, flush: bool) -> list[bytes]:
        """
        Extract speech sentences using Silero VAD.
        
        Silero VAD is a neural network that accurately distinguishes speech from noise.
        This prevents false positives (hallucinated audio) that energy-based methods cause.
        """
        if self.buffer.size == 0:
            return []

        # Ensure model is loaded (should never be None due to __init__ check, but defensive)
        if self.model is None or self.get_speech_ts is None:
            print("ERROR: Silero VAD model not available. Cannot process audio safely.")
            return []  # Return empty - don't hallucinate

        # Silero VAD needs minimum buffer size to work properly
        # Minimum: 0.5 seconds for reliable detection
        min_buffer_samples = int(self.sample_rate * 0.5)
        if not flush and self.buffer.size < min_buffer_samples:
            # Not enough audio yet, wait for more chunks
            return []

        # Calculate cutoff: for flush, process everything; otherwise keep guard_samples for lookahead
        if flush:
            cutoff = len(self.buffer)
        else:
            # Keep guard_samples (silence_duration) at the end to detect continuing speech
            # Only process if we have enough audio beyond the guard region
            min_process_samples = max(min_buffer_samples, self.guard_samples + int(self.sample_rate * 0.5))
            if self.buffer.size < min_process_samples:
                return []
            cutoff = max(0, len(self.buffer) - self.guard_samples)
        
        if cutoff <= 0:
            return []

        try:
            # Convert to tensor and run Silero VAD
            # Silero VAD expects float32 audio in range [-1.0, 1.0]
            audio_tensor = torch.from_numpy(self.buffer[:cutoff]).float()
            
            with torch.no_grad():
                speech_segments = self.get_speech_ts(
                    audio_tensor,
                    self.model,
                    sampling_rate=self.sample_rate,
                    min_speech_duration_ms=self.min_speech_ms,
                    min_silence_duration_ms=int(self.guard_samples / self.sample_rate * 1000),
                    speech_pad_ms=self.speech_pad_ms,
                )
        except Exception as exc:
            print(f"ERROR: Silero VAD processing failed: {exc}")
            import traceback
            traceback.print_exc()
            # Return empty - don't use fallback to prevent hallucinations
            return []

        sentences = []
        consumed_until = 0
        
        # Process detected speech segments from Silero VAD
        # Silero VAD returns list of dicts with 'start' and 'end' keys (in samples)
        for segment in speech_segments:
            # Extract start and end timestamps (in samples)
            start = segment.get("start", 0)
            end = segment.get("end", start + 1)
            
            # Ensure valid range
            start = max(0, int(start))
            end = max(start + 1, int(end))
            
            # Ensure indices are within buffer bounds
            start = min(start, len(self.buffer))
            end = min(end, len(self.buffer))
            
            if start >= end or end > cutoff:
                continue

            # Prepend pre-roll to capture speech onset (avoids clipping leading phonemes)
            start_pre = max(0, start - self.pre_roll_samples)
            clip = self.buffer[start_pre:end]
            
            # Validate minimum duration
            if clip.size < self.min_sentence_samples:
                continue

            sentences.append(self._to_wav_bytes(clip))
            consumed_until = max(consumed_until, end)
            
            duration_sec = clip.size / self.sample_rate
            print(f"✓ Silero VAD detected speech: {duration_sec:.2f}s (samples {start_pre}-{end})")

        # Update buffer: remove processed audio while preserving context
        if consumed_until > 0:
            # Remove processed audio, but keep some overlap to avoid cutting off speech
            # Keep pre_roll_samples to preserve context for next detection
            keep_samples = max(self.pre_roll_samples, int(self.sample_rate * 0.15))  # Keep 150ms or pre_roll
            keep_from = max(0, consumed_until - keep_samples)
            self.buffer = self.buffer[keep_from:]
            print(f"  Buffer updated: processed {consumed_until} samples, kept {len(self.buffer)} samples ({len(self.buffer)/self.sample_rate:.2f}s)")
        elif not flush:
            # No speech detected - keep buffer for future detection
            # Limit buffer size to prevent memory issues
            max_buffer_seconds = 6.0  # Keep up to 6 seconds of audio
            max_buffer_samples = int(self.sample_rate * max_buffer_seconds)
            if len(self.buffer) > max_buffer_samples:
                # Keep the most recent audio (preserves context for speech that might be starting)
                tail_start = len(self.buffer) - max_buffer_samples
                self.buffer = self.buffer[tail_start:]
                print(f"  Buffer truncated: {len(self.buffer)} samples ({len(self.buffer)/self.sample_rate:.2f}s) - no speech detected")
        else:
            # Flush: clear buffer
            self.buffer = np.array([], dtype=np.float32)
            print("  Buffer flushed")

        return sentences

    def _to_wav_bytes(self, clip: np.ndarray) -> bytes:
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            audio_clamped = np.clip(clip, -1.0, 1.0)
            audio_int16 = (audio_clamped * 32767).astype(np.int16)
            wav_file.writeframes(audio_int16.tobytes())
        wav_buffer.seek(0)
        return wav_buffer.read()


class ClientSpeechService:
    """
    Continuously capture client audio and emit English translations.
    Uses interval-based pseudo-streaming (no VAD) to match Tel User behavior.
    """

    def __init__(self, on_sentence, capture_rate: int = 48000):
        self.on_sentence = on_sentence
        self.capture_rate = capture_rate
        self.queue: queue.Queue[bytes] = queue.Queue()
        self.active = False
        self.capture_thread: threading.Thread | None = None
        
        # Audio accumulator for interval processing
        self.buffer = bytearray()
        self.last_transcript = ""
        self.asr_interval = 1.5  # Seconds between transcribes
        self.last_asr_ts = 0.0
        self.sample_rate = 16000 # Whisper expects 16k
        
    def start(self):
        if self.active:
            return
        self.active = True
        self.capture_thread = threading.Thread(target=self._run, daemon=True)
        self.capture_thread.start()

    def stop(self):
        self.active = False
        if self.capture_thread:
            # Wait for thread to finish
            self.capture_thread.join(timeout=1)
            self.capture_thread = None
        # Flush any remaining audio? Optional, but good practice.
        # For now, just clear.
        self.buffer.clear()

    def _run(self):
        try:
            device_index, resolved_device = resolve_capture_device()
            print(f"Client capture starting on device: {resolved_device} (index: {device_index})")
        except Exception as exc:
            print(f"Unable to resolve capture device: {exc}")
            self.active = False
            return

        stream_kwargs = {
            "channels": 1,
            "samplerate": self.capture_rate,
            "dtype": "float32",
            "callback": self._capture_callback,
            "blocksize": 2048,
        }

        if device_index is not None:
             stream_kwargs["device"] = device_index
        elif resolved_device:
             stream_kwargs["device"] = resolved_device

        print("Client capture stream opened (Interval Mode)...")
        try:
            with sd.InputStream(**stream_kwargs):
                while self.active:
                    try:
                        # Non-blocking get with short timeout
                        chunk = self.queue.get(timeout=0.2)
                        self._process_chunk(chunk)
                    except queue.Empty:
                        pass
                    
                    # Periodic processing check
                    now = time.monotonic()
                    if (now - self.last_asr_ts) >= self.asr_interval:
                        self._run_asr_interval(flush=False)

        except Exception as exc:
            print(f"Client capture stream error: {exc}")
            import traceback
            traceback.print_exc()
        finally:
            # Final flush on exit
            if len(self.buffer) > 0:
                 self._run_asr_interval(flush=True)
            self.active = False

    def _capture_callback(self, indata, frames, time_info, status):
        if status:
            print(f"Client capture status: {status}")
        if not self.active:
            return
        
        # Convert to int16
        audio_int16 = (indata[:, 0] * 32767).astype(np.int16)
        try:
            self.queue.put_nowait(audio_int16.tobytes())
        except queue.Full:
            pass

    def _process_chunk(self, chunk: bytes):
        if not chunk:
            return
        
        # Resample 48k -> 16k
        audio_int16 = np.frombuffer(chunk, dtype=np.int16)
        audio_float = audio_int16.astype(np.float32) / 32768.0
        resampled = signal.resample_poly(audio_float, 1, 3) # 48k / 3 = 16k
        
        # Convert back to int16 bytes for buffer
        resampled_int16 = (resampled * 32767).astype(np.int16)
        self.buffer.extend(resampled_int16.tobytes())

    def _run_asr_interval(self, flush=False):
        self.last_asr_ts = time.monotonic()
        
        # Need enough audio (e.g. 0.5s) unless flushing
        if not flush and len(self.buffer) < (16000 * 2 * 0.5): 
            return

        try:
            # Transcribe current buffer
            pcm_bytes = bytes(self.buffer)
            transcript = transcribe_pcm_stream(pcm_bytes, self.sample_rate)
            
            if not transcript:
                 return # Silence or no speech
            
            if not flush and transcript == self.last_transcript:
                 return # No change in transcript

            self.last_transcript = transcript
            
            # Translate JP -> EN (Client speaks JP usually per original code?)
            # Wait, ClientSpeechService docs said "emit English translations (from trinity2.py)" 
            # and code called translate_JP_EN. So Input=JP, Output=EN.
            translation_result = translate_JP_EN(transcript)
            english_text = translation_result.get("english", "").strip()
            
            # Emit result
            if english_text:
                result = {
                    "type": "final" if flush else "partial", # Or "sentence" used previously
                    "transcription": transcript,
                    "translation": english_text,
                    "language": "en",
                    "speaker": "client",
                    "partial": not flush # Add explicit partial flag for frontend
                }
                
                if self.on_sentence:
                    print(f"Broadcasting client update: {english_text[:40]}...")
                    self.on_sentence(result)

        except Exception as e:
            print(f"Client ASR Error: {e}")
            
        if flush:
             self.buffer.clear()
             self.last_transcript = ""


def broadcast_client_sentence(payload: dict):
    """Broadcast client sentence to all WebSocket subscribers."""
    with client_transcript_lock:
        subscribers = list(client_transcript_subscribers)
    print(f"Broadcasting client sentence to {len(subscribers)} subscriber(s)")
    for subscriber in subscribers:
        queue_ref = subscriber.get("queue")
        loop = subscriber.get("loop")
        try:
            asyncio.run_coroutine_threadsafe(queue_ref.put(payload), loop)
        except Exception as exc:
            print(f"Failed to broadcast client sentence: {exc}")


def ensure_client_transcript_service():
    """Ensure client transcript service is running."""
    global client_transcript_service
    with client_transcript_service_lock:
        if client_transcript_service is None:
            print("Initializing client transcript service...")
            client_transcript_service = ClientSpeechService(on_sentence=broadcast_client_sentence)
            # client_transcript_service.start() - Don't auto-start! Wait for user command.
            print("Client transcript service initialized (waiting for start command)")
        # REMOVED auto-restart logic. Start happens only via WebSocket command.


def get_device_index(device_name):
    """Get device index by name, returns None if not found"""
    try:
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device_name.lower() in str(device).lower() or device_name.lower() in str(device['name']).lower():
                return i
        # Try direct lookup by name
        try:
            device_info = sd.query_devices(device_name)
            return device_info['index']
        except:
            pass
    except Exception as e:
        print(f"Error querying devices: {e}")
    return None


def play_to_virtual_mic(pcm_data, sample_rate):
    """
    Play TTS audio to the ALSA device that maps to PulseAudio sink virtual_tts.
    Requires ~/.asoundrc entry injected by scripts/configure_asoundrc.sh
    """
    audio_np = np.frombuffer(pcm_data, dtype=np.int16)
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)
    audio_np = audio_np.astype(np.float32) / 32768.0

    sd.default.latency = (0.1, 0.2)
    sd.default.blocksize = 2048

    device_name = os.environ.get("TTS_ALSA_DEVICE", DEFAULT_TTS_ALSA_DEVICE)
    device_index = get_device_index(device_name)

    print(f"Playing TTS to {device_name} ({'index ' + str(device_index) if device_index is not None else 'by name'}) at {sample_rate} Hz")

    try:
        if device_index is not None:
            sd.play(audio_np, samplerate=sample_rate, device=device_index)
        else:
            sd.play(audio_np, samplerate=sample_rate, device=device_name)
        sd.wait()
        print(f"TTS audio sent to {VIRTUAL_TTS_NAME} successfully")
    except Exception as e:
        print(f"Error playing TTS to {device_name}: {e}")
        try:
            devices = [str(dev) for dev in sd.query_devices()]
            print(f"Available audio devices: {devices}")
        except Exception as dev_err:
            print(f"Unable to query audio devices: {dev_err}")

# --- Audio capture callback ---
def capture_from_virtual_mic_callback(indata, frames, time, status):
    global virtual_capture_active
    if status:
        print(f"Audio capture status: {status}")
    if virtual_capture_active:
        # Convert float32 to int16
        audio_int16 = (indata[:, 0] * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        virtual_capture_queue.put(audio_bytes)

        # Send to all WebSocket clients
        chunk_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        with websocket_lock:
            for client in websocket_clients:
                try:
                    client['queue'].put_nowait(chunk_base64)
                except:
                    pass

# --- Capture service functions ---
def start_virtual_capture(sample_rate=48000, channels=1):
    global virtual_capture_active, virtual_capture_thread, virtual_audio_buffer

    if virtual_capture_active:
        return {"status": "already_active"}

    virtual_capture_active = True
    virtual_audio_buffer = []

    def capture_thread():
        global virtual_capture_active, virtual_audio_buffer
        try:
            # Capture from configured virtual monitor, falling back to Pulse monitor name
            device_index, resolved_device = resolve_capture_device()

            stream_kwargs = {
                'channels': channels,
                'samplerate': sample_rate,
                'dtype': 'float32',
                'callback': capture_from_virtual_mic_callback,
                'blocksize': 2048
            }

            if device_index is not None:
                stream_kwargs['device'] = device_index
            elif resolved_device:
                stream_kwargs['device'] = resolved_device
            else:
                raise RuntimeError("No capture device available for virtual capture")

            with sd.InputStream(**stream_kwargs) as stream:
                while virtual_capture_active:
                    try:
                        chunk = virtual_capture_queue.get(timeout=0.1)
                        virtual_audio_buffer.append(chunk)
                    except:
                        continue
        except Exception as e:
            print(f"Capture thread error: {e}")
            virtual_capture_active = False

    virtual_capture_thread = threading.Thread(target=capture_thread, daemon=True)
    virtual_capture_thread.start()
    import time
    time.sleep(0.2)

    return {"status": "started"} if virtual_capture_active else {"status": "error", "message": "Failed to start capture"}

def stop_virtual_capture():
    global virtual_capture_active, virtual_audio_buffer

    if not virtual_capture_active:
        return None

    virtual_capture_active = False
    import time
    time.sleep(0.3)

    if not virtual_audio_buffer:
        return None

    audio_data = b''.join(virtual_audio_buffer)

    # Write WAV file in memory
    wav_buffer = io.BytesIO()
    sample_rate = 48000
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data)

    wav_buffer.seek(0)
    virtual_audio_buffer = []
    return wav_buffer.read()

def generate_tts(text: str, language: str = "ja") -> io.BytesIO:
    """
    Generate TTS in the requested language and send it to the virtual_tts sink.
    Audio flow: TTS → virtual_tts → virtual_source → Teams microphone
    """
    print(f"Generating TTS ({language}) for text: {text[:50]}...")
    tts = gTTS(text=text, lang=language)
    audio_bytes = io.BytesIO()
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)

    # Convert to mono 16-bit PCM
    audio = AudioSegment.from_file(audio_bytes, format="mp3")
    audio = audio.set_frame_rate(audio.frame_rate).set_channels(1).set_sample_width(2)
    pcm_data = audio.raw_data
    sample_rate = audio.frame_rate

    # Play into virtual_tts so Teams (set to virtual_source) hears the speech
    play_to_virtual_mic(pcm_data, sample_rate)

    return audio_bytes


# Translation JP -> EN
def translate_JP_EN(text: str) -> dict:
    client = Groq(api_key=API_KEY)
    strict_prompt = """
                    You are a STRICT English translator. Understand the language accurately and then perform the translation accordingly.

                    Rules:
                    1. Translate the ENTIRE text into clear English.
                    2. Do NOT summarize or shorten the meaning.
                    3. Do NOT omit any nuance or details.
                    4. OUTPUT MUST BE VALID JSON in this exact format:

                    {
                    "english": "<full English translation>"
                    }

                    Make sure the JSON is valid.
                    """
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role":"system","content":strict_prompt},
                  {"role":"user","content":text}],
        temperature=0
    )
    try:
        return json.loads(resp.choices[0].message.content)
    except:
        return {"english":""}

# Translation EN -> JP
def translate_EN_JP(text: str) -> dict:
    client = Groq(api_key=API_KEY)
    strict_prompt = """
                    You are a STRICT translation engine.

                    Rules you must follow:
                    1. Translate the ENTIRE input text into natural, fluent Japanese.
                    2. Do NOT shorten, simplify, or summarize the meaning.
                    3. Do NOT drop any parts of the sentence.
                    4. Keep proper nouns exactly as they are (e.g., "Dhvani").
                    5. OUTPUT MUST BE VALID JSON in this format:

                    {
                    "japanese": "<full accurate Japanese translation>"
                    }

                    Make sure the JSON is valid.
                    """
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role":"system","content":strict_prompt},
                  {"role":"user","content":text}],
        temperature=0
    )
    try:
        return json.loads(resp.choices[0].message.content)
    except:
        return {"japanese":""}

def translate_EN_HI(text: str) -> dict:
    client = Groq(api_key=API_KEY)
    strict_prompt = """
                    You are a STRICT translation engine.

                    Rules you must follow:
                    1. Translate the ENTIRE input text into natural, fluent Hindi.
                    2. Do NOT shorten, simplify, or summarize the meaning.
                    3. Do NOT drop any parts of the sentence.
                    4. Keep proper nouns exactly as they are (e.g., "Dhvani").
                    5. OUTPUT MUST BE VALID JSON in this format:

                    {
                    "hindi": "<full accurate Hindi translation>"
                    }

                    Make sure the JSON is valid.
                    """
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role":"system","content":strict_prompt},
                  {"role":"user","content":text}],
        temperature=0
    )
    try:
        return json.loads(resp.choices[0].message.content)
    except:
        return {"hindi":""}

# Transcription using Groq Whisper
def transcribe_audio(audio_data: bytes) -> str:
    client = Groq(api_key=API_KEY)
    # Convert bytes to AudioSegment for resampling if needed
    audio = AudioSegment.from_file(io.BytesIO(audio_data))  # Let pydub detect format
    audio = audio.set_frame_rate(16000).set_channels(1)  # Whisper expects 16kHz mono
    bio = io.BytesIO()
    audio.export(bio, format="wav")
    bio.seek(0)
    resp = client.audio.transcriptions.create(
        file=("speech.wav", bio, "audio/wav"), 
        model="whisper-large-v3"
    )
    return resp.text.strip()


def transcribe_pcm_stream(pcm_bytes: bytes, sample_rate: int) -> str:
    """Transcribe raw 16-bit mono PCM captured from the browser."""
    audio = AudioSegment.from_raw(
        io.BytesIO(pcm_bytes),
        sample_width=2,
        frame_rate=sample_rate,
        channels=1,
    )
    audio = audio.set_frame_rate(16000).set_channels(1)
    bio = io.BytesIO()
    audio.export(bio, format="wav")
    bio.seek(0)

    client = Groq(api_key=API_KEY)
    resp = client.audio.transcriptions.create(
        file=("speech.wav", bio, "audio/wav"),
        model="whisper-large-v3",
    )
    return resp.text.strip()


def translate_to_target(text: str, target_language: str) -> str:
    """Translate English text to the requested target language."""
    lang = (target_language or "ja").lower()
    try:
        if lang == "hi":
            data = translate_EN_HI(text)
            return data.get("hindi", "")
        data = translate_EN_JP(text)
        return data.get("japanese", "")
    except Exception as exc:
        print(f"translate_to_target error ({lang}): {exc}")
        return ""

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.wav', '.mp3', '.m4a', '.flac', '.webm')):
        raise HTTPException(status_code=400, detail="Unsupported file type")
    audio_data = await file.read()
    try:
        text = transcribe_audio(audio_data)
        return JSONResponse(content={"transcription": text})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@app.post("/translate/en-to-jp")
async def translate_en_to_jp(data: dict):
    text = data.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")
    try:
        result = translate_EN_JP(text)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

@app.post("/translate/en-to-hi")
async def translate_en_to_hi(data: dict):
    text = data.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")
    try:
        result = translate_EN_HI(text)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

@app.post("/translate/jp-to-en")
async def translate_jp_to_en(data: dict):
    text = data.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")
    try:
        result = translate_JP_EN(text)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")


@app.post("/tts")
async def tts(data: dict):
    text = data.get("text", "")
    language = data.get("language", "ja")
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")
    try:
        audio_bytes = generate_tts(text, language=language)
        return StreamingResponse(io.BytesIO(audio_bytes.getvalue()), media_type="audio/mp3")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)}")

@app.post("/get_transcription")
async def get_transcription(data: dict):
    messages = data.get("messages", [])
    if not messages:
        raise HTTPException(status_code=400, detail="Messages are required")
    try:
        transcription_lines = []
        for msg in messages:
            sender = "Tel User" if msg.get("sender") == "A" else "Client"
            if msg.get("sender") == "A":
                # Tel User: english (japanese)
                input_text = msg.get('input', '')
                output_text = msg.get('output', '')
                line = f"{sender}: {input_text} ({output_text})"
            else:
                # Client: english (japanese) - note: input may be empty for real-time updates
                output_text = msg.get('output') or msg.get('text', '')
                input_text = msg.get('input', '')
                if input_text:
                    line = f"{sender}: {output_text} ({input_text})"
                else:
                    line = f"{sender}: {output_text}"
            transcription_lines.append(line)
        full_transcription = "\n".join(transcription_lines)
        # Create a text file in memory
        text_bytes = io.BytesIO(full_transcription.encode('utf-8'))
        text_bytes.seek(0)
        # Return as downloadable text file
        return StreamingResponse(
            text_bytes,
            media_type="text/plain",
            headers={"Content-Disposition": "attachment; filename=conversation_transcription.txt"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate transcription: {str(e)}")


@app.websocket("/ws/audio-stream")
async def websocket_audio_stream(websocket: WebSocket):
    await websocket.accept()
    print(f"WebSocket connected: {websocket.client}")

    client_queue = asyncio.Queue()
    client_info = {'queue': client_queue}

    with websocket_lock:
        websocket_clients.append(client_info)

    async def send_chunks():
        try:
            while True:
                chunk = await client_queue.get()
                await websocket.send_json({"type": "audio_chunk", "chunk": chunk})
        except Exception as e:
            print(f"Error sending chunk: {e}")

    send_task = asyncio.create_task(send_chunks())

    try:
        while True:
            data = await websocket.receive_text()
            if data == "start":
                result = start_virtual_capture()
                await websocket.send_json(result)
            elif data == "stop":
                audio_data = stop_virtual_capture()
                await asyncio.sleep(0.2)
                if audio_data:
                    audio_base64 = base64.b64encode(audio_data).decode('utf-8')
                    await websocket.send_json({"status": "stopped", "audio": audio_base64, "format": "wav"})
                else:
                    await websocket.send_json({"status": "stopped", "error": "No audio captured"})
                break
    except WebSocketDisconnect:
        print(f"WebSocket disconnected: {websocket.client}")
    finally:
        send_task.cancel()
        with websocket_lock:
            websocket_clients.remove(client_info)
        try:
            await send_task
        except asyncio.CancelledError:
            pass
        print(f"WebSocket connection closed: {websocket.client}")


@app.websocket("/ws/tel-asr")
async def websocket_tel_asr(websocket: WebSocket):
    """Pseudo-streaming ASR for Tel user captured in the browser (from be5.py)."""
    await websocket.accept()
    print(f"Tel ASR WebSocket connected: {websocket.client}")

    session = {
        "buffer": bytearray(),
        "last_transcript": "",
        "target_language": "ja",
        "sample_rate": 48000,
        "spoken_len": 0,
        "last_spoken_translation": "",
        "last_speak_ts": 0.0,
        "closing": False,
    }
    asr_interval = 1.5
    last_asr_ts = 0.0
    processing_lock = asyncio.Lock()

    async def safe_send(payload: dict):
        """Send JSON if the WebSocket is still connected."""
        if websocket.application_state != WebSocketState.CONNECTED:
            return False
        try:
            await websocket.send_json(payload)
            return True
        except Exception as exc:
            print(f"Failed to send Tel ASR update: {exc}")
            return False

    async def speak_delta(translation: str, *, flush: bool = False):
        """
        Speak only the newly discovered portion of the translation.
        Throttles rapid partials and skips if nothing new or text regresses.
        """
        if not translation:
            return
        # Only speak the delta beyond what has already been voiced
        start = session.get("spoken_len", 0)
        if start < 0:
            start = 0
        if start > len(translation):
            start = 0
        new_segment = translation[start:].strip()
        # If model backs off/edits earlier text, sync spoken pointer without speaking
        if len(translation) < session.get("spoken_len", 0):
            session["spoken_len"] = len(translation)
            session["last_spoken_translation"] = translation
            return

        # Optionally throttle very rapid partials
        now = time.monotonic()
        min_gap = 0.8  # seconds between spoken deltas
        if not flush and (now - session.get("last_speak_ts", 0.0)) < min_gap:
            return

        # Delta based on current spoken length
        if len(new_segment) < 3:
            return
        session["spoken_len"] = len(translation)
        session["last_spoken_translation"] = translation
        session["last_speak_ts"] = now
        try:
            async with tts_semaphore:
                await asyncio.get_running_loop().run_in_executor(
                    None, lambda: generate_tts(new_segment, language=session["target_language"])
                )
        except Exception as exc:
            print(f"Tel ASR incremental TTS error: {exc}")

    async def run_asr(flush: bool = False):
        nonlocal last_asr_ts
        now = time.monotonic()
        if not flush and (now - last_asr_ts) < asr_interval:
            return
        last_asr_ts = now

        async with processing_lock:
            if (session["closing"] and not flush) or websocket.application_state != WebSocketState.CONNECTED:
                return

            pcm_bytes = bytes(session["buffer"])
            if not flush and len(pcm_bytes) < 64000:  # ~0.66s at 48kHz mono 16-bit
                return

            def do_transcribe():
                return transcribe_pcm_stream(pcm_bytes, session["sample_rate"])

            try:
                transcript = await asyncio.get_running_loop().run_in_executor(
                    None, do_transcribe
                )
            except Exception as exc:
                print(f"Tel ASR transcription error: {exc}")
                return

            if not transcript:
                return
            if not flush and transcript == session["last_transcript"]:
                return

            session["last_transcript"] = transcript
            translation = translate_to_target(transcript, session["target_language"])
            message_type = "final" if flush else "partial"

            # Speak newly identified text as we go
            await speak_delta(translation, flush=False)

            await safe_send(
                {
                    "type": message_type,
                    "transcription": transcript,
                    "translation": translation,
                    "language": session["target_language"],
                }
            )

            if flush and translation:
                # Ensure any tail is voiced
                await speak_delta(translation, flush=True)

    try:
        while True:
            message = await websocket.receive()
            if "bytes" in message and message["bytes"] is not None:
                session["buffer"].extend(message["bytes"])
                await run_asr()
            elif "text" in message and message["text"] is not None:
                try:
                    data = json.loads(message["text"])
                except Exception:
                    continue
                msg_type = data.get("type")
                if msg_type == "start":
                    session["target_language"] = data.get("targetLanguage", "ja")
                    session["sample_rate"] = int(data.get("sampleRate", 48000))
                    session["buffer"].clear()
                    session["last_transcript"] = ""
                    await safe_send({"status": "ready"})
                elif msg_type == "stop":
                    session["closing"] = True
                    await run_asr(flush=True)
                    break
            else:
                # Ignore unhandled frames
                continue
    except WebSocketDisconnect:
        print(f"Tel ASR WebSocket disconnected: {websocket.client}")
    finally:
        session["closing"] = True
        try:
            if websocket.application_state == WebSocketState.CONNECTED:
                await websocket.close()
        except Exception:
            pass
        print(f"Tel ASR WebSocket closed: {websocket.client}")


@app.websocket("/ws/client-transcripts")
async def websocket_client_transcripts(websocket: WebSocket):
    """WebSocket endpoint for real-time client transcript streaming."""
    await websocket.accept()
    print(f"Client transcript WebSocket connected: {websocket.client}")

    # Ensure service is running
    ensure_client_transcript_service()

    client_queue = asyncio.Queue()
    loop = asyncio.get_event_loop()
    client_info = {'queue': client_queue, 'loop': loop}

    with client_transcript_lock:
        client_transcript_subscribers.append(client_info)

    async def send_transcripts():
        try:
            while True:
                payload = await client_queue.get()
                await websocket.send_json(payload)
        except Exception as e:
            print(f"Error sending transcript: {e}")

    send_task = asyncio.create_task(send_transcripts())

    try:
        # Keep connection alive and listen for control messages
        while True:
            try:
                message = await websocket.receive_text()
                data = json.loads(message)
                if data.get("action") == "start":
                    print("Received remote START command for Client Speech Service")
                    ensure_client_transcript_service()
                    # Also ensure it isn't stopped if it was created but stopped
                    if client_transcript_service and not client_transcript_service.active:
                         client_transcript_service.start()
                elif data.get("action") == "stop":
                     print("Received remote STOP command for Client Speech Service")
                     if client_transcript_service:
                         client_transcript_service.stop()
            except json.JSONDecodeError:
                 pass
            except WebSocketDisconnect:
                break
            except Exception as e:
                print(f"WS Error: {e}")
                break
    except WebSocketDisconnect:
        print(f"Client transcript WebSocket disconnected: {websocket.client}")
    finally:
        send_task.cancel()
        with client_transcript_lock:
            if client_info in client_transcript_subscribers:
                client_transcript_subscribers.remove(client_info)
        try:
            await send_task
        except asyncio.CancelledError:
            pass
        print(f"Client transcript WebSocket connection closed: {websocket.client}")        



@app.get("/")
async def root():
    return FileResponse("static/last.html")

if __name__ == "__main__":
    import uvicorn    
    uvicorn.run(app, port=8003)






<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dhvani</title>
    <link rel="shortcut icon" href="static/favicon.ico">
    <style>
        :root {
            --bg-dark: #070710;
            --bg-panel: rgba(22, 22, 33, 0.9);
            --text-primary: #f6f6ff;
            --text-muted: #a9a9c4;
            --accent: #7d5cff;
            --accent-strong: #c982ff;
            --success: #48f3b2;
            --warning: #ffb25f;
            --gradient: radial-gradient(circle at 20% 20%, rgba(108, 95, 255, 0.35), transparent 50%), radial-gradient(circle at 80% 0%, rgba(255, 108, 222, 0.4), transparent 52%), #05030b;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        html,
        body {
            font-family: 'Inter', 'Segoe UI', 'Roboto', sans-serif;
            min-height: 100vh;
            color: var(--text-primary);
            background: var(--gradient);
        }

        body {
            margin: 0;
        }

        .app-shell {
            height: 100vh;
            padding: 2rem clamp(1.5rem, 3vw, 4rem);
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
            overflow: hidden;
        }

        main {
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
            min-height: 0;
            overflow: hidden;
        }

        .glass {
            background: var(--bg-panel);
            border-radius: 5px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            box-shadow: 0 40px 80px rgba(0, 0, 0, 0.5);
            backdrop-filter: blur(14px);
        }

        header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1.5rem;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        .logo-img {
            width: 56px;
            height: 56px;
            object-fit: contain;
            border-radius: 12px;
            box-shadow: none;
        }

        .brand h1 {
            font-size: clamp(1.75rem, 2vw, 2rem);
            letter-spacing: 0.05em;
        }

        .pill {
            border-radius: 900px;
            padding: 0.35rem 1rem;
            background: rgba(255, 255, 255, 0.08);
            color: var(--text-muted);
            font-size: 0.875rem;
        }

        .top-actions {
            margin-left: auto;
            display: flex;
            gap: 1rem;
            align-items: center;
        }

        button {
            border: none;
            border-radius: 5px;
            padding: 0.9rem 1.75rem;
            background: linear-gradient(135deg, var(--accent), var(--accent-strong));
            color: #fff;
            font-weight: 600;
            letter-spacing: 0.05em;
            cursor: pointer;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 20px 40px rgba(125, 92, 255, 0.3);
        }

        button.recording {
            background: linear-gradient(135deg, #ff5f95, #ff9770);
            box-shadow: 0 10px 30px rgba(255, 130, 170, 0.35);
        }

        .control-panel {
            padding: 2rem;
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
            flex: 0 0 auto;
        }

        .voice-stack {
            display: flex;
            flex-direction: column;
            gap: 1rem;
            align-items: center;
        }

        .voice-visualizer {
            height: 80px;
            border-radius: 5px;
            border: 1px dashed rgba(255, 255, 255, 0.15);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.2rem;
            overflow: hidden;
            padding: 0 1rem;
        }

        .voice-visualizer span {
            width: 6px;
            border-radius: 900px;
            background: rgba(132, 94, 247, 0.4);
            transition: height 0.12s ease, opacity 0.12s ease;
            height: 12px;
        }

        .voice-visualizer.active {
            border-color: rgba(125, 92, 255, 0.8);
            box-shadow: inset 0 0 25px rgba(125, 92, 255, 0.25);
        }

        .language-select {
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
            color: var(--text-muted);
        }

        .language-select.inline {
            width: auto;
            min-width: 200px;
            position: relative;
        }

        .language-select.inline label {
            text-align: left;
        }

        .language-select select {
            border-radius: 5px;
            border: 1px solid rgba(255, 255, 255, 0.15);
            background: rgba(255, 255, 255, 0.12);
            color: var(--text-primary);
            padding: 0.65rem 0.75rem;
            font-size: 0.95rem;
            outline: none;
            appearance: none;
            position: relative;
            padding-right: 2rem;
        }

        .language-select select option {
            background: #12121d;
            color: #f6f6ff;
        }

        .language-select.inline::after {
            content: "▾";
            position: absolute;
            right: 0.75rem;
            top: 50%;
            transform: translateY(-50%);
            color: var(--text-primary);
            pointer-events: none;
            font-size: 0.9rem;
            opacity: 0.8;
        }

        .language-select select:focus {
            border-color: var(--accent);
            box-shadow: 0 0 0 2px rgba(125, 92, 255, 0.2);
        }

        .recording-group {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 0.75rem;
        }

        .recording-group button {
            min-width: 220px;
        }

        .conversation-panel {
            display: flex;
            flex-direction: column;
            padding: 2rem;
            border-radius: 5px;
            background: radial-gradient(circle at top, rgba(125, 92, 255, 0.25), transparent 55%), var(--bg-panel);
            flex: 1;
            min-height: 0;
        }

        .conversation-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 1.5rem;
        }

        .conversation-panel h2 {
            margin-bottom: 1rem;
        }

        .chat-container {
            flex: 1;
            min-height: 0;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 1.25rem;
            padding-right: 0.5rem;
            scroll-behavior: smooth;
            scrollbar-width: thin;
        }

        .chat-container::-webkit-scrollbar {
            width: 6px;
        }

        .chat-container::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.2);
            border-radius: 900px;
        }

        .chat-bubble-a,
        .chat-bubble-b {
            padding: 1.25rem;
            border-radius: 5px;
            width: min(520px, 48%);
            line-height: 1.55;
            font-size: 0.95rem;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.35);
            word-break: break-word;
            border: 1px solid rgba(255, 255, 255, 0.08);
        }

        .chat-bubble-a {
            align-self: flex-start;
            background: linear-gradient(135deg, rgba(103, 80, 255, 0.4), rgba(103, 80, 255, 0.15));
            border: 1px solid rgba(125, 92, 255, 0.4);
        }

        .chat-bubble-b {
            align-self: flex-end;
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.08);
        }

        .chat-bubble-a b,
        .chat-bubble-b b {
            display: block;
            font-size: 0.8rem;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            margin-bottom: 0.5rem;
            color: var(--text-muted);
        }

        .chat-bubble-a audio,
        .chat-bubble-b audio {
            margin-top: 0.75rem;
            width: 100%;
        }

        @media (max-width: 640px) {
            header {
                flex-direction: column;
                align-items: flex-start;
            }

            .top-actions {
                width: 100%;
                flex-direction: column;
                align-items: stretch;
            }

            button,
            .select {
                width: 100%;
            }

            .chat-bubble-a,
            .chat-bubble-b {
                width: 100%;
            }
        }
    </style>
</head>

<body>
    <div class="app-shell">
        <header>
            <div class="brand">
                <img src="static/Dhvani_Logo.svg" alt="Dhvani logo" class="logo-img"
                    onerror="this.onerror=null; this.style.display='none';">
                <h1>Dhvani</h1>
            </div>
            <div class="top-actions">
                <div class="language-select inline">
                    <select id="targetLanguageSelect" aria-label="Tel User Output Language">
                        <option value="ja" selected>Japanese</option>
                        <option value="hi">Hindi</option>
                    </select>
                </div>
                <button id="getTranscriptionBtn">Export Transcript</button>
            </div>
        </header>

        <main>
            <section class="control-panel glass">
                <div class="voice-stack">
                    <div class="voice-visualizer" id="voiceVisualizer" aria-hidden="true">
                        <span></span><span></span><span></span><span></span><span></span>
                        <span></span><span></span><span></span><span></span><span></span>
                        <span></span><span></span><span></span><span></span><span></span>
                        <span></span><span></span><span></span><span></span><span></span>
                        <span></span><span></span><span></span><span></span><span></span>
                        <span></span><span></span><span></span><span></span><span></span>
                    </div>
                    <div class="recording-group">
                        <button id="btnA">Start Recording (Tel User)</button>
                        <button id="btnB">Start Recording (Client)</button>
                    </div>
                </div>
            </section>

            <section class="conversation-panel glass">
                <div class="conversation-header">
                    <div>
                        <h2>Meeting Conversation</h2>
                        <p class="pill">Synced bilingual transcript</p>
                    </div>
                </div>
                <div class="chat-container" id="chatContainer"></div>
            </section>
        </main>
    </div>

    <script>
        /**
         * THREADING MODEL & SYNCHRONIZATION:
         * 
         * Backend:
         * - Client (ClientSpeechService): Runs in a separate Python thread (threading.Thread)
         *   - Continuously captures audio from virtual microphone
         *   - Processes audio in the thread, then uses asyncio.run_coroutine_threadsafe()
         *     to send messages to async WebSocket handlers
         * - Tel User: Handled via async WebSocket endpoint (/ws/tel-asr)
         *   - Runs in the async event loop (same thread as FastAPI server)
         *   - Receives audio chunks from browser via WebSocket
         * 
         * Frontend:
         * - Both WebSocket connections run on the main JavaScript thread
         * - Messages arrive asynchronously via WebSocket callbacks
         * - Synchronization strategy:
         *   1. Use requestAnimationFrame to batch DOM updates (prevents excessive re-renders)
         *   2. Atomic state checks (check and update in minimal time window)
         *   3. Defensive checks to ensure bubble references are still valid
         *   4. Clear currentClientBubble when Tel User message appears (ensures conversation flow)
         */
        let messages = [];
        let isRecordingA = false;
        let isRecordingB = false;

        let telAsrSocket = null;
        let telAudioContext = null;
        let telWorkletNode = null;
        let telWorkletUrl = null;
        let telMediaStream = null;
        let telSourceNode = null;
        let telCurrentMessage = null;
        let telStopping = false;

        let analyser = null;
        let analyserSource = null;
        let visualizerAnimation = null;
        let clientTranscriptSocket = null; // WebSocket for client transcripts
        let currentClientBubble = null; // Current client chat bubble being updated

        // DOM update optimization: batch updates using requestAnimationFrame
        let updateChatScheduled = false;
        let updateChatAnimationFrame = null;

        const btnA = document.getElementById('btnA');
        const btnB = document.getElementById('btnB');
        const chatContainer = document.getElementById('chatContainer');
        const getTranscriptionBtn = document.getElementById('getTranscriptionBtn');
        const voiceVisualizer = document.getElementById('voiceVisualizer');
        const visualizerBars = voiceVisualizer ? Array.from(voiceVisualizer.querySelectorAll('span')) : [];
        const targetLanguageSelect = document.getElementById('targetLanguageSelect');

        const workletSource = `
            class PCMWorklet extends AudioWorkletProcessor {
                constructor() {
                    super();
                    this.buffer = [];
                    this.chunkSize = 4096; // ~85ms at 48k
                }
                process(inputs) {
                    const input = inputs[0];
                    if (!input || !input[0]) return true;
                    const data = input[0];
                    for (let i = 0; i < data.length; i++) {
                        const sample = Math.max(-1, Math.min(1, data[i]));
                        const intSample = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
                        this.buffer.push(intSample);
                    }
                    while (this.buffer.length >= this.chunkSize) {
                        const chunk = this.buffer.splice(0, this.chunkSize);
                        const int16 = new Int16Array(chunk);
                        this.port.postMessage({ type: 'chunk', payload: int16 }, [int16.buffer]);
                    }
                    return true;
                }
            }
            registerProcessor('pcm-worklet', PCMWorklet);
        `;

        btnA.addEventListener('click', async () => {
            if (!isRecordingA) {
                await startTelRecording();
            } else {
                stopTelRecording();
            }
        });

        btnB.addEventListener('click', () => {
            if (!clientTranscriptSocket || clientTranscriptSocket.readyState !== WebSocket.OPEN) {
                alert("Client WebSocket not connected. Reconnecting...");
                initializeClientTranscriptSocket();
                return;
            }

            if (!isRecordingB) {
                // Start Client Recording (Server-side)
                clientTranscriptSocket.send(JSON.stringify({ action: "start" }));
                isRecordingB = true;
                btnB.textContent = "Stop Recording (Client)";
                btnB.classList.add('recording');
            } else {
                // Stop Client Recording
                clientTranscriptSocket.send(JSON.stringify({ action: "stop" }));
                isRecordingB = false;
                btnB.textContent = "Start Recording (Client)";
                btnB.classList.remove('recording');
            }
        });

        // Initialize client transcript WebSocket connection
        initializeClientTranscriptSocket();

        getTranscriptionBtn.addEventListener('click', async () => {
            if (messages.length === 0) {
                alert('No conversation to transcribe.');
                return;
            }
            try {
                const response = await fetch('/get_transcription', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ messages: messages })
                });
                if (response.ok) {
                    const blob = await response.blob();
                    const url = window.URL.createObjectURL(blob);
                    const now = new Date();
                    const timestamp = now.getFullYear() + '-' +
                        String(now.getMonth() + 1).padStart(2, '0') + '-' +
                        String(now.getDate()).padStart(2, '0') + '_' +
                        String(now.getHours()).padStart(2, '0') + '-' +
                        String(now.getMinutes()).padStart(2, '0') + '-' +
                        String(now.getSeconds()).padStart(2, '0');

                    const a = document.createElement('a');
                    a.style.display = 'none';
                    a.href = url;
                    a.download = `conversation_transcription_${timestamp}.txt`;
                    document.body.appendChild(a);
                    a.click();
                    window.URL.revokeObjectURL(url);
                    document.body.removeChild(a);
                } else {
                    alert('Error retrieving transcription. Please try again.');
                }
            } catch (error) {
                console.error('Error getting transcription:', error);
                alert('Error retrieving transcription. Please try again.');
            }
        });

        function initializeClientTranscriptSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws/client-transcripts`;

            clientTranscriptSocket = new WebSocket(wsUrl);

            clientTranscriptSocket.onopen = () => {
                console.log('Client transcript WebSocket connected');
            };
            console.log("going to get data");
            clientTranscriptSocket.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    console.log('data came');
                    if ((data.type === 'sentence' || data.type === 'partial' || data.type === 'final') && data.speaker === 'client') {
                        const englishText = data.translation || '';
                        const japaneseText = data.transcription || '';
                        let isPartial = data.partial === true;
                        // For 'final' type, ensure isPartial is false
                        if (data.type === 'final') isPartial = false;

                        if (englishText) {
                            updateClientBubble(englishText, japaneseText, isPartial);
                        }
                    }
                } catch (error) {
                    console.error('Error parsing client transcript message:', error);
                }
            };

            clientTranscriptSocket.onerror = (error) => {
                console.error('Client transcript WebSocket error:', error);
            };

            clientTranscriptSocket.onclose = () => {
                console.warn('Client transcript WebSocket closed. Reconnecting...');
                setTimeout(initializeClientTranscriptSocket, 2000);
            };
        }

        function updateClientBubble(englishText, japaneseText = '', isPartial = false) {
            // Atomic check: get the last message and check currentClientBubble state in one go
            // This minimizes the window for race conditions
            const lastMessage = messages.length > 0 ? messages[messages.length - 1] : null;
            const lastWasTelUser = lastMessage && lastMessage.sender === 'A';

            // Check if currentClientBubble is still valid (exists and is in messages array)
            const hasValidClientBubble = currentClientBubble &&
                messages.includes(currentClientBubble) &&
                currentClientBubble.sender === 'B';

            // Only create a new bubble if:
            // 1. The last message was from Tel User (sender 'A'), OR
            // 2. There's no valid current client bubble
            const shouldCreateNew = lastWasTelUser || !hasValidClientBubble;

            if (shouldCreateNew) {
                // Create a new client bubble
                currentClientBubble = {
                    sender: 'B',
                    input: japaneseText,
                    output: englishText,
                    text: englishText,
                    isUpdating: isPartial
                };
                messages.push(currentClientBubble);
            } else {
                // Update the existing client bubble atomically
                // Only update if it's still the current bubble (defensive check)
                if (currentClientBubble && messages.includes(currentClientBubble)) {
                    currentClientBubble.text = englishText;
                    currentClientBubble.output = englishText;
                    if (japaneseText) {
                        currentClientBubble.input = japaneseText;
                    }
                    currentClientBubble.isUpdating = isPartial;
                } else {
                    // Fallback: currentClientBubble was invalidated, create new one
                    currentClientBubble = {
                        sender: 'B',
                        input: japaneseText,
                        output: englishText,
                        text: englishText,
                        isUpdating: isPartial
                    };
                    messages.push(currentClientBubble);
                }
            }

            // Mark as not updating when final, but keep the reference for future updates
            if (!isPartial && currentClientBubble) {
                currentClientBubble.isUpdating = false;
            }

            updateChat();
        }

        async function startTelRecording() {
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                alert('Your browser does not support microphone access. Please use a modern browser like Chrome, Firefox, or Edge.');
                return;
            }

            try {
                const devices = await navigator.mediaDevices.enumerateDevices();
                const audioInputs = devices.filter(device => device.kind === 'audioinput');
                if (audioInputs.length === 0) {
                    alert('No microphone found. Please connect a microphone and try again.');
                    return;
                }
            } catch (enumError) {
                console.warn('Could not enumerate devices:', enumError);
            }

            let stream;
            try {
                stream = await navigator.mediaDevices.getUserMedia({
                    audio: {
                        echoCancellation: true,
                        noiseSuppression: true,
                        autoGainControl: true
                    }
                });
            } catch (err) {
                console.error('Failed to access microphone:', err);
                alert('Error accessing microphone: ' + err.message);
                return;
            }

            telAudioContext = new (window.AudioContext || window.webkitAudioContext)({ latencyHint: 'interactive' });
            telMediaStream = stream;
            const sampleRate = telAudioContext.sampleRate || 48000;

            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws/tel-asr`;
            telAsrSocket = new WebSocket(wsUrl);

            telAsrSocket.onopen = () => {
                telAsrSocket.send(JSON.stringify({
                    type: 'start',
                    targetLanguage: targetLanguageSelect?.value || 'ja',
                    sampleRate: sampleRate
                }));
            };

            telAsrSocket.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    if (data.type === 'partial' || data.type === 'final') {
                        handleTelSocketMessage(data);
                    }
                } catch (err) {
                    console.error('Error parsing Tel ASR message:', err);
                }
            };

            telAsrSocket.onclose = () => {
                console.warn('Tel ASR socket closed');
                telAsrSocket = null;
                telStopping = false;
                finalizeTelCleanup();
            };
            telAsrSocket.onerror = (err) => {
                console.error('Tel ASR socket error:', err);
            };

            telWorkletUrl = URL.createObjectURL(new Blob([workletSource], { type: 'application/javascript' }));
            await telAudioContext.audioWorklet.addModule(telWorkletUrl);

            telWorkletNode = new AudioWorkletNode(telAudioContext, 'pcm-worklet');
            telWorkletNode.port.onmessage = (event) => {
                if (event.data?.type === 'chunk' && telAsrSocket && telAsrSocket.readyState === WebSocket.OPEN) {
                    telAsrSocket.send(event.data.payload);
                }
            };
            telWorkletNode.onprocessorerror = (err) => console.error('Worklet error:', err);

            telSourceNode = telAudioContext.createMediaStreamSource(stream);
            telSourceNode.connect(telWorkletNode);
            telWorkletNode.connect(telAudioContext.destination); // keep the graph alive

            initAudioGraph(stream);

            isRecordingA = true;
            telCurrentMessage = null;
            btnA.textContent = 'Stop Recording (Tel User)';
            btnA.classList.add('recording');
        }

        function stopTelRecording() {
            telStopping = true;
            if (telAsrSocket && telAsrSocket.readyState === WebSocket.OPEN) {
                // Ask server to flush final transcript/translation before closing
                telAsrSocket.send(JSON.stringify({ type: 'stop' }));
            }
            // Stop sending more audio immediately
            if (telWorkletNode) {
                telWorkletNode.disconnect();
                telWorkletNode = null;
            }
            if (telSourceNode) {
                telSourceNode.disconnect();
                telSourceNode = null;
            }
            if (telMediaStream) {
                telMediaStream.getTracks().forEach(track => track.stop());
                telMediaStream = null;
            }
            if (telAudioContext) {
                telAudioContext.close();
                telAudioContext = null;
            }
            // Keep WebSocket alive to receive final message; cleanup in onclose
            isRecordingA = false;
            btnA.textContent = 'Start Recording (Tel User)';
            btnA.classList.remove('recording');
            stopVisualizerIfIdle();
        }

        function finalizeTelCleanup() {
            if (telWorkletUrl) {
                URL.revokeObjectURL(telWorkletUrl);
                telWorkletUrl = null;
            }
        }

        function handleTelSocketMessage(data) {
            ensureTelBubble(data.language || 'ja');
            if (telCurrentMessage && messages.includes(telCurrentMessage)) {
                // Atomic update of Tel User message
                telCurrentMessage.input = data.transcription || '';
                telCurrentMessage.output = data.translation || '';
                telCurrentMessage.lang = data.language || telCurrentMessage.lang || 'ja';
                telCurrentMessage.isUpdating = data.type !== 'final';
                if (data.type === 'final') {
                    telCurrentMessage = null;
                    // Clear current client bubble so next client message starts a new bubble
                    currentClientBubble = null;
                }
            }
            updateChat();
        }

        function ensureTelBubble(language) {
            // Check if we already have a valid Tel User bubble
            if (telCurrentMessage && messages.includes(telCurrentMessage)) {
                return;
            }
            // Create new Tel User bubble atomically
            telCurrentMessage = {
                sender: 'A',
                input: '',
                output: '',
                lang: language || 'ja',
                isUpdating: true
            };
            messages.push(telCurrentMessage);
            // Clear current client bubble so it stops updating and next client message starts a new bubble
            // This ensures proper conversation flow: Tel User message interrupts Client bubble
            currentClientBubble = null;
        }

        function escapeHtml(text) {
            if (!text) return '';
            return text
                .replace(/&/g, "&amp;")
                .replace(/</g, "&lt;")
                .replace(/>/g, "&gt;")
                .replace(/"/g, "&quot;")
                .replace(/'/g, "&#039;");
        }

        function updateChat() {
            // Batch DOM updates using requestAnimationFrame to avoid excessive re-renders
            if (updateChatScheduled) {
                return; // Already scheduled, skip
            }
            updateChatScheduled = true;

            if (updateChatAnimationFrame) {
                cancelAnimationFrame(updateChatAnimationFrame);
            }

            updateChatAnimationFrame = requestAnimationFrame(() => {
                updateChatScheduled = false;
                updateChatAnimationFrame = null;

                // Create document fragment for efficient DOM manipulation
                const fragment = document.createDocumentFragment();
                messages.forEach(msg => {
                    const bubble = document.createElement('div');
                    bubble.className = msg.sender === 'A' ? 'chat-bubble-a' : 'chat-bubble-b';
                    const speakerLabel = msg.sender === 'A' ? '👤 Tel User' : '👤 Client';

                    if (msg.sender === 'A') {
                        const languageLabel = msg.lang ? ` (${msg.lang === 'hi' ? 'Hindi' : 'Japanese'})` : '';
                        const originalText = msg.input || '(no transcript)';
                        const translatedText = msg.output || '(no translation)';
                        const updatingIndicator = msg.isUpdating ? ' <span style="opacity: 0.6;">(listening...)</span>' : '';
                        bubble.innerHTML = `<b>${speakerLabel}${languageLabel}</b>${updatingIndicator}<br><i>${escapeHtml(originalText)}</i><br><b>${escapeHtml(translatedText)}</b>`;
                    } else {
                        const englishText = msg.text || '(no translation)';
                        const updatingIndicator = msg.isUpdating ? ' <span style="opacity: 0.6;">(updating...)</span>' : '';
                        bubble.innerHTML = `<b>${speakerLabel}</b><br>${escapeHtml(englishText)}${updatingIndicator}`;
                    }
                    fragment.appendChild(bubble);
                });

                // Single DOM update
                chatContainer.innerHTML = '';
                chatContainer.appendChild(fragment);
                chatContainer.scrollTop = chatContainer.scrollHeight;
            });
        }

        function initAudioGraph(stream) {
            if (!voiceVisualizer || visualizerBars.length === 0) return;
            if (!telAudioContext) return;
            if (analyserSource) {
                analyserSource.disconnect();
            }
            analyser = telAudioContext.createAnalyser();
            analyser.fftSize = 128;
            const source = telAudioContext.createMediaStreamSource(stream);
            source.connect(analyser);
            analyserSource = source;
            startVisualizer();
        }

        function startVisualizer() {
            if (!analyser) return;
            const bufferLength = analyser.frequencyBinCount;
            const dataArray = new Uint8Array(bufferLength);

            const renderFrame = () => {
                analyser.getByteFrequencyData(dataArray);
                visualizerBars.forEach((bar, index) => {
                    const value = dataArray[index % bufferLength] || 0;
                    const height = Math.max(10, (value / 255) * 110);
                    bar.style.height = `${height}px`;
                    bar.style.opacity = (value / 255) * 0.6 + 0.3;
                });
                visualizerAnimation = requestAnimationFrame(renderFrame);
            };

            voiceVisualizer.classList.add('active');
            if (visualizerAnimation) cancelAnimationFrame(visualizerAnimation);
            visualizerAnimation = requestAnimationFrame(renderFrame);
        }

        function stopVisualizerIfIdle() {
            if (isRecordingA) return;
            if (visualizerAnimation) {
                cancelAnimationFrame(visualizerAnimation);
                visualizerAnimation = null;
            }
            if (voiceVisualizer) {
                voiceVisualizer.classList.remove('active');
            }
            visualizerBars.forEach(bar => {
                bar.style.height = '12px';
                bar.style.opacity = '0.35';
            });
            if (analyserSource) {
                analyserSource.disconnect();
                analyserSource = null;
            }
        }
    </script>
</body>

</html>








