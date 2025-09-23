"""Voice assistant entry point using Hailo Whisper encoder (offline only)."""

import json
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pygame
import pyaudio
import serial
import sounddevice as sd
import torch
import noisereduce as nr
from scipy.io import wavfile as wav
from serial.tools import list_ports
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from vosk import KaldiRecognizer, Model

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


@dataclass
class AssistantConfig:
    use_hailo_encoder: bool = True
    hailo_encoder_hef: str = "base-whisper-encoder-5s_h8l.hef"
    hailo_window_seconds: int = 5
    decoder_local_dir: Optional[str] = None
    use_esp: bool = True


@dataclass
class ProgramState:
    last_command: str = "nyalakan_lampu"
    current_light_state: str = "ON"
    gender: str = "pria"
    predicted_language_from_wake_word: str = "Indonesian"


@dataclass
class SpeechModels:
    hailo_encoder: Optional[Any] = None
    hf_processor: Optional[Any] = None
    hf_decoder: Optional[Any] = None


@dataclass
class IntentModels:
    tokenizer: AutoTokenizer
    model: AutoModelForSequenceClassification
    id_to_label: Dict[int, str]


class AudioRingBuffer:
    def __init__(self, size_bytes: int = 300_000) -> None:
        self._size = size_bytes
        self._buffer = bytearray(size_bytes)
        self._write_pos = 0
        self._total_written = 0
        self._lock = threading.Lock()
        self._data_available = threading.Condition(self._lock)

    def write(self, data: bytes) -> None:
        if not data:
            return
        with self._lock:
            chunk_size = len(data)
            end_pos = self._write_pos + chunk_size
            if end_pos > self._size:
                bytes_until_end = self._size - self._write_pos
                self._buffer[self._write_pos : self._size] = data[:bytes_until_end]
                bytes_remaining = chunk_size - bytes_until_end
                self._buffer[0:bytes_remaining] = data[bytes_until_end:]
                self._write_pos = bytes_remaining
            else:
                self._buffer[self._write_pos:end_pos] = data
                self._write_pos = end_pos % self._size
            self._total_written += chunk_size
            self._data_available.notify_all()

    def wait_for_data(self, timeout: Optional[float] = None) -> bool:
        with self._lock:
            if self._total_written > 0:
                return True
            return self._data_available.wait(timeout=timeout)

    def read_available_since(self, last_total_bytes: int, wait_timeout: Optional[float] = None) -> Tuple[bytes, int]:
        with self._lock:
            if self._total_written == last_total_bytes:
                if wait_timeout is not None:
                    self._data_available.wait(timeout=wait_timeout)
                if self._total_written == last_total_bytes:
                    return b"", last_total_bytes
            available = self._total_written - last_total_bytes
            if available <= 0:
                return b"", last_total_bytes
            start_index = last_total_bytes % self._size
            end_index = (last_total_bytes + available) % self._size
            if end_index > start_index:
                data = bytes(self._buffer[start_index:end_index])
            else:
                data = bytes(self._buffer[start_index:] + self._buffer[:end_index])
            return data, last_total_bytes + available

    def snapshot_total_written(self) -> int:
        with self._lock:
            return self._total_written


@dataclass
class AssistantRuntime:
    esp: Optional[serial.Serial] = None
    wake_model: Optional[Model] = None
    ring_buffer: AudioRingBuffer = field(default_factory=AudioRingBuffer)


CONFIG = AssistantConfig()
STATE = ProgramState()
MODELS = SpeechModels()
RUNTIME = AssistantRuntime()
INTENT_MODELS: Optional[IntentModels] = None

try:
    pygame.mixer.init()
except pygame.error as e:
    print(f"Gagal inisialisasi pygame mixer: {e}. Feedback suara mungkin tidak berfungsi.")

this_file_dir = os.path.dirname(os.path.abspath(__file__))

# Preferred input device/channels (auto-detected; used to extract ReSpeaker channel 0)
PREFERRED_INPUT_DEVICE_INDEX: Optional[int] = None
PREFERRED_INPUT_CHANNELS: int = 1
# Effective samplerate chosen for the current input device/channels
EFFECTIVE_INPUT_SAMPLERATE: Optional[int] = None

# Which channel to extract from multi-channel devices.
# For some ReSpeaker firmwares the beamformed/noise-suppressed channel is index 7
# (0-based) when 8 channels are exposed. Allow override via env var.
EXTRACT_CHANNEL_INDEX: int = int(os.getenv("RESPEAKER_EXTRACT_CHANNEL", "0"))

# --- Definisi Label untuk Mode Offline ---
# Daftar label untuk model OFFLINE (yang sudah dilatih)
nlp_label_list_offline = [
    "nyalakan lampu merah", "nyalakan lampu hijau", "nyalakan lampu biru",
    "nyalakan lampu lavender", "nyalakan lampu magenta", "nyalakan lampu pink",
    "nyalakan lampu violet", "nyalakan lampu aqua", "nyalakan lampu kuning",
    "nyalakan lampu emas", "nyalakan lampu abu", "nyalakan mode senang",
    "nyalakan mode sad", "matikan lampu", "tidak relevan", "nyalakan lampu",
    "gender ke wanita", "gender ke pria", "fitur belum didukung", "turunkan kecerahan", "naikkan kecerahan"
]


# Inisialisasi model NLP
try:
    nlp_model_path = os.path.join(
        this_file_dir,
        "natural_language_processing/mdeberta-intent-classification-final",
    )
    tokenizer = AutoTokenizer.from_pretrained(nlp_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(nlp_model_path)
    model = model.eval().to("cpu")
    label_map = {idx: label for idx, label in enumerate(nlp_label_list_offline)}
    INTENT_MODELS = IntentModels(tokenizer=tokenizer, model=model, id_to_label=label_map)
    LOGGER.info("Model NLP berhasil dimuat.")
except Exception as e:
    print(f"Gagal memuat model NLP: {e}. Program mungkin tidak berfungsi dengan benar.")
    sys.exit(1)

# --- Inisialisasi Dataset Exact Match (Lapisan Baru) ---
exact_match_intent_dict = {}
try:
    exact_match_dataset_path = os.path.join(this_file_dir, "natural_language_processing/final_train_data.json")
    with open(exact_match_dataset_path, 'r', encoding='utf-8') as f:
        dataset_data = json.load(f)
    
    for item in dataset_data:
        # Normalisasi teks (lowercase, strip whitespace) untuk pencocokan yang lebih andal
        normalized_text = item['teks'].lower().strip()
        exact_match_intent_dict[normalized_text] = item['label']
    
    print(f"Berhasil memuat {len(exact_match_intent_dict)} entri ke kamus exact match.")
except FileNotFoundError:
    print(f"PERINGATAN: File dataset 'final_train_data.json' tidak ditemukan. Fitur exact match dinonaktifkan.")
except Exception as e:
    print(f"Gagal memuat dataset exact match: {e}. Fitur ini akan dinonaktifkan.")


# Inisialisasi STT (Hailo encoder + HF decoder) atau fallback ke faster-whisper
try:
    if CONFIG.use_hailo_encoder:
        from hailo_whisper import HailoWhisperEncoder
        from transformers import AutoProcessor, WhisperForConditionalGeneration

        MODELS.hailo_encoder = HailoWhisperEncoder(CONFIG.hailo_encoder_hef, float_output=True)

        if CONFIG.decoder_local_dir:
            MODELS.hf_processor = AutoProcessor.from_pretrained(CONFIG.decoder_local_dir)
            MODELS.hf_decoder = (
                WhisperForConditionalGeneration.from_pretrained(CONFIG.decoder_local_dir)
                .eval()
                .to("cpu")
            )
        else:
            print("FATAL: decoder_local_dir tidak diset. Untuk mode offline penuh, siapkan decoder lokal dan set CONFIG.decoder_local_dir.")
            sys.exit(1)

        LOGGER.info("STT siap: Hailo encoder (base, 5s) + HF decoder (CPU, lokal).")
    else:
        raise RuntimeError("CONFIG.use_hailo_encoder is False")
except Exception as e:
    print(f"FATAL: Gagal init Hailo encoder/decoder pipeline: {e}")
    sys.exit(1)


# --- Audio Buffer Management ---

def callback(indata, frames, time_info, status):
    """Stream callback that pushes audio chunks into the ring buffer."""
    if status and status.input_overflow:
        print("PERINGATAN: Input audio overflow! Data audio mungkin hilang.", file=sys.stderr)
    try:
        # If capturing multi-channel (e.g., ReSpeaker), extract configured channel
        if PREFERRED_INPUT_CHANNELS and PREFERRED_INPUT_CHANNELS > 1:
            # RawInputStream provides bytes; interpret as interleaved int16
            data_i16 = np.frombuffer(indata, dtype=np.int16)
            if data_i16.size % PREFERRED_INPUT_CHANNELS == 0:
                ch_idx = max(0, min(EXTRACT_CHANNEL_INDEX, PREFERRED_INPUT_CHANNELS - 1))
                chx = data_i16.reshape(-1, PREFERRED_INPUT_CHANNELS)[:, ch_idx]
                RUNTIME.ring_buffer.write(chx.tobytes())
            else:
                # Fallback: write raw if shape unexpected
                RUNTIME.ring_buffer.write(bytes(indata))
        else:
            # Mono path
            RUNTIME.ring_buffer.write(bytes(indata))
    except Exception as _exc:
        # On any parsing issue, fallback to raw write
        RUNTIME.ring_buffer.write(bytes(indata))


# --- Utility Functions ---

def find_respeaker_input_device_sd() -> Tuple[Optional[int], int]:
    """Find ReSpeaker input device using sounddevice; return (device_index, channels).
    Preference order: a dedicated 1-channel processed endpoint (often already beamformed),
    otherwise the multi-channel endpoint (6 or 8).
    """
    try:
        devices = sd.query_devices()
        default_in = sd.default.device[0] if isinstance(sd.default.device, (list, tuple)) else sd.default.device
        chosen_index: Optional[int] = None
        chosen_channels: int = 1

        # Prefer a ReSpeaker device. If multiple entries exist (e.g. 1ch and 8ch endpoints),
        # pick in this order: 1ch endpoint > highest channel count endpoint.
        respeaker_candidates: List[Tuple[int, int, str]] = []
        for idx, dev in enumerate(devices):
            name = str(dev.get('name', '')).lower()
            max_in = int(dev.get('max_input_channels', 0) or 0)
            if max_in <= 0:
                continue
            if 'respeaker' in name or 'seeed' in name or 'mic array' in name:
                respeaker_candidates.append((idx, max_in, name))

        if respeaker_candidates:
            one_ch = [c for c in respeaker_candidates if c[1] == 1]
            if one_ch:
                chosen_index, chosen_channels, _ = one_ch[0]
            else:
                chosen_index, chosen_channels, _ = sorted(respeaker_candidates, key=lambda t: t[1], reverse=True)[0]

        # Fallback to default input device
        if chosen_index is None:
            if isinstance(default_in, int) and default_in >= 0:
                try:
                    dev = devices[default_in]
                    chosen_index = default_in
                    max_in = int(dev.get('max_input_channels', 0) or 0)
                    chosen_channels = max_in if max_in > 0 else 1
                except Exception:
                    chosen_index = None
                    chosen_channels = 1
            else:
                chosen_index = None
                chosen_channels = 1

        # Do not cap here; choose specific extract channel later
        chosen_channels = max(chosen_channels, 1)
        return chosen_index, chosen_channels
    except Exception as _exc:
        return None, 1

# Internet connectivity no longer used (offline mode)

def find_esp32_port():
    ports = list_ports.comports()
    print("Mencari port ESP32...")
    for port in ports:
        if "ACM" in port.device.upper() or "USB" in port.device.upper() or \
           (port.manufacturer and "Silicon Labs" in port.manufacturer) or \
           (port.manufacturer and "wch.cn" in port.manufacturer) or \
           (port.description and "USB-SERIAL CH340" in port.description.upper()) or \
           (port.description and "CP210x" in port.description.upper()):
            print(f"ESP32 kemungkinan ditemukan di {port.device} ({port.description})")
            return port.device
    print("ESP32 tidak ditemukan. Pastikan terhubung dan driver terinstal.")
    return None

def check_microphones():
    p = None
    try:
        p = pyaudio.PyAudio()
        print("Mikrofon yang terdeteksi:")
        found_mic = False
        default_mic_index = -1
        try:
            default_mic_info = p.get_default_input_device_info()
            default_mic_index = default_mic_info['index']
            print(f"  Default Input Device: ID {default_mic_index}, Nama: {default_mic_info['name']}")
        except IOError:
            print("  Tidak ada default input device yang terkonfigurasi di sistem.")

        for i in range(p.get_device_count()):
            device_info = p.get_device_info_by_index(i)
            if device_info.get("maxInputChannels", 0) > 0:
                print(f"  ID: {i}, Nama: {device_info['name']}{' (DEFAULT)' if i == default_mic_index else ''}")
                if i == default_mic_index or \
                   "ReSpeaker" in device_info["name"] or \
                   "Realtek" in device_info["name"] or \
                   "Microphone Array" in device_info["name"] or \
                   "USB PnP Sound Device" in device_info["name"]:
                    if not found_mic:
                         print(f"    -> Mikrofon yang sesuai/default ditemukan: {device_info['name']}")
                    found_mic = True
        if not found_mic:
            print("PERINGATAN: Tidak ada mikrofon yang dikenali (ReSpeaker, Realtek, default, dll.) terdeteksi. Pastikan mic terhubung dan dipilih sebagai default.")
        return found_mic
    except Exception as e:
        print(f"Error saat memeriksa mikrofon dengan PyAudio: {e}")
        return False
    finally:
        if p:
            p.terminate()

def check_speakers():
    p = None
    try:
        p = pyaudio.PyAudio()
        print("Speaker yang terdeteksi:")
        default_speaker_index = -1
        try:
            default_speaker_info = p.get_default_output_device_info()
            default_speaker_index = default_speaker_info['index']
            print(f"  Default Output Device: ID {default_speaker_index}, Nama: {default_speaker_info['name']}")
        except IOError:
            print("  Tidak ada default output device yang terkonfigurasi di sistem.")

        for i in range(p.get_device_count()):
            device_info = p.get_device_info_by_index(i)
            if device_info.get("maxOutputChannels", 0) > 0:
                print(f"  ID: {i}, Nama: {device_info['name']}{' (DEFAULT)' if i == default_speaker_index else ''}")
    except Exception as e:
        print(f"Error saat memeriksa speaker dengan PyAudio: {e}")
    finally:
        if p:
            p.terminate()

def select_preferred_input_device():
    """Detect and set the preferred input device and channels.
    Stores results in PREFERRED_INPUT_DEVICE_INDEX and PREFERRED_INPUT_CHANNELS.
    """
    global PREFERRED_INPUT_DEVICE_INDEX, PREFERRED_INPUT_CHANNELS, EXTRACT_CHANNEL_INDEX
    # Environment override takes precedence if provided
    env_idx = os.getenv("RESPEAKER_INDEX") or os.getenv("AUDIO_INPUT_INDEX")
    env_ch = os.getenv("RESPEAKER_CHANNELS") or os.getenv("AUDIO_INPUT_CHANNELS")
    dev_idx: Optional[int] = None
    ch: int = 1
    try:
        devices = sd.query_devices()
        if env_idx is not None:
            try:
                idx_val = int(env_idx)
                if 0 <= idx_val < len(devices):
                    dev_idx = idx_val
                    max_in = int(devices[idx_val].get('max_input_channels', 0) or 0)
                    ch = max_in if max_in > 0 else 1
                else:
                    dev_idx = None
            except Exception:
                dev_idx = None
        if env_ch is not None:
            try:
                ch_val = int(env_ch)
                if ch_val > 0:
                    ch = ch_val
            except Exception:
                pass
        # If no valid env override, auto-detect
        if dev_idx is None:
            dev_idx, ch = find_respeaker_input_device_sd()
        # Clamp channels to device capability if known
        if dev_idx is not None and 0 <= dev_idx < len(devices):
            max_in = int(devices[dev_idx].get('max_input_channels', 0) or 0)
            if max_in > 0:
                ch = min(ch, max_in)
    except Exception:
        # On any failure, fallback to auto-detect without env
        dev_idx, ch = find_respeaker_input_device_sd()
    PREFERRED_INPUT_DEVICE_INDEX = dev_idx
    PREFERRED_INPUT_CHANNELS = ch if ch and ch > 0 else 1
    try:
        devices = sd.query_devices()
        name = None
        if dev_idx is not None and 0 <= dev_idx < len(devices):
            name = devices[dev_idx].get('name', 'Unknown')
        # Smart default: if 8+ channels and no override, assume channel 7 is processed beamformed
        if os.getenv("RESPEAKER_EXTRACT_CHANNEL") is None and PREFERRED_INPUT_CHANNELS >= 8:
            EXTRACT_CHANNEL_INDEX = 7
        if name:
            print(f"Input device terpilih: index={dev_idx}, nama='{name}', channels={PREFERRED_INPUT_CHANNELS}, extract_channel={EXTRACT_CHANNEL_INDEX}")
        else:
            print(f"Input device terpilih: default (None), channels={PREFERRED_INPUT_CHANNELS}, extract_channel={EXTRACT_CHANNEL_INDEX}")
    except Exception:
        print(f"Input device terpilih: index={PREFERRED_INPUT_DEVICE_INDEX}, channels={PREFERRED_INPUT_CHANNELS}, extract_channel={EXTRACT_CHANNEL_INDEX}")


def determine_working_input_config() -> Tuple[int, int]:
    """Determine a samplerate and channels that are valid for the selected input device.

    Tries the device's default samplerate first, then a set of common rates.
    If multi-channel is unsupported at that rate, fall back to mono.
    Returns (samplerate, channels).
    """
    global PREFERRED_INPUT_DEVICE_INDEX, PREFERRED_INPUT_CHANNELS
    try:
        # Prefer the selected device; fall back to default input device
        dev_arg = PREFERRED_INPUT_DEVICE_INDEX if PREFERRED_INPUT_DEVICE_INDEX is not None else None
        dev_info = sd.query_devices(dev_arg, 'input')
        default_sr_val = int(dev_info.get('default_samplerate', 16000) or 16000)
    except Exception:
        dev_arg = None
        default_sr_val = 16000

    # Try current requested channels first, then fall back to mono
    channel_candidates = [max(1, PREFERRED_INPUT_CHANNELS)]
    if 1 not in channel_candidates:
        channel_candidates.append(1)

    # Candidate samplerates to try
    sr_candidates: List[int] = []
    for s in [default_sr_val, 16000, 48000, 44100, 32000, 22050, 8000]:
        if s not in sr_candidates:
            sr_candidates.append(int(s))

    # Probe combinations using PortAudio's check function
    for ch in channel_candidates:
        for sr in sr_candidates:
            try:
                sd.check_input_settings(device=dev_arg, channels=ch, samplerate=sr, dtype='int16')
                # Found a working combo
                return int(sr), int(ch)
            except Exception:
                continue

    # As a last resort, return a very safe default
    return 16000, 1

def record_audio_dynamic(duration_min=3, duration_max=6, silence_duration=0.4, sampling_rate=16000):
    """Record audio until trailing silence using VAD (if available),
    with a lower minimum duration so we stop near end-of-speech.

    duration_min: minimum seconds to capture once speech is present
    duration_max: hard cap on capture seconds
    silence_duration: trailing non-speech required to stop (seconds)
    sampling_rate: stream sample rate
    """
    try:
        import webrtcvad  # type: ignore
        HAVE_VAD = True
    except Exception:
        HAVE_VAD = False

    # Use small chunks for better endpointing; VAD prefers 10/20/30ms frames
    chunk_duration = 0.02 if HAVE_VAD else 0.1
    chunk_samples = int(chunk_duration * sampling_rate)
    chunks: List[np.ndarray] = []
    min_samples = int(duration_min * sampling_rate)
    max_samples = int(duration_max * sampling_rate)
    silence_threshold = 0.01  # fallback for energy-based VAD
    print("Mulai merekam (dinamis)...")
    try:
        # Use preferred device/channels; extract selected channel on multi-channel devices (e.g., ReSpeaker)
        # Ensure the provided samplerate/channels are valid; if not, fall back
        use_sr, use_ch = sampling_rate, max(1, PREFERRED_INPUT_CHANNELS)
        try:
            sd.check_input_settings(device=PREFERRED_INPUT_DEVICE_INDEX if PREFERRED_INPUT_DEVICE_INDEX is not None else None,
                                    channels=use_ch, samplerate=use_sr, dtype='float32')
        except Exception:
            # Fallback to a safe config
            use_sr, use_ch = determine_working_input_config()
            print(f"Perekaman dinamis: fallback ke samplerate={use_sr}, channels={use_ch}")

        # Ensure chunk size matches the effective samplerate
        chunk_samples = int(chunk_duration * use_sr)
        # Configure VAD if available and supported samplerate
        vad = None
        vad_supported = HAVE_VAD and use_sr in (8000, 16000, 32000, 48000)
        if vad_supported:
            try:
                vad = webrtcvad.Vad(2)  # 0-3 (3 = most aggressive)
            except Exception:
                vad = None
                vad_supported = False

        with sd.InputStream(
            samplerate=use_sr,
            channels=use_ch,
            dtype='float32',
            blocksize=chunk_samples,
            device=PREFERRED_INPUT_DEVICE_INDEX,
        ) as stream:
            start_time = time.time()
            # Measure ambient level first 200ms to set a dynamic threshold for fallback VAD
            ambient_frames = []
            ambient_collect = int(max(1, int(0.2 / chunk_duration)))
            try:
                for _ in range(ambient_collect):
                    amb, _ovf = stream.read(chunk_samples)
                    ambient_frames.append(amb.copy())
                ambient = np.concatenate(ambient_frames, axis=0)
                if use_ch > 1 and ambient.ndim == 2 and ambient.shape[1] >= 1:
                    ambient_mono = ambient[:, min(EXTRACT_CHANNEL_INDEX, ambient.shape[1]-1)].astype(np.float32)
                else:
                    ambient_mono = ambient.flatten()
                ambient_level = float(np.mean(np.abs(ambient_mono)))
                silence_threshold = max(0.015, ambient_level * 3.0)
            except Exception:
                pass

            # Pre-roll to avoid cutting initial consonants
            pre_roll: List[np.ndarray] = []
            pre_roll_seconds = 0.2
            max_pre_frames = int(pre_roll_seconds / chunk_duration)
            voiced_started = False
            voiced_last_time = time.time()

            # Main loop: start writing once voice is detected; stop after min_samples and trailing silence
            while True:
                now = time.time()
                if now - start_time > duration_max + 2.0:
                    print("Perekaman dinamis melebihi batas waktu maksimum, berhenti.")
                    break
                try:
                    chunk, overflowed = stream.read(chunk_samples)
                    if overflowed:
                        print("PERINGATAN: Input audio overflow saat merekam perintah (loop).", file=sys.stderr)
                except sd.CallbackStop:
                    print("Perekaman dinamis dihentikan oleh callback.")
                    break
                except Exception as e:
                    print(f"Error saat membaca stream: {e}")
                    break

                # Extract selected channel
                if use_ch > 1 and chunk.ndim == 2 and chunk.shape[1] >= 1:
                    ch_idx = max(0, min(EXTRACT_CHANNEL_INDEX, chunk.shape[1] - 1))
                    mono = chunk[:, ch_idx].astype(np.float32)
                else:
                    mono = chunk.flatten()

                # VAD decision
                is_voiced = False
                if vad_supported and vad is not None:
                    # Convert to 16-bit PCM for VAD, per-frame
                    bytes16 = np.clip(mono * 32768.0, -32768, 32767).astype(np.int16).tobytes()
                    is_voiced = vad.is_speech(bytes16, use_sr)
                else:
                    is_voiced = float(np.mean(np.abs(mono))) >= silence_threshold

                # Maintain pre-roll buffer until voice starts
                if not voiced_started:
                    pre_roll.append(mono)
                    if len(pre_roll) > max_pre_frames:
                        pre_roll.pop(0)
                    if is_voiced:
                        voiced_started = True
                        voiced_last_time = now
                        # Flush pre-roll
                        for f in pre_roll:
                            chunks.append(f)
                        pre_roll.clear()
                    continue

                # Append chunk
                chunks.append(mono)
                if is_voiced:
                    voiced_last_time = now

                # Stop if we have recorded at least min_samples and observed trailing silence
                if int(sum(len(c) for c in chunks)) >= min_samples:
                    if (now - voiced_last_time) >= silence_duration:
                        print("Deteksi akhir ucapan (silence), menghentikan perekaman dinamis.")
                        break

                # Hard cap
                if int(sum(len(c) for c in chunks)) >= max_samples:
                    print("Mencapai durasi maksimum perekaman dinamis, berhenti.")
                    break
        print("Selesai merekam (dinamis).")
        if not chunks:
            return np.array([])
        return np.concatenate(chunks, axis=0)
    except sd.PortAudioError as e:
        print(f"Error SoundDevice saat merekam (dinamis): {e}")
        return np.array([])
    except Exception as e:
        print(f"Error tak terduga saat merekam (dinamis): {e}")
        return np.array([])

def record_audio(duration=2, sampling_rate=16000, noise_reduce=True, dynamic=True):
    if dynamic:
        audio_data = record_audio_dynamic(duration_min=duration, duration_max=5, silence_duration=1, sampling_rate=sampling_rate)
    else:
        print(f"Mulai merekam (durasi tetap: {duration} detik)...")
        try:
            # Validate samplerate for the selected device; fall back if needed
            try:
                sd.check_input_settings(device=PREFERRED_INPUT_DEVICE_INDEX if PREFERRED_INPUT_DEVICE_INDEX is not None else None,
                                        channels=max(1, PREFERRED_INPUT_CHANNELS), samplerate=sampling_rate, dtype='float32')
                effective_sr = int(sampling_rate)
            except Exception:
                effective_sr, _effective_ch = determine_working_input_config()
                print(f"Perekaman tetap: fallback ke samplerate={effective_sr}")

            num_samples = int(duration * effective_sr)
            recording = sd.rec(
                num_samples,
                samplerate=effective_sr,
                channels=max(1, PREFERRED_INPUT_CHANNELS),
                dtype='float32',
                device=PREFERRED_INPUT_DEVICE_INDEX,
            )
            sd.wait()
            if PREFERRED_INPUT_CHANNELS > 1 and recording.ndim == 2 and recording.shape[1] >= 1:
                audio_data = recording[:, 0].astype(np.float32)
            else:
                audio_data = recording.flatten()
            print("Selesai merekam (durasi tetap).")
        except sd.PortAudioError as e:
            print(f"Error SoundDevice saat merekam (fixed): {e}")
            return np.array([])
        except Exception as e:
            print(f"Error tak terduga saat merekam (fixed): {e}")
            return np.array([])

    if audio_data.size == 0:
        return np.array([])

    if noise_reduce and audio_data.size > int(0.5 * sampling_rate): 
        noise_sample_duration = min(0.5, audio_data.size / sampling_rate * 0.2) 
        noise_sample_samples = int(noise_sample_duration * sampling_rate)
        if noise_sample_samples > 0:
             noise_sample = audio_data[:noise_sample_samples]
             audio_data = reduce_noise(audio_data, noise_sample, sampling_rate)
        else:
            print("Audio terlalu pendek untuk mengumpulkan sampel noise.")
    elif noise_reduce:
        print("Audio terlalu pendek untuk noise reduction, dilewati.")
    return audio_data

def reduce_noise(audio, noise_sample, sr_rate, prop_decrease=0.8):
    try:
        if noise_sample.size == 0 or audio.size <= noise_sample.size: 
            return audio
        return nr.reduce_noise(
            y=audio,
            y_noise=noise_sample,
            sr=sr_rate,
            prop_decrease=prop_decrease,
            stationary=True 
        )
    except Exception as e:
        print(f"Error dalam noise reduction: {e}")
        return audio

def save_audio(audio_array, filename, sampling_rate=16000):
    try:
        if audio_array.size == 0:
            print(f"Tidak ada data audio untuk disimpan ke {filename}.")
            return
        max_abs_val = np.max(np.abs(audio_array))
        if max_abs_val == 0: 
             audio_normalized = np.zeros_like(audio_array, dtype=np.int16)
        else:
            audio_normalized = np.int16(audio_array * 32767 / max_abs_val)
        wav.write(filename, sampling_rate, audio_normalized)
        print(f"Audio disimpan ke {filename}")
    except Exception as e:
        print(f"Gagal menyimpan audio ke {filename}: {e}")

def is_audio_present(audio_array_float32, threshold=0.003):
    if audio_array_float32.size == 0:
        print("Audio array kosong, tidak ada audio terdeteksi.")
        return False
    audio_value = np.mean(np.abs(audio_array_float32))
    print(f"Audio presence check value: {audio_value} (Threshold: {threshold})")
    return audio_value > threshold

def _fit_to_hailo_window(feats_np, window_seconds):
    # feats_np: (1, 80, T_any). Whisper uses 3000 frames for 30s.
    import numpy as np
    target_T = int(3000 * window_seconds / 30)  # 5s -> 500
    T = feats_np.shape[2]
    if T == target_T:
        return feats_np
    if T > target_T:
        return feats_np[:, :, :target_T]
    pad = np.zeros((1, 80, target_T - T), dtype=feats_np.dtype)
    return np.concatenate([feats_np, pad], axis=2)

def transcribe_audio(audio_array_float32, sampling_rate=16000, language="Indonesian"):
    """Transcribe audio either via Hailo encoder or the faster-whisper fallback."""
    if (
        MODELS.hailo_encoder is not None
        and MODELS.hf_processor is not None
        and MODELS.hf_decoder is not None
    ):
        try:
            import time
            import numpy as np
            import torch
            import transformers
            from packaging import version
            from transformers.modeling_outputs import BaseModelOutput

            if audio_array_float32.size == 0:
                return {'text': '', 'processing_time': 0}

            start = time.time()
            processor = MODELS.hf_processor
            decoder = MODELS.hf_decoder
            inputs = processor(audio_array_float32, sampling_rate=sampling_rate, return_tensors="np")
            feats = inputs["input_features"].astype(np.float32)

            if feats.size == 0 or feats.shape[-1] == 0:
                feats = np.zeros(
                    (1, 80, int(3000 * CONFIG.hailo_window_seconds / 30)),
                    dtype=np.float32,
                )
            feats = _fit_to_hailo_window(feats, CONFIG.hailo_window_seconds)
            feats = np.ascontiguousarray(feats, dtype=np.float32)

            enc_np = MODELS.hailo_encoder.encode(feats)
            enc_t = torch.from_numpy(enc_np).to("cpu")
            enc_out = BaseModelOutput(last_hidden_state=enc_t)

            lang_map = {"Indonesian": "indonesian", "English": "english", "Japanese": "japanese"}
            lang_name = lang_map.get(language, "indonesian")
            gen_kwargs = dict(
                max_new_tokens=48,
                num_beams=1,
                no_repeat_ngram_size=3,
                repetition_penalty=1.15,
                length_penalty=1.0,
                early_stopping=True,
                do_sample=False,
            )

            if version.parse(transformers.__version__) >= version.parse("4.38.0"):
                gen_kwargs.update({"task": "transcribe", "language": lang_name})
            else:
                gen_kwargs.update(
                    {
                        "forced_decoder_ids": processor.get_decoder_prompt_ids(
                            language=lang_name, task="transcribe"
                        )
                    }
                )

            decoder.generation_config.eos_token_id = decoder.config.eos_token_id
            decoder.generation_config.pad_token_id = decoder.config.eos_token_id
            decoder.generation_config.do_sample = False
            decoder.generation_config.temperature = 0.0

            with torch.no_grad():
                gen_ids = decoder.generate(encoder_outputs=enc_out, **gen_kwargs)

            text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
            return {'text': text, 'processing_time': time.time() - start}
        except Exception as e:
            print(f"Error Hailo path: {e}")
    return {'text': '', 'processing_time': 0}


def predict_intent(text):
    def predict_offline(text_input):
        if INTENT_MODELS is None:
            print("Model NLP belum dimuat. Kembali ke intent 'tidak relevan'.")
            return "tidak relevan"
        try:
            tokenizer = INTENT_MODELS.tokenizer
            model = INTENT_MODELS.model
            label_map = INTENT_MODELS.id_to_label
            inputs = tokenizer(
                text_input,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to("cpu")
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits
            pred_id = torch.argmax(logits, dim=-1).item()
            return label_map[pred_id]
        except Exception as e:
            print(f"Error saat prediksi intent offline: {e}")
            return "tidak relevan"
    return predict_offline(text)

def send_to_esp(command: str) -> bool:
    """Send command to ESP device if configured."""
    if not CONFIG.use_esp:
        return True
    try:
        esp = RUNTIME.esp
        if esp is not None and esp.is_open:
            esp.write((command + "\n").encode())
            return True

        print("Port serial ESP tidak terbuka atau ESP belum diinisialisasi.")
        esp_port = find_esp32_port()
        if not esp_port:
            print("Tidak dapat menemukan port ESP32 untuk re-koneksi.")
            return False

        try:
            esp = serial.Serial(esp_port, 115200, timeout=1)
            RUNTIME.esp = esp
            print(f"Berhasil terhubung ke ESP32 di {esp_port}.")
            esp.write((command + "\n").encode())
            return True
        except serial.SerialException as serial_exc:
            print(f"Gagal re-koneksi ke ESP32: {serial_exc}")
            RUNTIME.esp = None
            return False
    except serial.SerialException as exc:
        print(f"Gagal mengirim ke ESP: {exc}. Port mungkin terputus.")
        RUNTIME.esp = None
        return False
    except Exception as exc:
        print(f"Error tak terduga saat mengirim ke ESP: {exc}")
        return False


def intent_feedback(intent, predicted_language="Indonesian", main_loop_flag=None):
    if intent == STATE.last_command and intent not in ["wake", "off", "system_on"]:
        print(f"Intent '{intent}' sudah dieksekusi sebelumnya dan bukan wake/off/system_on. Mengabaikan feedback audio & ESP.")
        return

    feedback_audio_base_path = os.path.join(this_file_dir, "feedback_audio")
    colors = ["merah", "hijau", "biru", "lavender", "magenta", "pink", "violet", "aqua", "kuning", "emas", "abu"]
    audio_file_to_play = None

    lang_path = os.path.join(feedback_audio_base_path, predicted_language)
    if not os.path.isdir(lang_path):
        print(f"Peringatan: Direktori feedback audio '{predicted_language}' tidak ditemukan. Fallback ke 'Indonesian'.")
        lang_path = os.path.join(feedback_audio_base_path, "Indonesian")
        if not os.path.isdir(lang_path):
            print(f"Peringatan: Direktori feedback audio default 'Indonesian' juga tidak ditemukan.")
            return 

    original_last_command = STATE.last_command 

    if "nyalakan lampu" in intent:
        real_intent_color = None
        for color in colors:
            if color in intent.lower():
                real_intent_color = color
                break
        if real_intent_color:
            if send_to_esp(real_intent_color.upper()):
                suffix = "_pria" if STATE.gender == "pria" else ""
                audio_file_to_play = os.path.join(lang_path, f"ganti_warna{suffix}.mp3")
                STATE.current_light_state = real_intent_color.upper()
                STATE.last_command = intent
        elif intent == "nyalakan lampu": 
            if send_to_esp("ON"):
                suffix = "_pria" if STATE.gender == "pria" else ""
                audio_file_to_play = os.path.join(lang_path, f"menyalakan_lampu{suffix}.mp3")
                STATE.current_light_state = "ON"
                STATE.last_command = intent
    elif "matikan lampu" in intent:
        if send_to_esp("OFF"):
            suffix = "_pria" if STATE.gender == "pria" else ""
            audio_file_to_play = os.path.join(lang_path, f"mematikan_lampu{suffix}.mp3")
            STATE.current_light_state = "OFF"
            STATE.last_command = intent
    elif "nyalakan mode senang" in intent:
        if send_to_esp("HAPPY"):
            suffix = "_pria" if STATE.gender == "pria" else ""
            audio_file_to_play = os.path.join(lang_path, f"senang{suffix}.mp3")
            STATE.last_command = intent
    elif "nyalakan mode sad" in intent:
        if send_to_esp("SAD"):
            suffix = "_pria" if STATE.gender == "pria" else ""
            audio_file_to_play = os.path.join(lang_path, f"sedih{suffix}.mp3")
            STATE.last_command = intent
    elif "gender ke wanita" in intent:
        STATE.gender = "wanita"
        audio_file_to_play = os.path.join(lang_path, "ganti_suara_ke_wanita.mp3") 
        print("Gender diubah ke wanita.")
        STATE.last_command = intent
    elif "gender ke pria" in intent:
        STATE.gender = "pria"
        audio_file_to_play = os.path.join(lang_path, "ganti_suara_ke_pria.mp3") 
        print("Gender diubah ke pria.")
        STATE.last_command = intent
    elif "fitur belum didukung" in intent:
        suffix = "_pria" if STATE.gender == "pria" else ""
        audio_file_to_play = os.path.join(lang_path, f"fitur_belum_didukung{suffix}.mp3")
        STATE.last_command = intent
    
    elif "wake" in intent: 
        audio_file_to_play = os.path.join(feedback_audio_base_path, "ping_berbicara.mp3")
        pygame.mixer.music.load(audio_file_to_play)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy() and (main_loop_flag is None or main_loop_flag.is_set()): 
            time.sleep(0.05)
        if main_loop_flag is not None and not main_loop_flag.is_set(): 
            pygame.mixer.music.stop()

        suffix = "_pria" if STATE.gender == "pria" else ""
        audio_file_to_play = os.path.join(lang_path, f"berbicara{suffix}.mp3")
        STATE.last_command = intent 
    elif "off" in intent: 
        audio_file_to_play = os.path.join(feedback_audio_base_path, "off_to_wakeword.mp3")
        STATE.last_command = intent 
    elif "system_on" in intent:
        audio_file_to_play = os.path.join(feedback_audio_base_path, "system_on.mp3")
        STATE.last_command = intent
    elif "turunkan kecerahan" in intent:
        if send_to_esp("turunkan brightness"):
            suffix = "_pria" if STATE.gender == "pria" else ""
            audio_file_to_play = os.path.join(lang_path, f"turunkan_kecerahan{suffix}.mp3")
            STATE.last_command = intent
    elif "naikkan kecerahan" in intent:
        if send_to_esp("naikkan brightness"):
            suffix = "_pria" if STATE.gender == "pria" else ""
            audio_file_to_play = os.path.join(lang_path, f"naikkan_kecerahan{suffix}.mp3")
            STATE.last_command = intent
    elif "tidak relevan" in intent:
        suffix = "_pria" if STATE.gender == "pria" else ""
        print("Perintah tidak dikenali atau tidak ada suara signifikan.")

    if audio_file_to_play:
        if os.path.exists(audio_file_to_play):
            try:
                if pygame.mixer.music.get_busy():
                    pygame.mixer.music.stop()
                    pygame.mixer.music.unload()
                    time.sleep(0.05) 

                pygame.mixer.music.load(audio_file_to_play)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy() and (main_loop_flag is None or main_loop_flag.is_set()): 
                    time.sleep(0.05)
                if main_loop_flag is not None and not main_loop_flag.is_set(): 
                     pygame.mixer.music.stop()
            except pygame.error as e:
                print(f"Gagal memainkan file audio '{audio_file_to_play}': {e}")
            except Exception as e:
                print(f"Error tak terduga saat memainkan audio: {e}")
        else:
            print(f"File audio feedback tidak ditemukan: {audio_file_to_play}")
    elif intent != "tidak relevan" and original_last_command == intent:
        pass


def initialize_system(main_loop_flag=None):
    """Initialise hardware, wake word model, and base state."""
    print("Inisialisasi sistem...")
    STATE.last_command = "system_booting"

    if CONFIG.use_esp:
        esp_port = find_esp32_port()
        if esp_port:
            try:
                RUNTIME.esp = serial.Serial(esp_port, 115200, timeout=1)
                print(f"Berhasil terhubung ke ESP32 di {esp_port}")
                time.sleep(1.5)
            except serial.SerialException as exc:
                print(f"Gagal terhubung ke ESP32: {exc}. Mode tanpa ESP akan digunakan jika memungkinkan.")
                RUNTIME.esp = None
            except Exception as exc:
                print(f"Error tak terduga saat inisialisasi ESP32: {exc}")
                RUNTIME.esp = None
        else:
            print("ESP32 tidak terdeteksi.")
            RUNTIME.esp = None
    else:
        print("Mode tanpa ESP32 diaktifkan.")
        RUNTIME.esp = None

    try:
        wake_model_path_name = "vosk-model-en-us-0.22-lgraph"
        wake_model_path = os.path.join(this_file_dir, "vosk_models", wake_model_path_name)
        if not os.path.exists(wake_model_path):
            print(
                f"PERINGATAN: Model Vosk di path '{wake_model_path}' tidak ditemukan. Mencoba memuat dengan nama..."
            )
            RUNTIME.wake_model = Model(model_name=wake_model_path_name)
        else:
            RUNTIME.wake_model = Model(model_path=wake_model_path)
        print("Model Vosk loaded successfully.")
    except Exception as exc:
        print(
            f"FATAL: Gagal memuat model Vosk wake word: {exc}. Pastikan model ada di direktori yang benar atau nama model valid. Program berhenti."
        )
        sys.exit(1)

    if not check_microphones():
        print(
            "PERINGATAN PENTING: Mikrofon yang sesuai tidak terdeteksi atau tidak ada default. Program mungkin tidak dapat menerima input suara."
        )
    check_speakers()

    # Select preferred input device (ReSpeaker if available) and channels
    select_preferred_input_device()
    intent_feedback("system_on", main_loop_flag=main_loop_flag)
    print("Sistem siap.")
    return True

SAMPLE_RATE = 16000

def run_vosk_wake_word_detector(wake_event, vosk_recognizer, main_loop_flag, samplerate, result_language_container):
    print("Vosk wake word detector thread dimulai.")
    ring_buffer = RUNTIME.ring_buffer
    try:
        if not ring_buffer.wait_for_data(timeout=5.0):
            print("Vosk thread menunggu data audio pertama...")
        current_pos = ring_buffer.snapshot_total_written()

        while not wake_event.is_set() and main_loop_flag.is_set():
            chunk_data, current_pos = ring_buffer.read_available_since(
                current_pos, wait_timeout=0.1
            )
            if wake_event.is_set() or not main_loop_flag.is_set():
                break
            if not chunk_data:
                continue

            if vosk_recognizer.AcceptWaveform(chunk_data):
                result_json = vosk_recognizer.Result()
                result = json.loads(result_json)
                if 'text' in result and result['text']:
                    detected_text = result['text'].lower().strip()
                    lang = None
                    if ("oke toyota" in detected_text or ("oke" in detected_text and "toyota" in detected_text)) or ("okay toyota" in detected_text):
                        lang = "Indonesian"
                    elif "hello toyota" in detected_text or ("hello" in detected_text and "toyota" in detected_text):
                        lang = "English"
                    elif ("hai toyota" in detected_text or ("hai" in detected_text and "toyota" in detected_text)) or ("moshi moshi" in detected_text):
                        lang = "Japanese"

                    if lang:
                        print(f"Vosk mendeteksi wake word! Bahasa: {lang} (Teks: '{detected_text}')")
                        if not wake_event.is_set():
                            wake_event.set()
                            result_language_container['language'] = lang
            else:
                partial = vosk_recognizer.PartialResult()
                if partial:
                    _ = partial  # placeholder to avoid unused variable warning
    except Exception as exc:
        print(f"Error pada Vosk wake word detector: {exc}")


def process_command_audio_in_thread(audio_float32_data, language, sampling_rate, relevant_command_event, main_loop_flag_ref):
    thread_id = threading.get_ident()

    if not main_loop_flag_ref.is_set():
        print(f"Thread ID {thread_id}: Program shutting down. Exiting command thread.")
        return
    if relevant_command_event.is_set():
        print(f"Thread ID {thread_id}: Relevant command already processed by another thread. Exiting early.")
        return

    text_transcribed = ""
    stt_processing_time = 0

    # --- STT Phase (offline, Hailo encoder + HF decoder) ---
    if not main_loop_flag_ref.is_set() or relevant_command_event.is_set(): return
    offline_result_cmd = transcribe_audio(audio_float32_data, sampling_rate=sampling_rate, language=language)
    text_transcribed = offline_result_cmd["text"]
    stt_processing_time = offline_result_cmd["processing_time"]

    if not main_loop_flag_ref.is_set() or relevant_command_event.is_set():
        print(f"Thread ID {thread_id}: Exiting after STT phase due to shutdown or relevant command event.")
        return

    if not text_transcribed.strip():
        print(f"Thread ID {thread_id}: Tidak ada teks perintah yang berhasil ditranskripsi.")
        if not relevant_command_event.is_set():
            intent_feedback("tidak relevan", language, main_loop_flag=main_loop_flag_ref)
        return 

    print(f"Thread ID {thread_id}: Transkripsi perintah: '{text_transcribed}' (Waktu STT: {stt_processing_time:.2f} dtk, Bahasa: {language})")

    # --- Intent Prediction Phase (with Exact Match Layer) ---
    if not main_loop_flag_ref.is_set() or relevant_command_event.is_set():
        print(f"Thread ID {thread_id}: Exiting before Intent Prediction due to shutdown or relevant command event.")
        return
    
    predicted_intent = None
    normalized_text = text_transcribed.lower().strip()

    # 1. Cek di kamus exact match terlebih dahulu
    if exact_match_intent_dict:
        predicted_intent = exact_match_intent_dict.get(normalized_text)

    if predicted_intent:
        print(f"Thread ID {thread_id}: Intent ditemukan via exact match: '{predicted_intent}'. Melewati NLP.")
    else:
        # 2. Jika tidak ada, baru panggil model NLP (online/offline)
        print(f"Thread ID {thread_id}: Tidak ada exact match. Memanggil model NLP...")
        predicted_intent = predict_intent(text_transcribed)
    
    print(f"Thread ID {thread_id}: Prediksi intent final: '{predicted_intent}'")

    # --- Feedback Phase & Setting Relevant Event ---
    if not main_loop_flag_ref.is_set(): 
        print(f"Thread ID {thread_id}: Exiting before feedback due to shutdown.")
        return
    
    is_this_thread_winner = False
    if predicted_intent != "tidak relevan":
        if not relevant_command_event.is_set():
            relevant_command_event.set()
            is_this_thread_winner = True
            print(f"Thread ID {thread_id}: Perintah relevan '{predicted_intent}' diproses. Menandai untuk keluar dari mode perintah (WINNER).")
        else:
            print(f"Thread ID {thread_id}: Perintah relevan '{predicted_intent}' juga ditemukan, tapi event sudah diset oleh thread lain.")
    
    play_this_feedback = False
    if is_this_thread_winner:
        play_this_feedback = True
    elif predicted_intent != "tidak relevan":
        play_this_feedback = True
    elif predicted_intent == "tidak relevan" and not relevant_command_event.is_set():
        play_this_feedback = True
    
    if play_this_feedback:
        intent_feedback(predicted_intent, language, main_loop_flag=main_loop_flag_ref)
    else:
        print(f"Thread ID {thread_id}: Suppressing 'tidak relevan' feedback as another relevant command was processed or this thread lost the race with a relevant command.")


# --- MAIN PROGRAM ---


def main() -> None:
    main_loop_active_flag = threading.Event()
    main_loop_active_flag.set()

    if not initialize_system(main_loop_flag=main_loop_active_flag):
        print("FATAL: Gagal melakukan inisialisasi sistem. Program berhenti.")
        sys.exit(1)

    # Determine a working samplerate/channels for the selected input device
    try:
        sr, ch = determine_working_input_config()
        # Update globals for consistency
        global EFFECTIVE_INPUT_SAMPLERATE, PREFERRED_INPUT_CHANNELS
        EFFECTIVE_INPUT_SAMPLERATE = int(sr)
        PREFERRED_INPUT_CHANNELS = int(ch)
        print(f"Konfigurasi audio input: samplerate={EFFECTIVE_INPUT_SAMPLERATE} Hz, channels={PREFERRED_INPUT_CHANNELS}.")
        samplerate = EFFECTIVE_INPUT_SAMPLERATE
    except Exception as exc:
        print(f"PERINGATAN: Gagal menentukan konfigurasi input yang valid: {exc}. Menggunakan default 16000 Hz, mono.")
        samplerate = 16000
        PREFERRED_INPUT_CHANNELS = 1

    audio_stream = None
    vosk_thread = None
    online_check_thread = None  # removed (offline mode)
    active_command_threads: List[threading.Thread] = []

    try:
        while main_loop_active_flag.is_set():
            print("\nMenunggu wake word (Offline Vosk kontinu, Online Google STT periodik [id-ID])...")
            wake_word_detected_event = threading.Event()
            detected_language_container = {'language': "Indonesian"}
            RUNTIME.ring_buffer = AudioRingBuffer()

            callback_chunk_duration_sec = 0.2
            blocksize_callback_samples = int(samplerate * callback_chunk_duration_sec)

            try:
                # Validate the streaming settings before opening
                try:
                    sd.check_input_settings(
                        device=PREFERRED_INPUT_DEVICE_INDEX if PREFERRED_INPUT_DEVICE_INDEX is not None else None,
                        channels=max(1, PREFERRED_INPUT_CHANNELS),
                        samplerate=samplerate,
                        dtype='int16',
                    )
                except Exception:
                    # Re-determine working config on-the-fly (device may change)
                    samplerate, PREFERRED_INPUT_CHANNELS = determine_working_input_config()
                    print(f"Stream wake word: fallback ke samplerate={samplerate}, channels={PREFERRED_INPUT_CHANNELS}")

                audio_stream = sd.RawInputStream(
                    samplerate=samplerate,
                    blocksize=int(samplerate * callback_chunk_duration_sec),
                    device=PREFERRED_INPUT_DEVICE_INDEX,
                    dtype="int16",
                    channels=max(1, PREFERRED_INPUT_CHANNELS),
                    callback=callback,
                )
                audio_stream.start()
                print("Stream audio dimulai untuk deteksi wake word.")
            except sd.PortAudioError as port_audio_exc:
                if not main_loop_active_flag.is_set():
                    break
                print(f"SoundDevice Error saat memulai stream: {port_audio_exc}. Mencoba lagi dalam 2 detik...")
                time.sleep(2)
                continue
            except Exception as stream_exc:
                if not main_loop_active_flag.is_set():
                    break
                print(f"Error tak terduga saat memulai stream audio: {stream_exc}. Mencoba lagi dalam 2 detik...")
                time.sleep(2)
                continue

            vosk_grammar = json.dumps(
                ["hello", "oke", "okay", "hai", "toyota", "moshi", "one", "two", "three", "four", "[unk]"],
                ensure_ascii=False,
            )
            vosk_recognizer = KaldiRecognizer(RUNTIME.wake_model, samplerate, vosk_grammar)

            vosk_thread = threading.Thread(
                target=run_vosk_wake_word_detector,
                args=(wake_word_detected_event, vosk_recognizer, main_loop_active_flag, samplerate, detected_language_container),
                daemon=True,
            )
            vosk_thread.start()

            while not wake_word_detected_event.is_set() and main_loop_active_flag.is_set():
                time.sleep(0.05)

            if audio_stream and audio_stream.active:
                print("Menghentikan stream audio wake word...")
                audio_stream.stop()
            if audio_stream and not audio_stream.closed:
                audio_stream.close()
                audio_stream = None
                print("Stream audio wake word ditutup.")

            wake_word_detected_event.set()

            if vosk_thread and vosk_thread.is_alive():
                print("Menunggu Vosk thread selesai...")
                vosk_thread.join(timeout=2.0)
                if vosk_thread.is_alive():
                    print("PERINGATAN: Vosk thread tidak berhenti tepat waktu setelah stop stream.")

            if detected_language_container.get('language') and main_loop_active_flag.is_set():
                STATE.predicted_language_from_wake_word = detected_language_container['language']
                print(f"Wake word terdeteksi! Bahasa yang digunakan: {STATE.predicted_language_from_wake_word}")
                intent_feedback("wake", STATE.predicted_language_from_wake_word, main_loop_flag=main_loop_active_flag)

                command_mode_timeout = 15
                current_predicted_language = STATE.predicted_language_from_wake_word

                relevant_command_processed_this_session = threading.Event()
                command_session_start_time = time.time()

                active_command_threads = [thread for thread in active_command_threads if thread.is_alive()]

                while main_loop_active_flag.is_set():
                    if relevant_command_processed_this_session.is_set():
                        print("Perintah relevan terdeteksi dan diproses. Keluar dari mode perintah sesi ini.")
                        break

                    current_loop_time = time.time()
                    if current_loop_time - command_session_start_time >= command_mode_timeout:
                        print(f"Waktu mode perintah ({command_mode_timeout} detik) habis.")
                        break

                    remaining_time = command_mode_timeout - (current_loop_time - command_session_start_time)
                    print(f"\nSilahkan ucapkan perintah (Bahasa: {current_predicted_language}, Sisa waktu: {remaining_time:.0f} detik)...")
                    # Use the same effective samplerate for command recording
                    command_sampling_rate = samplerate
                    recorded_audio_cmd_float32 = record_audio(
                        duration=1.5,
                        sampling_rate=command_sampling_rate,
                        dynamic=True,
                        noise_reduce=True,
                    )

                    if not main_loop_active_flag.is_set():
                        break

                    if is_audio_present(recorded_audio_cmd_float32, threshold=0.003):
                        audio_copy_for_thread = recorded_audio_cmd_float32.copy()
                        cmd_thread = threading.Thread(
                            target=process_command_audio_in_thread,
                            args=(
                                audio_copy_for_thread,
                                current_predicted_language,
                                command_sampling_rate,
                                relevant_command_processed_this_session,
                                main_loop_active_flag,
                            ),
                            daemon=True,
                        )
                        cmd_thread.start()
                        active_command_threads.append(cmd_thread)
                    else:
                        print("Tidak ada suara signifikan terdeteksi untuk perintah.")

                    time.sleep(0.1)

                current_active_threads_after_loop = [thread for thread in active_command_threads if thread.is_alive()]
                if current_active_threads_after_loop:
                    print(
                        f"Menunggu {len(current_active_threads_after_loop)} thread perintah yang mungkin masih berjalan setelah sesi perintah..."
                    )
                    for thread in current_active_threads_after_loop:
                        thread.join(timeout=1.5)
                        if thread.is_alive():
                            print(f"PERINGATAN: Thread perintah ID {thread.ident} tidak berhenti tepat waktu setelah sesi perintah berakhir.")
                active_command_threads.clear()

                if main_loop_active_flag.is_set():
                    print("Kembali ke mode deteksi wake word.")
                    intent_feedback("off", current_predicted_language, main_loop_flag=main_loop_active_flag)
            elif not main_loop_active_flag.is_set():
                break
            else:
                print("Tidak ada wake word yang terdeteksi dengan jelas atau bahasa tidak ditentukan. Mencoba lagi...")
                time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nCtrl+C terdeteksi. Menghentikan program...")
    except Exception as exc_main:
        print(f"\nError tak terduga di main loop: {exc_main}")
        import traceback

        traceback.print_exc()
    finally:
        print("Membersihkan resource sebelum keluar...")
        main_loop_active_flag.clear()

        if audio_stream and audio_stream.active:
            print("Menghentikan stream audio final...")
            audio_stream.stop()
        if audio_stream and not audio_stream.closed:
            audio_stream.close()
            print("Stream audio final ditutup.")

        if vosk_thread and vosk_thread.is_alive():
            print("Menunggu Vosk thread (final)...")
            vosk_thread.join(timeout=2.0)
            if vosk_thread.is_alive():
                print("PERINGATAN: Vosk thread tidak berhenti tepat waktu saat shutdown.")
        # no online_check_thread in offline mode

        final_active_command_threads = [thread for thread in active_command_threads if thread.is_alive()]
        if final_active_command_threads:
            print(
                f"Menunggu {len(final_active_command_threads)} thread perintah aktif untuk selesai (final cleanup)..."
            )
            for idx, thread in enumerate(final_active_command_threads, start=1):
                if thread.is_alive():
                    print(f"  Menunggu thread perintah {idx}/{len(final_active_command_threads)} (ID: {thread.ident})...")
                    thread.join(timeout=5.0)
                    if thread.is_alive():
                        print(f"  PERINGATAN: Thread perintah ID {thread.ident} tidak berhenti tepat waktu saat shutdown final.")
        print("Semua thread perintah yang bisa dijoin telah dijoin (final cleanup).")

        if RUNTIME.esp and RUNTIME.esp.is_open:
            RUNTIME.esp.close()
            print("Port serial ESP ditutup.")
        if pygame.mixer.get_init():
            pygame.mixer.music.stop()
            pygame.mixer.quit()
            print("Pygame mixer dihentikan.")
        print("Program selesai.")

        try:
            if MODELS.hailo_encoder is not None:
                MODELS.hailo_encoder.close()
                print("Hailo encoder ditutup.")
        except Exception:
            pass


if __name__ == "__main__":
    main()