"""Voice assistant entry point using Hailo Whisper encoder and fallback pipelines."""

import json
import logging
import os
import socket
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pygame
import pyaudio
import requests
import serial
import sounddevice as sd
import speech_recognition as sr
import torch
import noisereduce as nr
from dotenv import load_dotenv
from pywifi import PyWiFi, Profile, const
from scipy.io import wavfile as wav
from serial.tools import list_ports
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from vosk import KaldiRecognizer, Model

from faster_whisper import WhisperModel

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


@dataclass
class AssistantConfig:
    use_hailo_encoder: bool = True
    hailo_encoder_hef: str = "base-whisper-encoder-5s_h8l.hef"
    hailo_window_seconds: int = 5
    use_google_stt: bool = False
    decoder_local_dir: Optional[str] = None
    send_to_webserver: bool = True
    web_server_url: str = "http://10.0.0.76:5000/task"
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
    faster_whisper: Optional[Any] = None


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

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("PERINGATAN: GEMINI_API_KEY tidak ditemukan di .env file. Fungsi online intent akan gagal.")
    GEMINI_URL = ""
else:
    GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=" + GEMINI_API_KEY

try:
    pygame.mixer.init()
except pygame.error as e:
    print(f"Gagal inisialisasi pygame mixer: {e}. Feedback suara mungkin tidak berfungsi.")

this_file_dir = os.path.dirname(os.path.abspath(__file__))

# --- Definisi Label untuk Mode Offline dan Online ---
# Daftar label untuk model OFFLINE (yang sudah dilatih)
nlp_label_list_offline = [
    "nyalakan lampu merah", "nyalakan lampu hijau", "nyalakan lampu biru",
    "nyalakan lampu lavender", "nyalakan lampu magenta", "nyalakan lampu pink",
    "nyalakan lampu violet", "nyalakan lampu aqua", "nyalakan lampu kuning",
    "nyalakan lampu emas", "nyalakan lampu abu", "nyalakan mode senang",
    "nyalakan mode sad", "matikan lampu", "tidak relevan", "nyalakan lampu",
    "gender ke wanita", "gender ke pria", "fitur belum didukung", "turunkan kecerahan", "naikkan kecerahan"
]

# Label baru yang hanya akan digunakan dalam mode ONLINE
new_online_only_labels = [
    "buka kap mobil",
    "buka tutup bensin",
    # "posisi dongkrak",
    # "cara buka ban serep",
    # "cara ganti ban",
    # "info seatbelt pretensioner"
]

# Daftar label gabungan untuk mode ONLINE (Gemini API)
nlp_label_list_online = nlp_label_list_offline + new_online_only_labels


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


# Inisialisasi model STT (faster-whisper)
# try:
#     stt_model_faster_whisper = WhisperModel("base", device="cpu", compute_type="int8")
#     print("Model STT (faster-whisper 'base') berhasil dimuat.")
# except Exception as e:
#     print(f"Gagal memuat model STT (faster-whisper): {e}. Program mungkin tidak berfungsi dengan benar.")
#     sys.exit(1)

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
            MODELS.hf_processor = AutoProcessor.from_pretrained("openai/whisper-base")
            MODELS.hf_decoder = (
                WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
                .eval()
                .to("cpu")
            )

        MODELS.faster_whisper = None
        LOGGER.info("STT siap: Hailo encoder (base, 5s) + HF decoder (CPU).")
    else:
        raise RuntimeError("CONFIG.use_hailo_encoder is False")
except Exception as e:
    print(f"Gagal init Hailo encoder pipeline: {e}. Fallback ke faster-whisper.")
    MODELS.hailo_encoder = None
    MODELS.hf_processor = None
    MODELS.hf_decoder = None
    try:
        MODELS.faster_whisper = WhisperModel("base", device="cpu", compute_type="int8")
        LOGGER.info("Model STT (faster-whisper 'base') berhasil dimuat.")
    except Exception as e2:
        print(f"Gagal memuat model STT (faster-whisper): {e2}")
        sys.exit(1)


# --- Audio Buffer Management ---

def callback(indata, frames, time_info, status):
    """Stream callback that pushes audio chunks into the ring buffer."""
    if status and status.input_overflow:
        print("PERINGATAN: Input audio overflow! Data audio mungkin hilang.", file=sys.stderr)
    RUNTIME.ring_buffer.write(bytes(indata))


# --- Utility Functions ---

def connect_to_wifi(ssid, password):
    wifi = PyWiFi()
    iface = None
    try:
        ifaces = wifi.interfaces()
        if not ifaces:
            print("Tidak ada antarmuka Wi-Fi yang ditemukan.")
            return False
        iface = ifaces[0]
    except Exception as e:
        print(f"Error saat mengakses antarmuka Wi-Fi: {e}")
        return False
    try:
        iface.scan()
        print("Scanning Wi-Fi...")
        time.sleep(3)
        scan_results = iface.scan_results()
    except Exception as e:
        print(f"Error saat scan Wi-Fi: {e}")
        return False
    for network in scan_results:
        if network.ssid == ssid:
            print(f"SSID ditemukan: {ssid}")
            profile = Profile()
            profile.ssid = ssid
            profile.auth = const.AUTH_ALG_OPEN
            profile.akm.append(const.AKM_TYPE_WPA2PSK)
            profile.cipher = const.CIPHER_TYPE_CCMP
            profile.key = password
            iface.remove_all_network_profiles()
            temp_profile = iface.add_network_profile(profile)
            iface.connect(temp_profile)
            print(f"Mencoba terhubung ke {ssid}...")
            for _ in range(10):
                time.sleep(1)
                if iface.status() == const.IFACE_CONNECTED:
                    print("Berhasil terhubung ke Wi-Fi!")
                    return True
            print(f"Gagal terhubung ke {ssid} setelah beberapa percobaan.")
            return False
    print(f"SSID {ssid} tidak ditemukan.")
    return False

def is_internet_connected(timeout=2):
    if not CONFIG.use_google_stt:
        return False
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=timeout)
        return True
    except OSError:
        return False
    except Exception:
        return False

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

def record_audio_dynamic(duration_min=3, duration_max=6, silence_duration=1, sampling_rate=16000):
    chunk_duration = 0.5
    chunk_samples = int(chunk_duration * sampling_rate)
    total_audio = []
    min_samples = int(duration_min * sampling_rate)
    max_samples = int(duration_max * sampling_rate)
    silence_threshold = 0.01
    print("Mulai merekam (dinamis)...")
    try:
        with sd.InputStream(samplerate=sampling_rate, channels=1, dtype='float32', blocksize=chunk_samples) as stream:
            start_time = time.time()
            while len(total_audio) < min_samples:
                if time.time() - start_time > duration_max + 3: # Safety break
                    print("Perekaman dinamis (min_samples) melebihi batas waktu maksimum.")
                    break
                try:
                    chunk, overflowed = stream.read(chunk_samples)
                    if overflowed:
                        print("PERINGATAN: Input audio overflow saat merekam perintah (min_samples loop).", file=sys.stderr)
                    total_audio.extend(chunk.flatten())
                except sd.CallbackStop:
                    print("Perekaman dinamis (min_samples) dihentikan oleh callback.")
                    break
                except Exception as e:
                     print(f"Error saat membaca stream (min_samples): {e}")
                     break

            last_audio_activity_time = time.time()
            while len(total_audio) < max_samples:
                 current_time = time.time()
                 if current_time - start_time > duration_max + 3: # Safety break
                     print("Perekaman dinamis (max_samples) melebihi batas waktu maksimum.")
                     break
                 if current_time - last_audio_activity_time > silence_duration + chunk_duration: 
                      print("Deteksi silence (timeout aktivitas), menghentikan perekaman dinamis.")
                      break
                 try:
                    chunk, overflowed = stream.read(chunk_samples)
                    if overflowed:
                        print("PERINGATAN: Input audio overflow saat merekam perintah (max_samples loop).", file=sys.stderr)
                    total_audio.extend(chunk.flatten())
                    if np.mean(np.abs(chunk)) >= silence_threshold: 
                        last_audio_activity_time = current_time
                 except sd.CallbackStop:
                    print("Perekaman dinamis (max_samples) dihentikan oleh callback.")
                    break
                 except Exception as e:
                     print(f"Error saat membaca stream (max_samples): {e}")
                     break
        print("Selesai merekam (dinamis).")
        return np.array(total_audio)
    except sd.PortAudioError as e:
        print(f"Error SoundDevice saat merekam (dinamis): {e}")
        return np.array([])
    except Exception as e:
        print(f"Error tak terduga saat merekam (dinamis): {e}")
        return np.array([])

def online_stt_recognize(audio_data_sr, language, result_container):
    try:
        recognizer = sr.Recognizer()
        result_container['text'] = recognizer.recognize_google(audio_data_sr, language=language)
    except sr.UnknownValueError:
        result_container['text'] = ""
    except sr.RequestError as e:
        print(f"Google STT: Error koneksi/API; {e}")
        result_container['text'] = ""
    except Exception as e:
        print(f"Google STT: Error tak terduga: {e}")
        result_container['text'] = ""

def record_audio(duration=2, sampling_rate=16000, noise_reduce=True, dynamic=True):
    if dynamic:
        audio_data = record_audio_dynamic(duration_min=duration, duration_max=5, silence_duration=1, sampling_rate=sampling_rate)
    else:
        print(f"Mulai merekam (durasi tetap: {duration} detik)...")
        try:
            num_samples = int(duration * sampling_rate)
            recording = sd.rec(
                num_samples,
                samplerate=sampling_rate,
                channels=1,
                dtype='float32'
            )
            sd.wait()
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

    stt_model = MODELS.faster_whisper
    if stt_model is None or audio_array_float32.size == 0:
        return {'text': '', 'processing_time': 0}
    try:
        import time
        import numpy as np

        start = time.time()
        lang_code_map = {"Indonesian": "id", "English": "en", "Japanese": "ja"}
        fw_lang_code = lang_code_map.get(language, "id")
        if audio_array_float32.dtype != np.float32:
            audio_array_float32 = audio_array_float32.astype(np.float32)
        segments, _info = stt_model.transcribe(
            audio_array_float32, beam_size=5, language=fw_lang_code, vad_filter=True
        )
        text = "".join(segment.text for segment in segments).strip()
        return {'text': text, 'processing_time': time.time() - start}
    except Exception as e:
        print(f"Error faster-whisper: {e}")
        return {'text': '', 'processing_time': 0}


def predict_intent(is_online, text):
    def predict_online(text_input):
        if not GEMINI_URL or not GEMINI_API_KEY:
            print("Gemini API URL atau Key tidak dikonfigurasi. Beralih ke model offline.")
            return predict_offline(text_input)
        # Gunakan daftar label ONLINE untuk prompt Gemini
        prompt = f"""Anda adalah sebuah model untuk melakukan intent classification. Berikut adalah list intent yang mendukung: {nlp_label_list_online} Tolong respons dengan absolut intent yang sesuai dengan kalimat yang diberikan saja, jangan tambahkan kata-kata lain. Classification hanya boleh 1 intent saja. Jika lebih dari satu maksud terdeteksi atau tidak ada yang cocok, kembalikan "tidak relevan". Pikirkan matang-matang maksud dari teks yang diberikan, terkadang terdapat maksud tersirat yang harus dipahami. Jangan terpaku pada teks yang diberikan saja. Teks yang diberikan adalah: "{text_input}". Untuk saat ini program hanya mendukung action untuk ambient light, ganti suara, buka kap mobil, buka tutup bensin, posisi dongkrak, cara buka ban serep, cara ganti ban, dan info seatbelt pretensioner. fitur fitur lain yang berhubungan dengan mobil tapi belum didukung, tolong klasifikasikan sebagai "fitur belum didukung". Intent yang paling sesuai adalah:"""
        data = {"contents" : [{"parts" : [{"text" : prompt}]}]}
        try:
            response = requests.post(GEMINI_URL, json=data, timeout=8)
            response.raise_for_status() 
            response_json = response.json()
            if "candidates" in response_json and \
               len(response_json["candidates"]) > 0 and \
               "content" in response_json["candidates"][0] and \
               "parts" in response_json["candidates"][0]["content"] and \
               len(response_json["candidates"][0]["content"]["parts"]) > 0 and \
               "text" in response_json["candidates"][0]["content"]["parts"][0]:
                predicted_intent = response_json["candidates"][0]["content"]["parts"][0]["text"].strip()
                # Cek apakah hasil prediksi ada di dalam daftar label online
                if predicted_intent in nlp_label_list_online:
                    return predicted_intent
                else:
                    for label in nlp_label_list_online:
                        if label in predicted_intent: 
                            return label
                    return "tidak relevan" 
            else:
                return "tidak relevan"
        except requests.exceptions.Timeout:
            print("Gemini API: Timeout.")
            return "tidak relevan"
        except requests.exceptions.HTTPError as e:
            print(f"Gemini API: HTTP Error: {e}")
            return "tidak relevan"
        except requests.exceptions.RequestException as e:
            print(f"Gemini API: Request Exception: {e}")
            return "tidak relevan"
        except json.JSONDecodeError:
            print("Gemini API: Gagal parse JSON response.")
            return "tidak relevan"
        except KeyError as e:
            print(f"Gemini API: KeyError saat parsing response - {e}.")
            return "tidak relevan"
        except Exception as e:
            print(f"Gemini API: Terjadi error tak terduga: {e}")
            return "tidak relevan"

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
            
    if is_online:
        return predict_online(text)
    else:
        return predict_offline(text)

def send_to_webserver(command: str) -> bool:
    if not CONFIG.send_to_webserver:
        LOGGER.info("Pengiriman ke web server dinonaktifkan.")
        return True
    try:
        params = {'label': command}
        response = requests.get(CONFIG.web_server_url, params=params)
        response.raise_for_status()
        LOGGER.info("Berhasil mengirim ke web server: %s", command)
        return True
    except requests.exceptions.Timeout:
        print("Timeout saat mengirim ke web server.")
        return False
    except requests.exceptions.RequestException as exc:
        print(f"Gagal mengirim ke web server: {exc}")
        return False

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
    elif intent in new_online_only_labels:
        print(f"Fitur terdeteksi tapi belum ada feedback audio untuk intent '{intent}'.")
        send_to_webserver(intent)
        STATE.last_command = intent
        pass
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
    intent_feedback("system_on", main_loop_flag=main_loop_flag)
    print("Sistem siap.")
    return True

ONLINE_WAKE_WORD_CHECK_INTERVAL = 2
ONLINE_WAKE_WORD_RECORD_DURATION = 4
SAMPLE_RATE = 16000
SAMPLE_WIDTH_BYTES = 2

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


def run_online_wake_word_check(wake_event, samplerate, result_language_container, main_loop_flag):
    if wake_event.is_set() or not main_loop_flag.is_set():
        return

    print("Memulai pengecekan wake word online (1x STT id-ID)...")
    bytes_to_collect_requested = int(ONLINE_WAKE_WORD_RECORD_DURATION * samplerate * SAMPLE_WIDTH_BYTES)

    ring_buffer = RUNTIME.ring_buffer
    total_written = ring_buffer.snapshot_total_written()
    if total_written == 0:
        print("Online check: Tidak ada data audio di buffer untuk dicek (0 bytes available).")
        return

    start_pos = max(0, total_written - bytes_to_collect_requested)
    collected_bytes, _ = ring_buffer.read_available_since(start_pos, wait_timeout=0.0)
    if not collected_bytes:
        print("Online check: Tidak ada data audio diperoleh dari buffer.")
        return

    if len(collected_bytes) > bytes_to_collect_requested:
        collected_bytes = collected_bytes[-bytes_to_collect_requested:]

    if wake_event.is_set() or not main_loop_flag.is_set():
        print("Online check: Pengecekan dibatalkan setelah mengumpulkan data.")
        return

    audio_int16_np = np.frombuffer(collected_bytes, dtype=np.int16)
    audio_float32_np = audio_int16_np.astype(np.float32) / 32768.0

    print(
        f"Online check: Audio terkumpul {len(collected_bytes)} bytes ({len(collected_bytes)/(samplerate*SAMPLE_WIDTH_BYTES):.2f}s). Memanggil is_audio_present..."
    )
    if not is_audio_present(audio_float32_np, threshold=0.003):
        print("Online check: Audio tidak signifikan terdeteksi (setelah is_audio_present).")
        return

    try:
        audio_data_sr = sr.AudioData(
            collected_bytes, sample_rate=samplerate, sample_width=SAMPLE_WIDTH_BYTES
        )
        stt_result_container: Dict[str, str] = {}
        stt_thread = threading.Thread(
            target=online_stt_recognize,
            args=(audio_data_sr, "id-ID", stt_result_container),
            daemon=True,
        )
        stt_thread.start()
        stt_thread.join(timeout=5.0)

        if not main_loop_flag.is_set() or wake_event.is_set():
            return

        if stt_thread.is_alive():
            print("Online STT (wake word) timeout.")
            return

        text_transcribed = stt_result_container.get('text', "").lower().strip()
        if text_transcribed:
            print(f"Online check STT result: '{text_transcribed}'.")
        else:
            print("Online check: STT tidak menghasilkan teks.")
            return

        if wake_event.is_set() or not main_loop_flag.is_set():
            return

        potential_detected_language = None
        if "oke toyota" in text_transcribed or ("oke" in text_transcribed and "toyota" in text_transcribed) or ("okay toyota" in text_transcribed):
            potential_detected_language = "Indonesian"
        elif "hello toyota" in text_transcribed or ("hello" in text_transcribed and "toyota" in text_transcribed):
            potential_detected_language = "English"
        elif "hai toyota" in text_transcribed or ("hai" in text_transcribed and "toyota" in text_transcribed) or ("moshi moshi" in text_transcribed):
            potential_detected_language = "Japanese"

        if potential_detected_language:
            print(f"Online STT mendeteksi wake word! Bahasa: {potential_detected_language}")
            result_language_container['language'] = potential_detected_language
            wake_event.set()
        else:
            print("Online STT: Tidak mendeteksi wake word yang valid.")
    except Exception as exc:
        print(f"Online STT check error: {exc}")


def process_command_audio_in_thread(audio_float32_data, language, sampling_rate, relevant_command_event, main_loop_flag_ref):
    thread_id = threading.get_ident()

    if not main_loop_flag_ref.is_set():
        print(f"Thread ID {thread_id}: Program shutting down. Exiting command thread.")
        return
    if relevant_command_event.is_set():
        print(f"Thread ID {thread_id}: Relevant command already processed by another thread. Exiting early.")
        return

    internet_conn = is_internet_connected()
    text_transcribed = ""
    stt_processing_time = 0

    # --- STT Phase ---
    if CONFIG.use_google_stt and internet_conn:
        try:
            stt_online_start_time = time.time()
            audio_int16_cmd = (audio_float32_data * 32767).astype(np.int16)
            audio_data_sr_cmd = sr.AudioData(
                audio_int16_cmd.tobytes(),
                sample_rate=sampling_rate,
                sample_width=SAMPLE_WIDTH_BYTES 
            )
            lang_code_google_cmd = {"Indonesian": "id-ID", "English": "en-US", "Japanese": "ja-JP"}.get(language, "id-ID")
            result_container_cmd = {}
            stt_thread_cmd = threading.Thread(
                target=online_stt_recognize,
                args=(audio_data_sr_cmd, lang_code_google_cmd, result_container_cmd)
            )
            stt_thread_cmd.daemon = True
            stt_thread_cmd.start()
            
            join_timeout = 5.0
            wait_interval = 0.1
            elapsed_wait = 0
            while stt_thread_cmd.is_alive() and elapsed_wait < join_timeout:
                if not main_loop_flag_ref.is_set() or relevant_command_event.is_set():
                    print(f"Thread ID {thread_id}: STT online (Google) interrupted by shutdown or relevant command event.")
                    return 
                time.sleep(wait_interval)
                elapsed_wait += wait_interval
            
            if stt_thread_cmd.is_alive():
                print(f"Thread ID {thread_id}: Timeout STT online (Google) untuk perintah.")
                raise TimeoutError("Timeout STT online (perintah)")
            
            if not main_loop_flag_ref.is_set() or relevant_command_event.is_set(): return

            text_transcribed = result_container_cmd.get('text', "")
            stt_processing_time = time.time() - stt_online_start_time
            if not text_transcribed:
                print(f"Thread ID {thread_id}: STT Online (Google) tidak menghasilkan teks.")
                raise ValueError("STT Online (Google) gagal menghasilkan teks")

        except Exception as e_stt_online_cmd:
            if not main_loop_flag_ref.is_set() or relevant_command_event.is_set(): return
            print(f"Thread ID {thread_id}: Gagal STT online (perintah): {e_stt_online_cmd}. Menggunakan STT offline (faster-whisper)...")
            
            if not main_loop_flag_ref.is_set() or relevant_command_event.is_set(): return
            offline_result_cmd = transcribe_audio(audio_float32_data, sampling_rate=sampling_rate, language=language)
            text_transcribed = offline_result_cmd["text"]
            stt_processing_time = offline_result_cmd["processing_time"]
    else: 
        if not main_loop_flag_ref.is_set() or relevant_command_event.is_set(): return
        print(f"Thread ID {thread_id}: Tidak ada koneksi internet. Menggunakan STT offline (faster-whisper) untuk perintah...")
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
        predicted_intent = predict_intent(internet_conn, text_transcribed)
    
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

    try:
        device_info = sd.query_devices(None, "input")
        samplerate = int(device_info.get("default_samplerate", 16000))
        print(f"Menggunakan samplerate: {samplerate} Hz dari perangkat input default.")
    except Exception as exc:
        print(f"PERINGATAN: Gagal mendapatkan default samplerate dari SoundDevice: {exc}. Menggunakan default 16000 Hz.")
        samplerate = 16000

    audio_stream = None
    vosk_thread = None
    online_check_thread = None
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
                audio_stream = sd.RawInputStream(
                    samplerate=samplerate,
                    blocksize=blocksize_callback_samples,
                    device=None,
                    dtype="int16",
                    channels=1,
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

            last_online_check_time = time.time()
            while not wake_word_detected_event.is_set() and main_loop_active_flag.is_set():
                current_time = time.time()
                internet_available = is_internet_connected(timeout=0.5)
                if internet_available and (current_time - last_online_check_time > ONLINE_WAKE_WORD_CHECK_INTERVAL):
                    if online_check_thread is None or not online_check_thread.is_alive():
                        last_online_check_time = current_time
                        online_check_thread = threading.Thread(
                            target=run_online_wake_word_check,
                            args=(wake_word_detected_event, samplerate, detected_language_container, main_loop_active_flag),
                            daemon=True,
                        )
                        online_check_thread.start()
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

            if online_check_thread and online_check_thread.is_alive():
                print("Menunggu online check thread selesai...")
                online_check_thread.join(timeout=ONLINE_WAKE_WORD_RECORD_DURATION + 7.0)
                if online_check_thread.is_alive():
                    print("PERINGATAN: Online check thread tidak berhenti tepat waktu.")
            online_check_thread = None

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
                    command_sampling_rate = 16000
                    recorded_audio_cmd_float32 = record_audio(
                        duration=3,
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
        if online_check_thread and online_check_thread.is_alive():
            print("Menunggu online check thread (final)...")
            online_check_thread.join(timeout=5.0)
            if online_check_thread.is_alive():
                print("PERINGATAN: Online check thread tidak berhenti tepat waktu saat shutdown.")

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