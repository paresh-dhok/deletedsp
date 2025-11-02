import numpy as np
import wave
from scipy import signal


def normalize_audio(audio_data):
    if len(audio_data) == 0:
        return audio_data

    max_val = np.max(np.abs(audio_data))
    if max_val > 0:
        return audio_data / max_val
    return audio_data


def apply_gain(audio_data, gain_db):
    gain_linear = 10 ** (gain_db / 20.0)
    return audio_data * gain_linear


def compute_snr(signal, noise):
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power == 0:
        return float('inf')

    snr = 10 * np.log10(signal_power / noise_power)
    return snr


def save_audio_to_wav(filename, audio_data, sample_rate=16000):
    audio_data = np.clip(audio_data, -32768, 32767).astype(np.int16)

    with wave.open(filename, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())

    print(f"Audio saved to {filename}")


def load_audio_from_wav(filename):
    with wave.open(filename, 'rb') as wav_file:
        sample_rate = wav_file.getframerate()
        n_channels = wav_file.getnchannels()
        audio_data = wav_file.readframes(wav_file.getnframes())

        audio_array = np.frombuffer(audio_data, dtype=np.int16)

        if n_channels > 1:
            audio_array = audio_array.reshape(-1, n_channels)
            audio_array = np.mean(audio_array, axis=1).astype(np.int16)

    return audio_array, sample_rate


def frame_audio(audio_data, frame_size, hop_size):
    frames = []
    for i in range(0, len(audio_data) - frame_size + 1, hop_size):
        frame = audio_data[i:i + frame_size]
        frames.append(frame)

    return frames


def highpass_filter(audio_data, sample_rate=16000, cutoff_hz=120.0, order=4):
    """Apply a Butterworth high-pass filter to 1-D numpy int16 or float array.

    Returns filtered audio as same dtype as input (int16 -> int16, float -> float).
    """
    if len(audio_data) == 0:
        return audio_data

    is_int = np.issubdtype(np.asarray(audio_data).dtype, np.integer)
    data = np.asarray(audio_data).astype(np.float32)

    nyq = 0.5 * sample_rate
    normal_cutoff = min(cutoff_hz / nyq, 0.99)
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    filtered = signal.lfilter(b, a, data)

    if is_int:
        filtered = np.clip(filtered, -32768, 32767).astype(np.int16)
    return filtered
