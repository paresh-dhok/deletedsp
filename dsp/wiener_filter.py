import numpy as np
from scipy import signal
from config import DSPConfig


class WienerFilter:
    def __init__(self):
        self.config = DSPConfig()
        self.noise_power = None
        self.signal_power = None
        self.window = signal.get_window(self.config.WINDOW_TYPE, self.config.FFT_SIZE)

    def estimate_noise_power(self, noise_frames):
        all_noise = np.concatenate(noise_frames)
        noise_spectrum = self._compute_power_spectrum(all_noise)
        self.noise_power = noise_spectrum
        print(f"Noise power estimated from {len(noise_frames)} frames")

    def _compute_power_spectrum(self, audio_data):
        if len(audio_data) < self.config.FFT_SIZE:
            audio_data = np.pad(audio_data, (0, self.config.FFT_SIZE - len(audio_data)))

        windowed = audio_data[:self.config.FFT_SIZE] * self.window
        spectrum = np.fft.rfft(windowed, n=self.config.FFT_SIZE)
        power = np.abs(spectrum) ** 2

        return power

    def _compute_wiener_gain(self, noisy_power):
        if self.noise_power is None:
            return np.ones_like(noisy_power)

        snr = (noisy_power - self.noise_power) / (self.noise_power + 1e-10)
        snr = np.maximum(snr, 0)

        gain = snr / (snr + 1)
        gain = np.maximum(gain, self.config.WIENER_MIN_GAIN)

        return gain

    def process(self, audio_frame):
        if self.noise_power is None:
            return audio_frame

        original_length = len(audio_frame)

        if len(audio_frame) < self.config.FFT_SIZE:
            audio_frame = np.pad(audio_frame, (0, self.config.FFT_SIZE - len(audio_frame)))

        windowed = audio_frame[:self.config.FFT_SIZE] * self.window
        spectrum = np.fft.rfft(windowed, n=self.config.FFT_SIZE)
        magnitude = np.abs(spectrum)
        phase = np.angle(spectrum)
        complex_spectrum = enhanced_magnitude * np.exp(1j * phase)
        enhanced_audio = np.fft.irfft(complex_spectrum, n=self.config.FFT_SIZE)

        return enhanced_audio[:original_length]

    def adaptive_process(self, audio_frame, is_speech):
        if self.noise_power is None:
            return audio_frame

        original_length = len(audio_frame)

        if len(audio_frame) < self.config.FFT_SIZE:
            audio_frame = np.pad(audio_frame, (0, self.config.FFT_SIZE - len(audio_frame)))

        windowed = audio_frame[:self.config.FFT_SIZE] * self.window
        spectrum = np.fft.rfft(windowed, n=self.config.FFT_SIZE)
        magnitude = np.abs(spectrum)
        phase = np.angle(spectrum)
        power = magnitude ** 2

        if not is_speech:
            self.noise_power = (self.config.WIENER_ALPHA * self.noise_power +
                              (1 - self.config.WIENER_ALPHA) * power)

        gain = self._compute_wiener_gain(power)

        enhanced_magnitude = magnitude * gain

        complex_spectrum = enhanced_magnitude * np.exp(1j * phase)
        enhanced_audio = np.fft.irfft(complex_spectrum, n=self.config.FFT_SIZE)

        enhanced_audio = enhanced_audio * self.window[:len(enhanced_audio)]

        return enhanced_audio[:original_length]
