import numpy as np
from scipy import signal
from config import DSPConfig


class SpectralSubtraction:
    def __init__(self):
        self.config = DSPConfig()
        self.noise_profile = None
        self.noise_frames = []
        self.window = signal.get_window(self.config.WINDOW_TYPE, self.config.FFT_SIZE)

    def collect_noise_sample(self, audio_frame):
        self.noise_frames.append(audio_frame)

    def finalize_noise_profile(self):
        if len(self.noise_frames) == 0:
            print("Warning: No noise samples collected")
            return

        all_noise = np.concatenate(self.noise_frames)
        noise_spectrum = self._compute_magnitude_spectrum(all_noise)
        self.noise_profile = noise_spectrum
        print(f"Noise profile created from {len(self.noise_frames)} frames")

    def reset_noise_profile(self):
        self.noise_profile = None
        self.noise_frames = []

    def _compute_magnitude_spectrum(self, audio_data):
        if len(audio_data) < self.config.FFT_SIZE:
            audio_data = np.pad(audio_data, (0, self.config.FFT_SIZE - len(audio_data)))

        windowed = audio_data[:self.config.FFT_SIZE] * self.window
        spectrum = np.fft.rfft(windowed, n=self.config.FFT_SIZE)
        magnitude = np.abs(spectrum)

        return magnitude

    def _reconstruct_audio(self, magnitude, phase):
        complex_spectrum = magnitude * np.exp(1j * phase)
        audio_frame = np.fft.irfft(complex_spectrum, n=self.config.FFT_SIZE)

        return audio_frame

    def process(self, audio_frame):
        if self.noise_profile is None:
            return audio_frame

        original_length = len(audio_frame)

        if len(audio_frame) < self.config.FFT_SIZE:
            audio_frame = np.pad(audio_frame, (0, self.config.FFT_SIZE - len(audio_frame)))

        windowed = audio_frame[:self.config.FFT_SIZE] * self.window
        spectrum = np.fft.rfft(windowed, n=self.config.FFT_SIZE)
        magnitude = np.abs(spectrum)
        phase = np.angle(spectrum)

        noise_estimate = self.noise_profile * self.config.OVERSUBTRACTION_FACTOR

        enhanced_magnitude = magnitude - noise_estimate
        enhanced_magnitude = np.maximum(enhanced_magnitude,
                                       magnitude * self.config.SPECTRAL_FLOOR)

        enhanced_audio = self._reconstruct_audio(enhanced_magnitude, phase)

        return enhanced_audio[:original_length]

    def adaptive_process(self, audio_frame, is_speech):
        if not is_speech and self.noise_profile is not None:
            current_spectrum = self._compute_magnitude_spectrum(audio_frame)
            self.noise_profile = (self.config.NOISE_ALPHA * self.noise_profile +
                                 (1 - self.config.NOISE_ALPHA) * current_spectrum)

        return self.process(audio_frame)
