import numpy as np
from scipy import signal
from config import DSPConfig


class VoiceActivityDetector:
    def __init__(self):
        self.config = DSPConfig()
        # Use a lower sensible default if config value is too large for int16-normalized energy
        default_thresh = getattr(self.config, 'VAD_DEFAULT_THRESHOLD', None)
        if default_thresh is not None:
            self.energy_threshold = default_thresh
        else:
            self.energy_threshold = self.config.VAD_THRESHOLD
        self.smoothing_window = self.config.VAD_SMOOTHING
        self.recent_decisions = []
        self.noise_floor = None
        self.is_calibrated = False

    def calibrate_noise_floor(self, noise_frames):
        if len(noise_frames) == 0:
            return

        energies = [self._compute_energy(frame) for frame in noise_frames]
        self.noise_floor = np.mean(energies)
        self.energy_threshold = self.noise_floor * 3.0
        self.is_calibrated = True
        print(f"VAD calibrated: noise floor = {self.noise_floor:.6f}, threshold = {self.energy_threshold:.6f}")

    def _compute_energy(self, audio_frame):
        normalized = audio_frame.astype(np.float32) / 32768.0
        energy = np.mean(normalized ** 2)
        return energy

    def _compute_zero_crossing_rate(self, audio_frame):
        signs = np.sign(audio_frame)
        zero_crossings = np.sum(np.abs(np.diff(signs))) / (2 * len(audio_frame))
        return zero_crossings

    def _compute_spectral_centroid(self, audio_frame):
        fft_size = 512
        if len(audio_frame) < fft_size:
            audio_frame = np.pad(audio_frame, (0, fft_size - len(audio_frame)))

        spectrum = np.abs(np.fft.rfft(audio_frame[:fft_size]))
        freqs = np.fft.rfftfreq(fft_size, 1.0 / 16000)

        if np.sum(spectrum) == 0:
            return 0

        centroid = np.sum(freqs * spectrum) / np.sum(spectrum)
        return centroid

    def detect(self, audio_frame):
        energy = self._compute_energy(audio_frame)

        zcr = self._compute_zero_crossing_rate(audio_frame)

        spectral_centroid = self._compute_spectral_centroid(audio_frame)

        energy_decision = energy > self.energy_threshold

        zcr_decision = 0.01 < zcr < 0.3

        spectral_decision = spectral_centroid > 500

        is_speech = energy_decision and (zcr_decision or spectral_decision)

        self.recent_decisions.append(is_speech)
        if len(self.recent_decisions) > self.smoothing_window:
            self.recent_decisions.pop(0)

        smoothed_decision = sum(self.recent_decisions) > (self.smoothing_window / 2)

        return smoothed_decision

    def get_speech_probability(self, audio_frame):
        energy = self._compute_energy(audio_frame)
        # Compute a robust SNR-like measure for probability
        eps = 1e-12
        noise = self.noise_floor if (self.noise_floor is not None and self.noise_floor > eps) else eps

        ratio = (energy + eps) / (noise + eps)
        # map ratio to a probability in a smooth way
        # If ratio == 1 -> ~0.5, larger ratio -> closer to 1
        prob = 1.0 / (1.0 + np.exp(-2.0 * (np.log10(ratio) * 2.0)))

        # clip to [0,1]
        prob = float(np.clip(prob, 0.0, 1.0))
        return prob

    def reset(self):
        self.recent_decisions = []
