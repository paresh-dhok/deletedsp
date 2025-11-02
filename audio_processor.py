import numpy as np
import threading
import time
import wave
import os
import queue
from datetime import datetime
from audio import AudioCapture
from dsp import SpectralSubtraction, WienerFilter
from ml import VoiceActivityDetector
from config import AudioConfig, RecordingConfig
from collections import deque
from utils.audio_utils import highpass_filter


class AudioProcessor:
    def __init__(self):
        self.audio_capture = AudioCapture()
        self.spectral_subtraction = SpectralSubtraction()
        self.wiener_filter = WienerFilter()
        self.vad = VoiceActivityDetector()
        self.config = AudioConfig()
        self.recording_config = RecordingConfig()

        self.is_processing = False
        self.processing_thread = None
        self.bypass_mode = False

        self.use_spectral_subtraction = True
        self.use_wiener_filter = True
        self.use_adaptive_noise = True

        self.is_recording = False
        self.wav_file = None
        self.wav_writer = None
        self.recorded_frames = []

        # Prebuffer for recording speech-only mode
        prebuffer_frames = getattr(self.recording_config, 'RECORD_PREBUFFER_FRAMES', 3)
        self._record_prebuffer = deque(maxlen=prebuffer_frames)
        self._recording_active_segment = False
        self._post_silence_counter = 0

        self.stats = {
            'frames_processed': 0,
            'speech_frames': 0,
            'noise_frames': 0,
            'current_speech_prob': 0.0,
            'processing_time_ms': 0.0,
            'recording_time': 0.0,
            'recorded_frames': 0
        }

        os.makedirs(self.recording_config.RECORDINGS_DIR, exist_ok=True)

    def calibrate_noise(self, duration_seconds=2.0):
        print(f"\nCalibrating noise profile for {duration_seconds} seconds...")
        print("Please remain silent...")

        noise_frames = []
        start_time = time.time()

        while time.time() - start_time < duration_seconds:
            audio_data = self.audio_capture.read_audio(timeout=0.1)
            if audio_data is not None:
                noise_frames.append(audio_data)
            time.sleep(0.01)

        if len(noise_frames) > 0:
            self.spectral_subtraction.noise_frames = noise_frames
            self.spectral_subtraction.finalize_noise_profile()

            self.wiener_filter.estimate_noise_power(noise_frames)

            self.vad.calibrate_noise_floor(noise_frames)

            print(f"Calibration complete with {len(noise_frames)} frames")
        else:
            print("Warning: No audio frames captured during calibration")

    def _processing_loop(self):
        print("Processing loop started")
        recording_start_time = time.time()

        while self.is_processing:
            try:
                audio_data = self.audio_capture.read_audio(timeout=0.1)

                if audio_data is None:
                    continue

                start_time = time.time()

                if self.bypass_mode:
                    processed_audio = audio_data
                    # still compute speech probability for monitoring even when bypassed
                    try:
                        prob = float(self.vad.get_speech_probability(audio_data))
                    except Exception:
                        prob = 0.0
                    self.stats['current_speech_prob'] = prob
                    is_speech = False
                else:
                    processed_audio = self._process_frame(audio_data)

                    is_speech = self.vad.detect(audio_data)
                    try:
                        self.stats['current_speech_prob'] = float(self.vad.get_speech_probability(audio_data))
                    except Exception:
                        self.stats['current_speech_prob'] = 0.0

                    if is_speech:
                        self.stats['speech_frames'] += 1
                    else:
                        self.stats['noise_frames'] += 1

                # Always keep a short prebuffer of processed frames (int16) to include just before speech
                try:
                    self._record_prebuffer.append(processed_audio)
                except Exception:
                    pass

                if self.is_recording:
                    # Decide whether to append this frame based on recording policy
                    record_speech_only = getattr(self.recording_config, 'RECORD_SPEECH_ONLY', True)
                    vad_threshold = getattr(self.recording_config, 'RECORD_VAD_THRESHOLD', 0.5)
                    post_frames_allowed = getattr(self.recording_config, 'RECORD_POST_FRAMES', 5)

                    if record_speech_only:
                        prob = float(self.stats.get('current_speech_prob', 0.0))
                        if prob >= vad_threshold:
                            # flush prebuffer if entering speech segment
                            if not self._recording_active_segment:
                                while len(self._record_prebuffer) > 0:
                                    f = self._record_prebuffer.popleft()
                                    # apply high-pass filter to reduce thumps
                                    try:
                                        sample_rate = getattr(self.audio_capture, 'actual_rate', self.config.RATE)
                                        f = highpass_filter(f, sample_rate=sample_rate,
                                                            cutoff_hz=self.recording_config.RECORD_HP_CUTOFF_HZ)
                                    except Exception:
                                        pass
                                    self.recorded_frames.append(f)
                                self._recording_active_segment = True
                                self._post_silence_counter = 0

                            # append current frame (filtered)
                            try:
                                sample_rate = getattr(self.audio_capture, 'actual_rate', self.config.RATE)
                                filtered = highpass_filter(processed_audio, sample_rate=sample_rate,
                                                           cutoff_hz=self.recording_config.RECORD_HP_CUTOFF_HZ)
                            except Exception:
                                filtered = processed_audio
                            self.recorded_frames.append(filtered)
                            self._post_silence_counter = 0

                        else:
                            # below threshold: if we were recording, allow a few post frames
                            if self._recording_active_segment:
                                if self._post_silence_counter < post_frames_allowed:
                                    try:
                                        sample_rate = getattr(self.audio_capture, 'actual_rate', self.config.RATE)
                                        filtered = highpass_filter(processed_audio, sample_rate=sample_rate,
                                                                   cutoff_hz=self.recording_config.RECORD_HP_CUTOFF_HZ)
                                    except Exception:
                                        filtered = processed_audio
                                    self.recorded_frames.append(filtered)
                                    self._post_silence_counter += 1
                                else:
                                    # end of speech segment
                                    self._recording_active_segment = False
                                    self._post_silence_counter = 0
                            else:
                                # not in active speech segment; do not record
                                pass
                    else:
                        # record everything (but apply HP filter to reduce thumps)
                        try:
                            sample_rate = getattr(self.audio_capture, 'actual_rate', self.config.RATE)
                            filtered = highpass_filter(processed_audio, sample_rate=sample_rate,
                                                       cutoff_hz=self.recording_config.RECORD_HP_CUTOFF_HZ)
                        except Exception:
                            filtered = processed_audio
                        self.recorded_frames.append(filtered)

                    self.stats['recorded_frames'] = len(self.recorded_frames)
                    self.stats['recording_time'] = time.time() - recording_start_time

                self.stats['frames_processed'] += 1
                processing_time = (time.time() - start_time) * 1000
                self.stats['processing_time_ms'] = processing_time

            except (queue.Empty, queue.Full, IOError) as e:
                print(f"Error in processing loop: {e}")
                time.sleep(0.01)

        print("Processing loop ended")

    def _process_frame(self, audio_frame):
        audio_float = audio_frame.astype(np.float32)

        is_speech = self.vad.detect(audio_frame)

        if self.use_spectral_subtraction:
            if self.use_adaptive_noise:
                audio_float = self.spectral_subtraction.adaptive_process(audio_float, is_speech)
            else:
                audio_float = self.spectral_subtraction.process(audio_float)

        if self.use_wiener_filter:
            if self.use_adaptive_noise:
                audio_float = self.wiener_filter.adaptive_process(audio_float, is_speech)
            else:
                audio_float = self.wiener_filter.process(audio_float)

        audio_float = np.clip(audio_float, -32768, 32767)
        processed_audio = audio_float.astype(np.int16)

        return processed_audio

    def start_processing(self, input_device=None, output_device=None):
        if self.is_processing:
            print("Already processing")
            return

        self.audio_capture.start(input_device, output_device)

        self.is_processing = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()

        print(f"Audio processing started (Input Device: {input_device})")

    def stop_processing(self):
        if not self.is_processing:
            return

        self.is_processing = False

        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)

        self.audio_capture.stop()

        print("Audio processing stopped")

    def toggle_bypass(self):
        self.bypass_mode = not self.bypass_mode
        return self.bypass_mode

    def start_recording(self):
        if self.is_recording:
            print("Already recording")
            return False
        self.is_recording = True
        self.recorded_frames = []
        self.stats['recorded_frames'] = 0
        self.stats['recording_time'] = 0.0
        print("Recording started - Enhanced audio will be saved")
        return True

    def stop_recording(self):
        if not self.is_recording:
            print("Not currently recording")
            return None

        self.is_recording = False

        # If no frames were collected in the processed buffer, try to drain raw frames
        # from the audio capture queue as a fallback (best-effort).
        if len(self.recorded_frames) == 0:
            try:
                drained = []
                while True:
                    frame = self.audio_capture.audio_queue.get_nowait()
                    drained.append(frame)
                    if len(drained) >= 100:
                        break
                if len(drained) > 0:
                    print(f"Drained {len(drained)} raw frames from audio queue as fallback for recording")
                    # convert raw frames (int16) to numpy arrays and use them
                    self.recorded_frames = drained
                else:
                    print("No audio frames recorded and audio queue empty")
                    return None
            except Exception:
                print("No audio frames recorded and unable to drain audio queue")
                return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"enhanced_speech_{timestamp}.wav"
        filepath = os.path.join(self.recording_config.RECORDINGS_DIR, filename)

        try:
            # Concatenate frames which may already be numpy arrays
            audio_data = np.concatenate([np.asarray(f) for f in self.recorded_frames])
            audio_data = np.clip(audio_data, -32768, 32767).astype(np.int16)

            # Simple click/spike suppression: short-duration high-amplitude
            # transients (like mouse clicks) often show up as single-sample
            # spikes. Replace samples that differ from a local median by a
            # threshold with the median value.
            def _remove_spikes(arr, window=5, thresh=3000):
                try:
                    if arr.size < window:
                        return arr
                    pad = window // 2
                    padded = np.pad(arr, (pad, pad), mode='edge')
                    med = np.empty_like(arr)
                    for i in range(arr.size):
                        seg = padded[i:i+window]
                        med[i] = np.median(seg)
                    diff = np.abs(arr - med)
                    spikes = diff > thresh
                    if np.any(spikes):
                        out = arr.copy()
                        out[spikes] = med[spikes]
                        return out
                    return arr
                except Exception:
                    return arr

            try:
                audio_data = _remove_spikes(audio_data, window=5, thresh=3000)
            except Exception:
                pass

            with wave.open(filepath, 'w') as wav_file:
                wav_file.setnchannels(self.config.CHANNELS)
                wav_file.setsampwidth(2)
                # Use the actual rate of the audio capture if available to avoid
                # saving files at the wrong sample rate (which causes robotic/fast playback).
                framerate = getattr(self.audio_capture, 'actual_rate', self.config.RATE)
                wav_file.setframerate(framerate)
                wav_file.writeframes(audio_data.tobytes())

            print(f"Recording saved: {filepath}")
            print(f"Duration: {self.stats['recording_time']:.2f}s, Frames: {len(self.recorded_frames)}")

            self.recorded_frames = []
            return filepath

        except (wave.Error, IOError) as e:
            print(f"Error saving recording: {e}")
            return None

    def get_stats(self):
        return self.stats.copy()

    def cleanup(self):
        if self.is_recording:
            self.stop_recording()
        self.stop_processing()
        self.audio_capture.cleanup()
