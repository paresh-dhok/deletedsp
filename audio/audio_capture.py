import pyaudio
import numpy as np
import threading
import queue
from typing import Optional, Callable
from config import AudioConfig


class AudioCapture:
    def __init__(self):
        self.config = AudioConfig()
        self.audio = pyaudio.PyAudio()
        self.stream: Optional[pyaudio.Stream] = None
        self.is_running = False
        self.audio_queue = queue.Queue(maxsize=self.config.BUFFER_SIZE)
        self.output_queue = queue.Queue(maxsize=self.config.BUFFER_SIZE)
        self.callback_function: Optional[Callable] = None

    def list_devices(self):
        print("\n=== Available Audio Devices ===")
        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            print(f"Device {i}: {info['name']}")
            print(f"  Input Channels: {info['maxInputChannels']}")
            print(f"  Output Channels: {info['maxOutputChannels']}")
            print(f"  Default Sample Rate: {info['defaultSampleRate']}")
            print()

    def get_default_input_device(self):
        try:
            return self.audio.get_default_input_device_info()['index']
        except:
            return 0

    def get_default_output_device(self):
        try:
            return self.audio.get_default_output_device_info()['index']
        except:
            return 0

    def _audio_callback(self, in_data, frame_count, time_info, status):
        if status:
            print(f"Audio callback status: {status}")

        audio_data = np.frombuffer(in_data, dtype=np.int16)

        try:
            self.audio_queue.put_nowait(audio_data)
        except queue.Full:
            pass

        try:
            output_data = self.output_queue.get_nowait()
            output_bytes = output_data.astype(np.int16).tobytes()
        except queue.Empty:
            output_bytes = np.zeros(frame_count, dtype=np.int16).tobytes()

        return (output_bytes, pyaudio.paContinue)

    def start(self, input_device_index=None, output_device_index=None):
        if self.is_running:
            print("Audio capture already running")
            return

        if input_device_index is None:
            input_device_index = self.get_default_input_device()

        if output_device_index is None:
            output_device_index = self.get_default_output_device()

        try:
            # Determine whether selected devices support input/output
            # Default: input enabled, output disabled (we don't play back by default)
            input_flag = True
            output_flag = False

            try:
                input_info = self.audio.get_device_info_by_index(int(input_device_index))
                input_flag = input_info.get('maxInputChannels', 0) > 0
            except Exception:
                input_flag = True

            if output_device_index is not None:
                try:
                    output_info = self.audio.get_device_info_by_index(int(output_device_index))
                    output_flag = output_info.get('maxOutputChannels', 0) > 0
                except Exception:
                    output_flag = False

            open_kwargs = dict(
                format=pyaudio.paInt16,
                channels=self.config.CHANNELS,
                rate=self.config.RATE,
                frames_per_buffer=self.config.CHUNK_SIZE,
                stream_callback=self._audio_callback
            )

            # Only request input/output device indices when that direction is enabled
            if input_flag:
                open_kwargs['input'] = True
                open_kwargs['input_device_index'] = input_device_index
            else:
                open_kwargs['input'] = False

            if output_flag:
                open_kwargs['output'] = True
                open_kwargs['output_device_index'] = output_device_index
            else:
                open_kwargs['output'] = False

            self.stream = self.audio.open(**open_kwargs)

            # Record the actual rate used by the stream. Some devices don't support
            # the requested rate and PyAudio may open a stream with a different
            # underlying device rate; store it so the rest of the pipeline can
            # save files with the correct sample rate.
            try:
                # Many PortAudio/PyAudio builds expose sample rate via stream._rate
                actual_rate = int(getattr(self.stream, '_rate', self.config.RATE))
            except Exception:
                actual_rate = self.config.RATE
            # Update config to reflect actual rate
            try:
                self.config.RATE = actual_rate
            except Exception:
                pass
            self.actual_rate = actual_rate

            self.is_running = True
            self.stream.start_stream()
            print(f"Audio capture started: {self.config.RATE}Hz, {self.config.CHANNELS} channel(s), input={input_flag}, output={output_flag}")
        except Exception as e:
            # If opening at the requested rate failed, try the device's default rate
            try:
                device_info = self.audio.get_device_info_by_index(int(input_device_index))
                fallback_rate = int(device_info.get('defaultSampleRate', self.config.RATE))
            except Exception:
                fallback_rate = self.config.RATE

            if fallback_rate != self.config.RATE:
                try:
                    open_kwargs['rate'] = fallback_rate
                    self.stream = self.audio.open(**open_kwargs)
                    try:
                        actual_rate = int(getattr(self.stream, '_rate', fallback_rate))
                    except Exception:
                        actual_rate = fallback_rate
                    try:
                        self.config.RATE = actual_rate
                    except Exception:
                        pass
                    self.actual_rate = actual_rate
                    self.is_running = True
                    self.stream.start_stream()
                    print(f"Audio capture started (fallback): {self.config.RATE}Hz, {self.config.CHANNELS} channel(s), input={input_flag}, output={output_flag}")
                    return
                except Exception as e2:
                    print(f"Error starting audio capture with fallback rate {fallback_rate}: {e2}")

            print(f"Error starting audio capture: {e}")
            raise

    def start_file_processing(self, filepath):
        if self.is_running:
            print("Processing already running")
            return
        self.is_running = True
        self.file_processing_thread = threading.Thread(target=self._file_processing_loop, args=(filepath,), daemon=True)
        self.file_processing_thread.start()

    def _file_processing_loop(self, filepath):
        try:
            with wave.open(filepath, 'rb') as wf:
                self.config.RATE = wf.getframerate()
                self.config.CHANNELS = wf.getnchannels()
                
                while self.is_running:
                    data = wf.readframes(self.config.CHUNK_SIZE)
                    if not data:
                        break
                    audio_data = np.frombuffer(data, dtype=np.int16)
                    self.audio_queue.put(audio_data)
        except FileNotFoundError:
            print(f"Error: File not found at {filepath}")
        except Exception as e:
            print(f"Error processing file: {e}")
        finally:
            self.audio_queue.put(None) # Signal end of file

    def stop(self):
        if not self.is_running:
            return

        self.is_running = False

        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        if self.file_processing_thread:
            self.file_processing_thread.join(timeout=2.0)


        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except queue.Empty:
                break

        print("Audio capture stopped")

    def read_audio(self, timeout=0.1):
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def write_audio(self, audio_data):
        try:
            self.output_queue.put_nowait(audio_data)
        except queue.Full:
            pass

    def cleanup(self):
        self.stop()
        self.audio.terminate()
