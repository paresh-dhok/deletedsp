class AudioConfig:
    CHUNK_SIZE = 1024
    FORMAT = 'int16'
    CHANNELS = 1
    RATE = 16000
    FRAME_DURATION_MS = 64

    NOISE_PROFILE_DURATION = 2.0

    BUFFER_SIZE = 4


class DSPConfig:
    FFT_SIZE = 2048
    HOP_LENGTH = 512
    WINDOW_TYPE = 'hann'

    SPECTRAL_FLOOR = 0.002
    NOISE_ALPHA = 0.98
    OVERSUBTRACTION_FACTOR = 2.0

    WIENER_ALPHA = 0.99
    WIENER_MIN_GAIN = 0.1

    VAD_THRESHOLD = 0.03
    VAD_SMOOTHING = 5
    # Lower default threshold to detect speech reliably on typical mic levels
    # (previous value was too high for normalized int16 energy values)
    VAD_DEFAULT_THRESHOLD = 1e-05


class RecordingConfig:
    RECORDINGS_DIR = "recordings"
    AUTO_FILENAME = True
    USE_TIMESTAMP = True
    # Record only when speech is detected (helps avoid table thumps)
    RECORD_SPEECH_ONLY = True
    # VAD probability threshold to consider frame as speech
    RECORD_VAD_THRESHOLD = 0.5
    # Number of pre-roll frames to include before speech onset
    RECORD_PREBUFFER_FRAMES = 3
    # Number of consecutive post-silence frames to keep after speech ends
    RECORD_POST_FRAMES = 5
    # High-pass filter cutoff (Hz) to remove low-frequency thumps
    RECORD_HP_CUTOFF_HZ = 120


class GUIConfig:
    WINDOW_TITLE = "Real-Time Speech Enhancement System"
    WINDOW_WIDTH = 950
    WINDOW_HEIGHT = 750

    UPDATE_INTERVAL = 100

    PLOT_HISTORY = 100
