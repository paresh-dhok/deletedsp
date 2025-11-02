# Real-Time Speech Enhancement System

A comprehensive speech enhancement system designed for real-time audio processing with custom DSP algorithms and machine learning-based voice activity detection.

## Project Overview

This system captures real-time audio from your microphone, suppresses ambient noise, and enhances speech using advanced Digital Signal Processing (DSP) techniques and Machine Learning (ML) algorithms.

### Key Features

- **Real-Time Processing**: Low-latency audio capture with Hanning window windowing
- **Spectral Subtraction**: Custom algorithm for noise suppression
- **Wiener Filtering**: Advanced speech enhancement
- **Voice Activity Detection (VAD)**: ML-based detection using energy, zero-crossing rate, and spectral features
- **Adaptive Noise Estimation**: Continuously updates noise profile during silent periods
- **Audio Recording**: Save enhanced speech to WAV files without playback feedback
- **Input Device Selection**: Choose from available microphones in the GUI
- **GUI Interface**: User-friendly control panel with real-time visualization
- **Bypass Mode**: Toggle processing on/off for A/B comparison

## Technical Architecture

### Modules

1. **audio/**: Audio capture and playback using PyAudio
2. **dsp/**: DSP algorithms (Spectral Subtraction, Wiener Filter)
3. **ml/**: Machine learning components (Voice Activity Detector)
4. **gui/**: Tkinter-based graphical interface
5. **utils/**: Audio utilities and helper functions

### Algorithms

#### Spectral Subtraction
- Estimates noise spectrum from silent periods
- Subtracts noise from noisy speech in frequency domain
- Uses oversubtraction factor for better noise reduction
- Adaptive mode continuously updates noise profile

#### Wiener Filter
- Optimal filter for noise reduction
- Computes gain based on signal-to-noise ratio
- Minimizes mean square error between clean and enhanced speech
- Adaptive mode for dynamic noise environments

#### Voice Activity Detection
- Energy-based detection with adaptive threshold
- Zero-crossing rate analysis
- Spectral centroid computation
- Smoothing filter for stable decisions

## Installation

### Prerequisites

- Python 3.8 or higher
- Windows OS (currently supported)
- Microphone and speakers/headphones

### Step 1: Install Python Dependencies

```bash
cd speech_enhancement
pip install -r requirements.txt
```

### Step 2: Install PyAudio on Windows

PyAudio can be tricky on Windows. Use one of these methods:

**Method 1: Pre-built wheel**
```bash
pip install pipwin
pipwin install pyaudio
```

**Method 2: Direct wheel installation**
Download the appropriate wheel from [Unofficial Windows Binaries](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio) and install:
```bash
pip install PyAudio‑0.2.13‑cp311‑cp311‑win_amd64.whl
```

## Usage

### Running the Application

```bash
cd speech_enhancement
python main.py
```

### Step-by-Step Guide

1. **Launch the Application**
   - Run `python main.py`
   - The GUI window will open with all available audio devices listed

2. **Select Input Device**
   - Choose your microphone from the "Audio Device Selection" dropdown
   - The default device is marked with "(Default)"
   - Select the specific mic you want to use

3. **Calibrate Noise Profile**
   - Click "Calibrate Noise" button
   - Remain silent for 2 seconds
   - The system captures ambient noise profile for better enhancement

4. **Start Processing**
   - Click "Start Processing"
   - Speak into your microphone
   - Enhanced speech is processed but NOT played back (no feedback loop)

5. **Record Enhanced Audio**
   - Click "Start Recording" to save the enhanced speech to a file
   - Click "Stop Recording" when done
   - Recording indicator (REC) shows in red while recording
   - Enhanced audio is saved to `recordings/` folder as WAV files

6. **Adjust Settings**
   - Toggle Spectral Subtraction on/off
   - Toggle Wiener Filter on/off
   - Enable/disable Adaptive Noise Estimation

7. **Monitor Performance**
   - View real-time speech probability graph
   - Check processing time in milliseconds
   - Track frame statistics and recording duration

8. **Bypass Mode**
   - Click "Toggle Bypass" to disable processing
   - Use for A/B comparison or system testing

## Creating Windows Executable

### Using PyInstaller

```bash
pip install pyinstaller

pyinstaller --onefile --windowed --name "SpeechEnhancer" main.py
```

The executable will be in the `dist/` folder.

### Advanced PyInstaller Options

For better performance and smaller file size:

```bash
pyinstaller --onefile ^
    --windowed ^
    --name "SpeechEnhancer" ^
    --add-data "config.py;." ^
    --hidden-import numpy ^
    --hidden-import scipy ^
    --hidden-import matplotlib ^
    main.py
```

## Configuration

Edit `config.py` to customize:

### Audio Settings
- `CHUNK_SIZE`: Audio buffer size (default: 1024)
- `RATE`: Sample rate in Hz (default: 16000)
- `CHANNELS`: Number of channels (default: 1 - mono)
- `FRAME_DURATION_MS`: Frame duration in milliseconds (default: 64)

### DSP Settings
- `FFT_SIZE`: FFT window size (default: 2048)
- `HOP_LENGTH`: Hop length for FFT (default: 512)
- `WINDOW_TYPE`: Window function applied (default: 'hann' - Hanning window)
- `OVERSUBTRACTION_FACTOR`: Noise reduction aggressiveness (default: 2.0)
- `SPECTRAL_FLOOR`: Minimum gain to prevent over-suppression (default: 0.002)
- `NOISE_ALPHA`: Adaptive noise estimation smoothing factor (default: 0.98)

### VAD Settings
- `VAD_THRESHOLD`: Energy threshold for speech detection (default: 0.03)
- `VAD_SMOOTHING`: Smoothing window size (default: 5 frames)

### Recording Settings
- `RECORDINGS_DIR`: Directory to save recordings (default: "recordings")
- `AUTO_FILENAME`: Use auto-generated filenames with timestamps (default: True)
- `USE_TIMESTAMP`: Include timestamp in filename (default: True)

## Troubleshooting

### No Audio Devices Found
- Check microphone is connected
- Verify microphone permissions in Windows Settings
- Restart the application

### Audio Feedback/Echo
- Use headphones instead of speakers
- Reduce output volume
- Increase distance between mic and speakers

### High Processing Time
- Reduce `FFT_SIZE` in config.py
- Increase `CHUNK_SIZE` (with latency trade-off)
- Disable one of the processing algorithms

### Poor Noise Reduction
- Recalibrate with better noise profile (ensure quiet environment)
- Adjust `OVERSUBTRACTION_FACTOR` (increase for more aggressive noise removal)
- Enable adaptive noise estimation for dynamic noise environments
- Verify Hanning window is being used (`WINDOW_TYPE = 'hann'`)

### Recording Issues
- Ensure `recordings/` directory has write permissions
- Check disk space for WAV file storage
- Verify processing is running before attempting to record
- Files are timestamped and saved automatically

## Runtime notes (recent fixes)

The codebase includes a few runtime fixes added in November 2025 to address common audio capture and recording problems:

- Sample-rate negotiation: The audio capture now attempts to open the device at the configured rate (e.g., 16000 Hz). If the device does not support that rate, the code will fall back to the device's default sample rate and record the actual rate used. Saved WAV files use this actual rate to avoid robotic / chipmunk playback caused by sample-rate mismatches.

- Click/spike suppression: Short, high-amplitude transients (for example, mouse clicks) can appear in recordings as audible clicks. A small median-based spike suppression filter is applied before saving WAV files to reduce these artifacts. This is a light-weight fix — if clicks persist, consider lowering mic gain or enabling stronger de-clicking/resampling.

- Device/exception handling: GUI device detection and error handling have been made more permissive. PyAudio/device errors are caught more generally (OSError / Exception) to avoid crashes caused by platform-specific PyAudio exceptions.

- GUI parameter naming: The GUI now calls the audio processing start method with `input_device=` (the AudioProcessor API), rather than a legacy `input_device_index=` keyword. If integrating other code with the GUI, call `AudioProcessor.start_processing(input_device=...)`.

If you encounter any remaining issues after these fixes, please:

1. Confirm the input device selected in the GUI is your physical microphone (not 'Stereo Mix' or system output devices).
2. Note the console output when the app starts — it prints the actual sample rate used by the audio capture; paste that along with any problematic WAV file metadata when asking for help.


## Project Structure

```
speech_enhancement/
├── main.py                    # Application entry point
├── config.py                  # Configuration settings
├── audio_processor.py         # Main processing pipeline
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── audio/
│   ├── __init__.py
│   └── audio_capture.py       # PyAudio wrapper
├── dsp/
│   ├── __init__.py
│   ├── spectral_subtraction.py
│   └── wiener_filter.py
├── ml/
│   ├── __init__.py
│   └── voice_activity_detector.py
├── gui/
│   ├── __init__.py
│   └── main_window.py         # Tkinter GUI
└── utils/
    ├── __init__.py
    └── audio_utils.py         # Helper functions
```

## Technical Details for DSP Project Report

### Signal Processing Pipeline

1. **Audio Acquisition**: 16kHz mono audio capture at 1024 samples/frame
2. **Windowing**: Hann window applied before FFT
3. **FFT Transform**: 2048-point FFT for frequency analysis
4. **Noise Estimation**: Statistical estimation from silent periods
5. **Spectral Subtraction**: Magnitude spectrum modification
6. **Wiener Filtering**: Optimal gain computation
7. **IFFT**: Reconstruction to time domain
8. **Overlap-Add**: Smooth frame transitions

### Machine Learning Component

The Voice Activity Detector uses a feature-based approach:
- **Features**: Energy, Zero-Crossing Rate, Spectral Centroid
- **Classification**: Rule-based threshold with adaptive calibration
- **Post-processing**: Temporal smoothing for stability

### Performance Metrics

- **Latency**: ~64ms (configurable)
- **Processing Time**: 2-5ms per frame (depends on hardware)
- **Sample Rate**: 16kHz
- **Bit Depth**: 16-bit

## Technical Implementation Details

### Hanning Window
- Applied to all frames before FFT processing
- Reduces spectral leakage in frequency domain analysis
- Smooth windowing improves DSP algorithm accuracy
- Configured in `DSPConfig.WINDOW_TYPE`

### No Audio Playback (Feedback Prevention)
- Audio is NOT played back during processing
- Only records/processes the input signal
- Eliminates feedback loops that degrade quality
- Enable recording to save enhanced output

### Input Device Selection
- Dynamically detects all available audio input devices
- Dropdown menu in GUI for easy selection
- Default device marked with "(Default)"
- Device change requires restart of processing

## Future Enhancements

- Support for Linux and macOS
- Deep learning-based noise reduction (Neural Networks)
- Multi-channel audio support (stereo, surround)
- Real-time frequency domain visualization (spectrograms)
- Save/load noise profiles for different environments
- Batch processing for multiple audio files

## License

This project is for educational purposes (DSP Course Project).

## Authors

BTech ECE AIML Student - Third Year
Digital Signal Processing Project

## Acknowledgments

- DSP course instructors and teaching assistants
- Open-source Python audio community
- NumPy and SciPy contributors
