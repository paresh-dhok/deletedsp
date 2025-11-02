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

- **Python 3.8 or higher** (Python 3.9+ recommended)
- **Windows OS** (Windows 10/11 currently supported)
- **Microphone** (built-in or external USB microphone)
- **Speakers/Headphones** (recommended to avoid feedback loops)

### Step 1: Install Python

If you don't have Python installed:
1. Download Python from [python.org](https://www.python.org/downloads/)
2. During installation, **check "Add Python to PATH"**
3. Verify installation: Open Command Prompt and type:
   ```bash
   python --version
   ```

### Step 2: Install Python Dependencies

Navigate to the project directory and install all required packages:

```bash
cd speech_enhancement
pip install -r requirements.txt
```

**Required Dependencies:**
- `numpy` - Numerical computing library
- `scipy` - Scientific computing library (for signal processing)
- `scikit-learn` - Machine learning utilities (for VAD features)
- `matplotlib` - Plotting library (for GUI visualization)
- `pyaudio` - Audio I/O library (see Step 3 for installation)

### Step 3: Install PyAudio on Windows

PyAudio requires special installation on Windows due to PortAudio dependencies:

**Method 1: Using pipwin (Recommended - Easiest)**
```bash
pip install pipwin
pipwin install pyaudio
```

**Method 2: Direct wheel installation**
1. Visit [Unofficial Windows Binaries for Python Extension Packages](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio)
2. Download the wheel file matching your Python version:
   - For Python 3.11 64-bit: `PyAudio‑0.2.13‑cp311‑cp311‑win_amd64.whl`
   - For Python 3.10 64-bit: `PyAudio‑0.2.13‑cp310‑cp310‑win_amd64.whl`
   - For Python 3.9 64-bit: `PyAudio‑0.2.13‑cp39‑cp39‑win_amd64.whl`
3. Install the downloaded wheel:
   ```bash
   pip install PyAudio‑0.2.13‑cp311‑cp311‑win_amd64.whl
   ```
   (Replace with your downloaded filename)

**Method 3: Build from source (Advanced)**
```bash
pip install pyaudio
```
*Note: This requires Microsoft Visual C++ Build Tools*

### Step 4: Verify Installation

Test that all dependencies are installed correctly:

```bash
python -c "import pyaudio; import numpy; import scipy; import matplotlib; print('All dependencies installed successfully!')"
```

If no errors appear, you're ready to run the application!

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

## System Requirements

### Minimum Requirements
- **OS**: Windows 10 (64-bit) or Windows 11
- **RAM**: 4GB minimum, 8GB recommended
- **CPU**: Intel Core i3 or equivalent (modern dual-core processor)
- **Storage**: 500MB free space for installation and recordings
- **Audio**: Microphone input device

### Recommended Requirements
- **RAM**: 8GB or more
- **CPU**: Intel Core i5 or equivalent (quad-core processor)
- **Storage**: 2GB free space (for storing recordings)
- **Audio**: High-quality USB microphone or headset

## Creating Windows Executable

### Prerequisites for Building Executable

1. Install PyInstaller:
   ```bash
   pip install pyinstaller
   ```

### Basic Executable Build

```bash
pyinstaller --onefile --windowed --name "SpeechEnhancer" main.py
```

The executable will be in the `dist/` folder.

### Advanced PyInstaller Options (Recommended)

For better performance, smaller file size, and inclusion of all dependencies:

```bash
pyinstaller --onefile ^
    --windowed ^
    --name "SpeechEnhancer" ^
    --add-data "config.py;." ^
    --hidden-import numpy ^
    --hidden-import scipy ^
    --hidden-import scikit-learn ^
    --hidden-import matplotlib ^
    --hidden-import pyaudio ^
    --hidden-import sklearn.utils._cython_blas ^
    --collect-all matplotlib ^
    main.py
```

**Note**: The executable will be large (~200-300MB) as it bundles Python and all dependencies. This allows distribution to machines without Python installed.

## Dependencies

The project requires the following Python packages (automatically installed via `requirements.txt`):

### Core Dependencies
- **numpy** (>=1.19.0) - Array operations and numerical computations
- **scipy** (>=1.5.0) - Signal processing functions (FFT, filters, windows)
- **scikit-learn** (>=0.23.0) - Machine learning utilities for VAD features
- **matplotlib** (>=3.3.0) - Real-time plotting and visualization in GUI
- **pyaudio** (>=0.2.11) - Cross-platform audio I/O library

### Installation Verification

After installation, verify all packages:
```bash
python -c "import numpy, scipy, sklearn, matplotlib, pyaudio; print('✓ All packages installed')"
```

## Configuration

Edit `config.py` to customize system behavior:

### Audio Settings
- `CHUNK_SIZE`: Audio buffer size in samples (default: 1024)
  - Larger values = more latency but better stability
  - Smaller values = lower latency but more CPU usage
- `RATE`: Sample rate in Hz (default: 16000)
  - 16kHz is standard for speech processing
  - Higher rates (32kHz, 44.1kHz) increase CPU usage
- `CHANNELS`: Number of audio channels (default: 1 - mono)
  - Set to 2 for stereo (requires code modifications)
- `FRAME_DURATION_MS`: Frame duration in milliseconds (default: 64)
- `BUFFER_SIZE`: Audio queue buffer size (default: 4)

### DSP Settings
- `FFT_SIZE`: FFT window size (default: 2048)
  - Larger = better frequency resolution, more CPU usage
  - Smaller = faster processing, lower frequency resolution
- `HOP_LENGTH`: Hop length for FFT (default: 512)
  - Smaller = smoother output but more processing
- `WINDOW_TYPE`: Window function (default: 'hann' - Hanning window)
  - Options: 'hann', 'hamming', 'blackman', etc.
- `OVERSUBTRACTION_FACTOR`: Noise reduction aggressiveness (default: 2.0)
  - Higher = more noise removal (may cause artifacts)
  - Lower = less aggressive noise removal
- `SPECTRAL_FLOOR`: Minimum gain to prevent over-suppression (default: 0.002)
  - Prevents complete signal suppression in quiet bands
- `NOISE_ALPHA`: Adaptive noise estimation smoothing (default: 0.98)
  - Higher = slower adaptation to noise changes

### Wiener Filter Settings
- `WIENER_ALPHA`: Wiener filter adaptation rate (default: 0.99)
- `WIENER_MIN_GAIN`: Minimum gain floor (default: 0.1)
  - Prevents complete signal suppression

### VAD (Voice Activity Detection) Settings
- `VAD_THRESHOLD`: Energy threshold for speech detection (default: 0.03)
- `VAD_SMOOTHING`: Smoothing window size in frames (default: 5)
  - Larger = more stable but slower response
- `VAD_DEFAULT_THRESHOLD`: Fallback threshold (default: 1e-05)

### Recording Settings
- `RECORDINGS_DIR`: Directory to save recordings (default: "recordings")
  - Automatically created if it doesn't exist
- `AUTO_FILENAME`: Use auto-generated filenames (default: True)
- `USE_TIMESTAMP`: Include timestamp in filename (default: True)
- `RECORD_SPEECH_ONLY`: Only record when speech is detected (default: True)
  - Helps avoid recording silence and noise
- `RECORD_VAD_THRESHOLD`: VAD probability threshold for recording (default: 0.5)
  - Higher = only record clear speech
- `RECORD_PREBUFFER_FRAMES`: Frames to include before speech onset (default: 3)
- `RECORD_POST_FRAMES`: Frames to keep after speech ends (default: 5)
- `RECORD_HP_CUTOFF_HZ`: High-pass filter cutoff to remove thumps (default: 120 Hz)

### GUI Settings
- `WINDOW_TITLE`: Application window title
- `WINDOW_WIDTH`: Window width in pixels (default: 950)
- `WINDOW_HEIGHT`: Window height in pixels (default: 750)
- `UPDATE_INTERVAL`: GUI update interval in milliseconds (default: 100)
- `PLOT_HISTORY`: Number of data points to display in plots (default: 100)

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

## Recent Improvements & Bug Fixes

### Code Refinements (Latest Update)

The codebase has been thoroughly refined with comprehensive error handling and stability improvements:

#### 1. **Enhanced Error Handling**
- Fixed AttributeError risks by properly initializing all thread variables
- Replaced bare exception clauses with specific exception types for better debugging
- Added comprehensive null/empty frame checks throughout the processing pipeline
- Improved error messages for easier troubleshooting

#### 2. **Edge Case Protection**
- Added validation for empty audio frames at all processing stages
- Enhanced device detection with individual device error handling
- Improved concatenation safety in noise profile creation
- Added type conversion safety checks to prevent runtime errors

#### 3. **Runtime Fixes (Previous Updates)**
- **Sample-rate negotiation**: Automatically falls back to device's default sample rate if requested rate is unsupported. Saved WAV files use the actual rate to prevent robotic/chipmunk playback.

- **Click/spike suppression**: Median-based filter removes high-amplitude transients (mouse clicks, table thumps) from recordings before saving.

- **Device/exception handling**: Robust error handling prevents crashes when querying audio devices. Individual device failures don't stop device enumeration.

- **GUI parameter naming**: Standardized API calls use `input_device=` parameter consistently.

### Reporting Issues

If you encounter issues, please provide:

1. **Device Information**: Confirm the input device selected is your physical microphone (not 'Stereo Mix' or output devices)
2. **Console Output**: Note the actual sample rate printed when the app starts
3. **Error Messages**: Include full error traceback if available
4. **System Info**: Windows version, Python version, and audio device model
5. **Steps to Reproduce**: Detailed steps to trigger the issue


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
