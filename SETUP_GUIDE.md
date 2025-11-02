# Quick Setup Guide

## For Windows Users

### Step 1: Install Python
1. Download Python 3.11 from [python.org](https://www.python.org/downloads/)
2. During installation, check "Add Python to PATH"
3. Verify installation: Open Command Prompt and type `python --version`

### Step 2: Install Dependencies

Open Command Prompt in the `speech_enhancement` folder and run:

```bash
pip install numpy scipy scikit-learn matplotlib
```

### Step 3: Install PyAudio (Windows)

PyAudio requires special installation on Windows:

**Option A: Using pipwin (Recommended)**
```bash
pip install pipwin
pipwin install pyaudio
```

**Option B: Download pre-built wheel**
1. Visit: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio
2. Download the wheel matching your Python version (e.g., `PyAudio-0.2.13-cp311-cp311-win_amd64.whl` for Python 3.11 64-bit)
3. Install: `pip install PyAudio-0.2.13-cp311-cp311-win_amd64.whl`

**Option C: Build from source (Advanced)**
```bash
pip install pyaudio
```
Note: This requires Microsoft Visual C++ Build Tools

### Step 4: Run the Application

```bash
python main.py
```

## Creating Standalone Executable

### Method 1: PyInstaller (Recommended)

```bash
pip install pyinstaller

pyinstaller --onefile --windowed --name "SpeechEnhancer" main.py
```

The `.exe` file will be in the `dist/` folder.

### Method 2: Auto-py-to-exe (GUI Tool)

```bash
pip install auto-py-to-exe
auto-py-to-exe
```

Then use the GUI to:
1. Select `main.py` as the script
2. Choose "One File" option
3. Choose "Window Based" (hides console)
4. Click "Convert .py to .exe"

## Troubleshooting

### "No module named 'pyaudio'"
- PyAudio installation failed
- Try the pipwin method above

### "ModuleNotFoundError: No module named 'numpy'"
- Install all requirements: `pip install -r requirements.txt`

### "No audio devices found"
- Check microphone is connected
- Update audio drivers
- Run as Administrator

### High CPU Usage
- Reduce FFT_SIZE in config.py
- Increase CHUNK_SIZE
- Disable visualizations

## Testing the Installation

Run this simple test:

```bash
python -c "import pyaudio; import numpy; import scipy; print('All modules installed successfully!')"
```

If no errors appear, you're ready to go!

## System Requirements

- **OS**: Windows 10/11 (64-bit)
- **RAM**: 4GB minimum, 8GB recommended
- **CPU**: Any modern processor (Intel i3 or equivalent)
- **Audio**: Microphone and speakers/headphones

## Additional Notes

- First run will display all available audio devices
- Use headphones to avoid feedback
- Calibrate in a quiet environment for best results
- Processing works best with 16kHz sample rate
