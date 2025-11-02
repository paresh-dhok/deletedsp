@echo off
echo ========================================
echo Building Speech Enhancement System EXE
echo ========================================
echo.

echo Step 1: Installing PyInstaller...
pip install pyinstaller
echo.

echo Step 2: Building executable...
rem Using more robust options to ensure all dependencies are included.
pyinstaller --onefile ^
    --windowed ^
    --name "SpeechEnhancer" ^
    --icon=NONE ^
    --hidden-import "sklearn.utils._cython_blas" ^
    main.py
echo.

echo Step 3: Build complete!
echo.
echo The executable is located in: dist\SpeechEnhancer.exe
echo.
echo You can now distribute this .exe file to run on any Windows machine
echo without requiring Python installation.
echo.
pause
