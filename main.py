import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from audio_processor import AudioProcessor
from gui import MainWindow


def main():
    print("=" * 60)
    print("Real-Time Speech Enhancement System")
    print("DSP Project - BTech ECE AIML")
    print("=" * 60)
    print()

    try:
        audio_processor = AudioProcessor()

        print("Available audio devices:")
        audio_processor.audio_capture.list_devices()

        gui = MainWindow(audio_processor)

        gui.run()

    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'audio_processor' in locals() and audio_processor:
            audio_processor.cleanup()
        print("Application closed")


if __name__ == "__main__":
    main()
