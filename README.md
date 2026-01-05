# Voice Clone

AI Voice Generation app built with Python and Gradio using the XTTS v2 model. The interface supports multilingual text-to-speech, optional text file upload, and optional voice sample cloning for a friendly demo that can run locally or be deployed to Hugging Face Spaces.

## Features
- Text box for entering a script
- Optional .txt upload to provide longer prompts
- Optional short voice sample (.wav/.mp3) for voice cloning
- Language dropdown with multilingual support
- "Generate Speech" button with clear status messaging
- Built-in audio player and download link for the generated file
- Model loads once at startup for smoother performance

## Getting Started
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the app:
   ```bash
   python app.py
   ```
3. Open the provided Gradio link in your browser and start generating speech.

The default server port is `7860`. Set the `PORT` environment variable to override it (useful for Spaces).

## Notes
- The demo targets the [`tts_models/multilingual/multi-dataset/xtts_v2`](https://huggingface.co/coqui/XTTS-v2) model. A GPU is recommended for faster inference, but the app will fall back to CPU if necessary.
- The voice sample works best with a clean 5–10 second clip. Longer clips will be truncated by Gradio's file size limits.
- When using file uploads, Gradio provides file paths to the backend; errors like "Voice sample not found" usually mean the upload was interrupted. Re-upload and try again.
- Generated audio files are written to your system temp directory with unique names so the player and download link continue to work after each run.
