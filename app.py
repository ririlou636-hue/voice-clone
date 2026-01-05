"""Gradio app for multilingual voice cloning using XTTS v2.

The app supports text input, optional text file upload, optional
speaker reference audio, multiple languages, and produces a downloadable
speech audio file with an inline player.
"""
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple
from uuid import uuid4

import gradio as gr
import torch
from TTS.api import TTS

# Configure logging early so that load/generate status is visible in logs.
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)

# The model is loaded once and reused for all requests.
_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
_TTS_MODEL: Optional[TTS] = None

# Commonly supported languages for XTTS v2. The list can be extended if needed.
LANGUAGES = [
    "en",
    "es",
    "fr",
    "de",
    "it",
    "pt",
    "pl",
    "tr",
    "ru",
    "ar",
    "hi",
    "zh",
    "ja",
    "ko",
]


def load_model() -> TTS:
    """Load the XTTS model once and reuse it.

    Returns
    -------
    TTS
        Loaded model instance bound to CPU/GPU.
    """

    global _TTS_MODEL
    if _TTS_MODEL is not None:
        return _TTS_MODEL

    device = "cuda" if torch.cuda.is_available() else "cpu"
    LOGGER.info("Loading model %s on %s...", _MODEL_NAME, device)
    _TTS_MODEL = TTS(model_name=_MODEL_NAME, progress_bar=False).to(device)
    LOGGER.info("Model loaded successfully")
    return _TTS_MODEL


def _resolve_text(text: str, text_file: Optional[str]) -> str:
    """Return the text to speak, preferring direct input over file content."""

    direct_text = text.strip() if text else ""
    if direct_text:
        return direct_text

    if text_file:
        try:
            file_content = Path(text_file).read_text(encoding="utf-8")
            return file_content.strip()
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Could not read text file: {exc}") from exc

    return ""


def synthesize_speech(
    text: str,
    text_file: Optional[str],
    voice_sample: Optional[str],
    language: str,
) -> Tuple[str, Optional[str]]:
    """Generate speech audio from the provided inputs.

    Returns a tuple of (status message, audio filepath).
    """

    try:
        resolved_text = _resolve_text(text, text_file)
    except ValueError as exc:
        return str(exc), None

    if not resolved_text:
        return "Please provide text or upload a .txt file to generate speech.", None

    model = load_model()

    speaker_wav: Optional[str] = None
    if voice_sample:
        speaker_path = Path(voice_sample)
        if speaker_path.exists():
            speaker_wav = str(speaker_path)
        else:
            return "Voice sample not found. Please re-upload and try again.", None

    # Create a temporary directory for the generated file.
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "speech.wav"
        LOGGER.info("Generating speech: lang=%s, voice=%s", language, bool(speaker_wav))
        try:
            model.tts_to_file(
                text=resolved_text,
                file_path=str(out_path),
                speaker_wav=speaker_wav,
                language=language,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Speech generation failed")
            return f"Error generating speech: {exc}", None

        # Move the file to a stable path so Gradio can serve it.
        permanent_path = Path(tempfile.gettempdir()) / f"speech_{uuid4().hex}.wav"
        out_path.replace(permanent_path)
        return "Speech generated successfully!", str(permanent_path)


def build_interface() -> gr.Blocks:
    """Create the Gradio UI layout."""

    with gr.Blocks(title="AI Voice Generator", css="footer {display: none;}") as demo:
        gr.Markdown(
            """
            # AI Voice Generator
            Enter text or upload a .txt file, optionally include a short voice sample for cloning,
            choose a language, and generate natural-sounding speech.
            """
        )

        with gr.Row():
            text_input = gr.Textbox(
                label="Text to Speak",
                placeholder="Type your script here...",
                lines=6,
            )
            text_file = gr.File(
                label="Upload .txt (optional)",
                file_types=[".txt"],
                type="filepath",
            )

        with gr.Row():
            voice_sample = gr.Audio(
                label="Voice Sample (optional, 5-10 seconds)",
                type="filepath",
            )
            language = gr.Dropdown(
                LANGUAGES,
                value="en",
                label="Language",
                info="Select the language/accent for synthesis.",
            )

        generate_btn = gr.Button("Generate Speech", variant="primary")

        status = gr.Markdown("Ready.")
        audio_output = gr.Audio(label="Generated Audio", type="filepath", interactive=False)
        download_output = gr.File(label="Download Audio")

        def _generate(text, file_obj, voice_obj, lang):
            msg, audio_path = synthesize_speech(text, file_obj, voice_obj, lang)
            if audio_path:
                return msg, audio_path, audio_path
            return msg, None, None

        generate_btn.click(
            _generate,
            inputs=[text_input, text_file, voice_sample, language],
            outputs=[status, audio_output, download_output],
        )

    return demo


def main() -> None:
    demo = build_interface()
    demo.queue(api_open=False).launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", 7860)))


if __name__ == "__main__":
    main()
