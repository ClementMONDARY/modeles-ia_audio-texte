"""
Pipeline multimodal — Audio vers Audio
======================================
Enchaîne trois étapes de traitement :

  1. Speech-to-Text  (Whisper)  : transcrit le fichier audio d'entrée en texte.
  2. Text-to-Text    (TinyLlama): génère une réponse textuelle à la transcription.
  3. Text-to-Speech  (Orpheus)  : synthétise la réponse en un nouveau fichier audio.

Usage :
  python main.py <fichier_audio> [options]

Exemples :
  python main.py samples/question.wav
  python main.py samples/question.wav --output reponse.wav --voice leo --play
"""

import argparse
import sys
from pathlib import Path

from stt import load_stt_model, transcribe_audio
from llm import load_llm, generate_response
from tts import load_tts_model, generate_speech, AVAILABLE_VOICES


# ─── Utilitaires ──────────────────────────────────────────────────────────────

def print_section(title: str) -> None:
    bar = "=" * 60
    print(f"\n{bar}\n  {title}\n{bar}")


def play_audio(path: str) -> None:
    """Joue un fichier audio WAV via sounddevice (optionnel)."""
    try:
        import sounddevice as sd
        import soundfile as sf

        data, samplerate = sf.read(path)
        print(f"\n[PLAY] Lecture de {path} ({samplerate} Hz)...")
        sd.play(data, samplerate)
        sd.wait()
    except ImportError:
        print("\n[PLAY] 'sounddevice' non installé — lecture ignorée.")
        print("       Pour activer la lecture : pip install sounddevice")


# ─── Point d'entrée ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline multimodal : Audio → STT → LLM → TTS → Audio",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "audio_input",
        help="Fichier audio d'entrée (wav, mp3, flac…)",
    )
    parser.add_argument(
        "--output",
        default="output.wav",
        help="Fichier WAV de sortie généré par le TTS",
    )
    parser.add_argument(
        "--voice",
        default="tara",
        choices=AVAILABLE_VOICES,
        help="Voix utilisée par le modèle Orpheus",
    )
    parser.add_argument(
        "--play",
        action="store_true",
        help="Jouer l'audio de sortie à la fin du pipeline",
    )
    parser.add_argument(
        "--stt-model",
        default="openai/whisper-large-v3",
        metavar="HF_MODEL",
        help="Modèle Whisper sur HuggingFace",
    )
    parser.add_argument(
        "--llm-model",
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        metavar="HF_MODEL",
        help="Modèle LLM sur HuggingFace",
    )

    args = parser.parse_args()

    if not Path(args.audio_input).is_file():
        print(f"Erreur : fichier introuvable → {args.audio_input}")
        sys.exit(1)

    # ── Étape 1 : Speech-to-Text ───────────────────────────────────────────────
    print_section("ÉTAPE 1 — Speech-to-Text  (Whisper)")

    stt_pipe     = load_stt_model(args.stt_model)
    stt_result   = transcribe_audio(stt_pipe, args.audio_input)
    transcription = stt_result["text"].strip()

    print(f"\n  Transcription : {transcription}")

    # ── Étape 2 : Text-to-Text (LLM) ──────────────────────────────────────────
    print_section("ÉTAPE 2 — Text-to-Text  (TinyLlama)")

    llm_pipe = load_llm(args.llm_model)
    response = generate_response(llm_pipe, transcription)

    print(f"\n  Réponse : {response}")

    # ── Étape 3 : Text-to-Speech (Orpheus) ────────────────────────────────────
    print_section("ÉTAPE 3 — Text-to-Speech  (Orpheus)")

    tokenizer, tts_model, snac_model, device = load_tts_model()
    output_path = generate_speech(
        tokenizer, tts_model, snac_model, device,
        text=response,
        voice=args.voice,
        output_path=args.output,
    )

    if output_path is None:
        print("\nErreur lors de la génération audio. Arrêt du pipeline.")
        sys.exit(1)

    # ── Récapitulatif ──────────────────────────────────────────────────────────
    print_section("Résultat final")
    print(f"  Entrée audio  : {args.audio_input}")
    print(f"  Transcription : {transcription}")
    print(f"  Réponse LLM   : {response}")
    print(f"  Sortie audio  : {output_path}")

    # ── Lecture optionnelle ────────────────────────────────────────────────────
    if args.play:
        play_audio(output_path)


if __name__ == "__main__":
    main()
