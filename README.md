# Pipeline multimodal — Audio & Texte

Pipeline local enchaînant trois modèles d'IA pour transformer un fichier audio en réponse vocale :

```
Fichier audio  ──►  Speech-to-Text  ──►  LLM  ──►  Text-to-Speech  ──►  Fichier audio
   (entrée)          (Whisper)        (TinyLlama)     (Orpheus)           (sortie)
```

---

## Modèles utilisés

| Étape | Modèle | Description |
|---|---|---|
| Speech-to-Text | [`openai/whisper-large-v3`](https://huggingface.co/openai/whisper-large-v3) | Transcription multilingue (99 langues) |
| Text-to-Text | [`TinyLlama/TinyLlama-1.1B-Chat-v1.0`](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) | Génération de réponse (LLM compact) |
| Text-to-Speech | [`canopylabs/orpheus-3b-0.1-ft`](https://huggingface.co/canopylabs/orpheus-3b-0.1-ft) | Synthèse vocale naturelle via codec SNAC |

---

## Prérequis

- Python 3.10 ou supérieur
- GPU NVIDIA recommandé (≥ 8 Go de VRAM) ; fonctionne sur CPU mais lentement
- [PyTorch](https://pytorch.org/get-started/locally/) installé selon votre configuration CUDA

## Installation

```bash
# 1. Cloner le dépôt
git clone <url-du-depot>
cd modeles-ia_audio-texte

# 2. Créer un environnement virtuel
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

# 3. Installer PyTorch (adapter selon votre version CUDA)
# https://pytorch.org/get-started/locally/
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Installer les dépendances du projet
pip install -r requirements.txt
```

---

## Utilisation

### Commande de base

```bash
python main.py samples/mon_audio.wav
```

L'audio de sortie est sauvegardé dans `output.wav`.

### Options disponibles

```
python main.py <audio_input> [options]

Arguments :
  audio_input           Fichier audio d'entrée (wav, mp3, flac…)

Options :
  --output OUTPUT       Fichier WAV de sortie (défaut : output.wav)
  --voice VOICE         Voix TTS : tara, leo, luna, stella, ryan, jess
                        (défaut : tara)
  --play                Jouer l'audio de sortie après la génération
  --stt-model HF_MODEL  Modèle Whisper alternatif (défaut : whisper-large-v3)
  --llm-model HF_MODEL  Modèle LLM alternatif (défaut : TinyLlama-1.1B-Chat)
```

## Structure du projet

```
modeles-ia_audio-texte/
├── main.py          # Point d'entrée — orchestre les trois étapes
├── stt.py           # Étape 1 : Speech-to-Text (Whisper)
├── llm.py           # Étape 2 : Text-to-Text (TinyLlama)
├── tts.py           # Étape 3 : Text-to-Speech (Orpheus + SNAC)
├── requirements.txt # Dépendances Python
└── samples/         # Dossier pour les fichiers audio d'entrée
```