# Dossier samples

Placez ici vos fichiers audio d'entrée.

## Formats supportés

WAV, MP3, FLAC, OGG, M4A — tout format décodable par `librosa`.

## Exemple d'utilisation

```bash
python main.py samples/ma_question.wav --output reponse.wav --play
```

## Conseils

- Durée recommandée : 5 à 30 secondes pour de meilleurs résultats.
- Le modèle Whisper détecte automatiquement la langue parlée.
- Le LLM et le TTS fonctionnent principalement en anglais ;
  si l'audio est en français, Whisper le transcrit correctement mais
  TinyLlama et Orpheus répondront en anglais.
