
    python .\crepe.py .\bollore30minutes.wav -o  chunks_rvc --chunk_len 10 --overlap 0.5  


Voici un script Python prêt à l’emploi pour découper ton .wav (mono, 16-bits, ~30 min) en segments (« chunks ») adaptés à l’entraînement d’un modèle RVC avec F0=**crepe** dans le repo *ultimate-rvc*. Il :

* vérifie mono & PCM-16,
* normalise en crête (–1 dBFS),
* découpe en morceaux de durée fixe,
* aligne (autant que possible) les coupures sur un **passage par zéro** pour éviter les clics,
* applique un léger **fade-in/out** (5 ms),
* exporte en **PCM 16-bits**.

### Recommandation taille des chunks (pour RVC + crepe)

* **10 s** est un très bon par défaut (parler/chant, stable pour F0 crepe).
* Si tu as **peu de données**, tu peux **chevaucher 0,5 s** (overlap=0.5) pour augmenter le nombre d’exemples.
* Évite >30 s (moins de diversité par batch) et <3–5 s (contexte F0 trop court).
* Donc : **chunk_len=10, overlap=0.0–0.5 s** est le sweet spot.

---

### Script `split_wav_for_rvc.py`

```python
import argparse
import os
from pathlib import Path

import numpy as np
import soundfile as sf

def peak_normalize(x, target_dbfs=-1.0, eps=1e-9):
    peak = np.max(np.abs(x)) + eps
    target_lin = 10 ** (target_dbfs / 20.0)
    gain = target_lin / peak
    return np.clip(x * gain, -1.0, 1.0)

def nearest_zero_crossing(x, center_idx, search_radius):
    """Trouve un index proche avec changement de signe (passage par zéro)."""
    start = max(1, center_idx - search_radius)
    end = min(len(x), center_idx + search_radius)
    segment = x[start:end]
    # indices où il y a changement de signe
    zero_cross = np.where(np.signbit(segment[:-1]) != np.signbit(segment[1:]))[0]
    if zero_cross.size == 0:
        return center_idx
    # convertir en indices absolus et choisir le plus proche de center_idx
    candidates = start + zero_cross
    return int(candidates[np.argmin(np.abs(candidates - center_idx))])

def apply_fade(x, sr, fade_ms=5.0):
    fade_samples = max(1, int(sr * (fade_ms / 1000.0)))
    if fade_samples * 2 >= len(x):
        return x  # trop court, skip
    # fade-in
    ramp_in = np.linspace(0.0, 1.0, fade_samples, endpoint=True)
    x[:fade_samples] *= ramp_in
    # fade-out
    ramp_out = np.linspace(1.0, 0.0, fade_samples, endpoint=True)
    x[-fade_samples:] *= ramp_out
    return x

def seconds_to_stamp(t):
    m, s = divmod(int(t), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}h{m:02d}m{s:02d}s"
    return f"{m:02d}m{s:02d}s"

def main():
    parser = argparse.ArgumentParser(
        description="Découpe un WAV mono PCM16 en chunks pour RVC (crepe)."
    )
    parser.add_argument("input_wav", type=str, help="Chemin du .wav source (mono, PCM16)")
    parser.add_argument("-o", "--outdir", type=str, default="chunks",
                        help="Dossier de sortie (défaut: ./chunks)")
    parser.add_argument("--chunk_len", type=float, default=10.0,
                        help="Durée d'un chunk en secondes (défaut: 10.0)")
    parser.add_argument("--overlap", type=float, default=0.0,
                        help="Chevauchement entre chunks en secondes (défaut: 0.0)")
    parser.add_argument("--min_keep", type=float, default=3.0,
                        help="Durée minimale pour garder le dernier chunk (défaut: 3.0s)")
    parser.add_argument("--zero_search_ms", type=float, default=8.0,
                        help="Fenêtre de recherche autour de la coupure pour aligner un passage par zéro (ms)")
    parser.add_argument("--fade_ms", type=float, default=5.0,
                        help="Fade-in/out par chunk (ms)")
    args = parser.parse_args()

    in_path = Path(args.input_wav)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Lecture
    data, sr = sf.read(in_path, always_2d=True)  # shape: (N, C)
    if data.shape[1] != 1:
        raise ValueError("Le fichier doit être mono (1 canal).")
    data = data[:, 0].astype(np.float32)

    # Vérification format fichier
    info = sf.info(in_path)
    if "PCM_16" not in str(info.subtype):
        raise ValueError(f"Le fichier doit être PCM 16 bits, trouvé: {info.subtype}")

    # Normalisation crête à -1 dBFS (sécurise les fades et re-écriture en 16 bits)
    data = peak_normalize(data, target_dbfs=-1.0)

    chunk_len_samples = int(round(args.chunk_len * sr))
    overlap_samples = int(round(args.overlap * sr))
    if overlap_samples >= chunk_len_samples:
        raise ValueError("overlap doit être < chunk_len.")
    step = chunk_len_samples - overlap_samples
    min_keep_samples = int(round(args.min_keep * sr))
    zero_search_radius = int(round((args.zero_search_ms / 1000.0) * sr))

    total = len(data)
    start = 0
    idx = 0
    base = in_path.stem

    while start < total:
        end = start + chunk_len_samples
        if end >= total:
            # Dernier tronçon
            end = total
            if (end - start) < min_keep_samples:
                break  # trop court, on jette

        # Aligner la frontière gauche (pas pour idx==0)
        if idx > 0:
            start = nearest_zero_crossing(data, start, zero_search_radius)
        # Aligner la frontière droite (sauf si dernier)
        if end < total:
            end = nearest_zero_crossing(data, end, zero_search_radius)

        chunk = np.copy(data[start:end])

        # Fade-in/out léger
        chunk = apply_fade(chunk, sr, fade_ms=args.fade_ms)

        # Écriture en PCM16 (soundfile gère l'encodage depuis float32)
        start_sec = start / sr
        end_sec = end / sr
        name = f"{base}_{seconds_to_stamp(start_sec)}_{seconds_to_stamp(end_sec)}.wav"
        out_path = outdir / name
        sf.write(out_path.as_posix(), chunk, sr, subtype="PCM_16")

        idx += 1
        start = start + step  # avancer selon le step

    print(f"Exporté {idx} chunks dans: {outdir.resolve()}")

if __name__ == "__main__":
    main()
```

#### Utilisation

```bash
# Par défaut: 10 s sans overlap
python split_wav_for_rvc.py "C:\chemin\vers\ton_fichier.wav" -o chunks_rvc --chunk_len 10 --overlap 0

# Avec 0,5 s d’overlap pour densifier l’entraînement
python split_wav_for_rvc.py "C:\chemin\vers\ton_fichier.wav" -o chunks_rvc --chunk_len 10 --overlap 0.5
```

#### Où mettre ces fichiers pour *ultimate-rvc*

* Place simplement les .wav générés dans le dossier **dataset**/sous-dossier que tu utilises pour l’entraînement (selon ta config du repo).
* Garde le **sample rate d’origine** (44.1 kHz ou 48 kHz). Le script n’upsample/downsample pas ; si tu veux forcer 48 kHz, je peux te donner une version avec resampling.

Si tu veux une variante qui **retire les silences** avant la découpe, ou qui **resample** automatiquement, dis-le et je te mets une version adaptée.
