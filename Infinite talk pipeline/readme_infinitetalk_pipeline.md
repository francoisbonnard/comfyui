# extraction audio avec stacher

Emmanuel Macron Discours
https://www.youtube.com/watch?v=Dm27NwstXcY


# découpage avec Adobe Audition

![informations de durée](./img/image.png)

# Google Colab Training

## Rappel des processeurs

Processeur (CPU) : calcul général. OK pour pré-traitement, petites tâches, pas pour l’entraînement GPU/TPU.

GPU T4 (NVIDIA, “Turing”) : entrée de gamme. 16 Go de VRAM typiquement. Bien pour inference/finetune légers (petits LLM, diffusion en 512²).

GPU L4 (NVIDIA, “Ada” datacenter) : milieu de gamme récent, orienté inférence et multimédia (NVENC/NVDEC costauds). Plus rapide que T4, VRAM typiquement 24 Go.

GPU A100 (NVIDIA, “Ampere”) : haut de gamme entraînement. Très rapide en FP16/BF16/TF32, beaucoup de VRAM (souvent 40 Go sur Colab). Idéal pour gros batchs, gros modèles.

TPU v5e-1 (Google) : bon rapport perf/€ pour JAX/TF. Très efficace en BF16, mais PyTorch nécessite XLA (moins plug-and-play). “-1” = plus petit slice.

TPU v6e-1 (Google) : génération plus récente que v5e, meilleure perf/efficacité sur JAX/TF. Même remarques côté PyTorch/XLA.

##  Ultimate_rvc_colab.ipynb

![Interface principale](./img/image-1.png)
![step1](./img/image-2.png)
![step2](./img/image-3.png)
![step3](./img/image-4.png)
![algorithmic](./img/image-5.png)
![datastorage](./img/image-6.png)
![Device and memory](./img/image-7.png)

### reco chatGPT5

Données : vise ≥ 10–15 min (mieux : 30–60 min) de voix propre, sans musique/bruit.

Avec 20–60 min de données : ~200–400 epochs suffisent généralement (on arrête quand la val loss stagne et que l’overfit apparaît).

Avec 3,7 min (ton cas) : tu peux pousser à 600–1 200 epochs pour “faire sortir un timbre”, mais tu auras vite de l’overfit (robotisation/artefacts) et la qualité restera limitée. Mieux vaut augmenter le dataset que d’augmenter encore les epochs.

### Feature Extraction

F0 Method : 
- mvpe
- crepe
- crepe-tiny

Embedder model
- contentvec
- chinese
- japanese
- corean
- custom

