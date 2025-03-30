- [Ep07 - Working With Text - Art Styles Update](#ep07---working-with-text---art-styles-update)
- [Ep09 - How to Use SDXL ControlNet Union](#ep09---how-to-use-sdxl-controlnet-union)
- [Ep10 - Flux GGUF and Custom Nodes](#ep10---flux-gguf-and-custom-nodes)
- [ThinkDiffusion Competitors](#thinkdiffusion-competitors)
- [Ep11 - LLM, Prompt Generation, img2txt, txt2txt Overview](#ep11---llm-prompt-generation-img2txt-txt2txt-overview)
  - [Florence](#florence)
  - [Searge](#searge)
- [Ep12 - How to Upscale Your AI Images](#ep12---how-to-upscale-your-ai-images)
- [Ep13 - Exploring Ollama, LLaVA, Gemma Models](#ep13---exploring-ollama-llava-gemma-models)
- [Ep14 - How to Use Flux ControlNet Union Pro](#ep14---how-to-use-flux-controlnet-union-pro)
  - [New model](#new-model)
  - [Custom nodes](#custom-nodes)
  - [Allocation on device](#allocation-on-device)
  - [Not working with BAE](#not-working-with-bae)
- [Ep15 - Styles Update, Prompts](#ep15---styles-update-prompts)
  - [Custom nodes](#custom-nodes-1)
- [Ep17 Flux LoRA ](#ep17-flux-lora-)
  - [Some Lora](#some-lora)
  - [from civitai](#from-civitai)
  - [Custom Nodes](#custom-nodes-2)
- [Flux LoRA Training with Kohya in 2025](#flux-lora-training-with-kohya-in-2025)
  - [Version Kohya v24.1.7 (sep 6,2024)](#version-kohya-v2417-sep-62024)
  - [Uploading images](#uploading-images)
  - [Blip Captioning](#blip-captioning)
  - [Dataset preparation](#dataset-preparation)


## Ep07 - Working With Text - Art Styles Update

For custom node "was-node-suite-comfyui" I am trying to setup "webui_styles" parameter in the "was_suite_config.json" file to fit my "mystyles.csv", but this doesn't work : "webui_styles": "/home/ubuntu/ComfyUI/custom_nodes/was-node-suite-comfyui/mystyles.csv"



../user_data/comfyui/custom_nodes/was-node-suite-comfyui/was_suite_config.json

    {
        "run_requirements": true,
        "suppress_uncomfy_warnings": true,  
        "show_startup_junk": true,
        "show_inspiration_quote": true,
        "text_nodes_type": "STRING",
        "webui_styles": "/home/ubuntu/ComfyUI/models/styles/styles_-_styles.csv",
        "webui_styles_persistent_update": true,
        "sam_model_vith_url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "sam_model_vitl_url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "sam_model_vitb_url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "history_display_limit": 36,
        "use_legacy_ascii_text": false,
        "ffmpeg_bin_path": "/path/to/ffmpeg",
        "ffmpeg_extra_codecs": {
        "avc1": ".mp4",
        "h264": ".mkv"
        },
        "wildcards_path": "/home/ubuntu/ComfyUI/custom_nodes/was-node-suite-comfyui/wildcards",
        "wildcard_api": true
    }


  replace with 

      "webui_styles": "/home/ubuntu/ComfyUI/models/styles/styles_-_styles.csv",

## [Ep09 - How to Use SDXL ControlNet Union](https://www.youtube.com/watch?v=C0zykaDF1ts)

https://huggingface.co/xinsir/controlnet-union-sdxl-1.0/tree/main

../user_data/comfyui/models/controlnet/diffusion_pytorch_model_promax.safetensors

Install custom nodes :
- comfyui-art-venture
- ComfyUI's ControlNet Auxiliary Preprocessors
- Comfyroll Studio (to stack multiple controlnet)


## [Ep10 - Flux GGUF and Custom Nodes](https://www.youtube.com/watch?v=Ym0oJpRbj4U)

Install custom nodes :
- comfyui-gguf
- crystools
- rgthree's ComfyUI Nodes

https://huggingface.co/city96/FLUX.1-dev-gguf/tree/main
https://huggingface.co/city96/FLUX.1-dev-gguf/blob/main/flux1-dev-Q8_0.gguf

copy in : ../user_data/comfyui/models/unet

Get clip_l.safetensors from here
https://huggingface.co/comfyanonymous/flux_text_encoders/tree/main
Place it in the clip folder ..ComfyUI\models\clip

Get one of the T5 encoders (recommended to use Q5_K_M or larger for the best results)
https://huggingface.co/city96/t5-v1_1-xxl-encoder-gguf/tree/main
Place it in the clip folder ..ComfyUI\models\clip

Get the ae.safetensor vae from here
https://huggingface.co/black-forest-labs/FLUX.1-schnell/blob/main/ae.safetensors
Place it in the vae folder ..ComfyUI\models\vae

## ThinkDiffusion Competitors 

https://comfy.icu/

[custome Nodes](https://comfy.icu/node/)

## [Ep11 - LLM, Prompt Generation, img2txt, txt2txt Overview](https://www.youtube.com/watch?v=yutYU97Bj7E)

### Florence 
Install Florence
![Select version](image-2.png)

I use the latest

DownloadAndLoadFlorence2Model -> automatic in ../user_data/comfyui/models/LLM/Florence-2-base

### Searge

Instal custom nodes : Searge-LLM for ComfyUI v1.0

create directory : models/llm_gguf

place Mistral-7B-Instruct-v0.3.Q4_K_M.gguf in the ComfyUI/models/llm_gguf directory.

https://huggingface.co/MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF/tree/main
Recommended : https://huggingface.co/MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf

## [Ep12 - How to Upscale Your AI Images](https://www.youtube.com/watch?v=i8v9RbNy4Zw)

Go to manger, model manager / sort by type Upscale : 
- 4x_NMKD-Siax_200k
- 4x-AnimeSharp
- 4x_foolhardy_Remacri

Refresh ComfyUI

Install this custom nodes :
- ControlAltAI Nodes
- ComfyUI-PixelResolutionCalculator
- ComfyUI Easy Use
- rgthree's ComfyUI Nodes

Restart ComfyUI

## [Ep13 - Exploring Ollama, LLaVA, Gemma Models](https://www.youtube.com/watch?v=eK6MXm7q37c)

Link pixorama workflows : https://discord.com/channels/1245221993746399232/1323521185132183694


[Ollama search](https://ollama.com/search)

![llava](image-5.png)

Install these custom nodes :

- ComfyUI Ollama created by stavsap
- ComfyUI Easy Use

## [Ep14 - How to Use Flux ControlNet Union Pro](https://www.youtube.com/watch?v=WHuhxKk40k4)

https://discord.com/channels/1245221993746399232/1323578980581904488

### New model

[model diffusion_pytorch_model.safetensors](https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro/tree/main)


How to change name inside of models inside controlnet ? 

![2 controlnet model](image-6.png)

### Custom nodes


ComfyUI's ControlNet Auxiliary Preprocessors


![personal Workspace files](image-7.png)

conflict with : 

![Comyui-art-venture in restart status](image-8.png)


Uninstall ComfyUI's ControlNet Auxiliary Preprocessors

ThinkDiff Team Advice

![Advice from Clem](image-10.png)

https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro
https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro/resolve/main/diffusion_pytorch_model.safetensors

### Allocation on device

![allocation](image-9.png)

### Not working with BAE

## [Ep15 - Styles Update, Prompts](https://www.youtube.com/watch?v=KMlUakdbdnc)
https://discord.com/channels/1245221993746399232/1323585163896033380

### Custom nodes

ComfyUI-iTools

 ## [Ep17 Flux LoRA ](https://www.youtube.com/watch?v=-aW1U8QEak0&t=234s)

### Some Lora

Download flux-ghibsky-illustration lora
https://huggingface.co/aleksa-codes/flux-ghibsky-illustration/tree/main
Place it in the loras folder ..ComfyUI\models\loras
Trigger Word: GHIBSKY

Download 70s SciFi Style
https://civitai.com/models/824478/70s-scifi-style-by-chronoknight-flux
Place it in the loras folder ..ComfyUI\models\loras
Trigger Word: ck-70scf

Download Flux Fantasy Lora
https://www.shakker.ai/modelinfo/3cf25bb29e0144e4849064b122150054/Flux-Fantasy-Hide?from=models
Place it in the loras folder ..ComfyUI\models\loras
Trigger Word: fantasy
Recommended weight 0.8

Download Sketchy Pastel Anime Flux Lora
https://www.shakker.ai/modelinfo/33815c53e3024899bde957fa012e1f43/TQ-Sketchy-Pastel-Anime-Flux?from=models
Place it in the loras folder ..ComfyUI\models\loras
Trigger Word: anime
Recommended weight 0.8

Download Flux_Sticker_Lora
https://huggingface.co/diabolic6045/Flux_Sticker_Lora
Place it in the loras folder ..ComfyUI\models\loras
Trigger Word: 5t1cker 5ty1e

Download UltraRealistic Lora
https://civitai.com/models/796382?modelVersionId=940466
Place it in the loras folder ..ComfyUI\models\loras
Trigger Words: amateurish photo
Guidance =2.5 Steps=40

Download diffusion_pytorch_model.safetensors
https://huggingface.co/alimama-creative/FLUX.1-Turbo-Alpha/tree/main
rename it to FLUX.1-Turbo-Alpha or any other name you want
Place it in the loras folder ..ComfyUI\models\loras
guidance_scale=3.5 and lora_scale=1 steps=8

### from civitai
Trigger Words: made out of clouds
Guidance =2.5 Steps=40

STRENGTH: 1
https://civitai.com/models/749668/flux-cloudstyle


### Custom Nodes

![NO NEED OF ComfyUI-Manager](image-11.png)

## [Flux LoRA Training with Kohya in 2025](https://learn.thinkdiffusion.com/flux-lora-training-with-kohya/)

### Version Kohya v24.1.7 (sep 6,2024)
![Relesea to use](image-12.png)

### Uploading images

/home/ubuntu/user_data/kohya/image/Trump

![Jpb & txt files](image-13.png)

### Blip Captioning

(done for trump)

### Dataset preparation

Go to Lora / Training / Dataset preparation

![Lora - Training](image-14.png)

![below](image-15.png)

1. Dreambooth/LoRA Folder preparation enter :
   - the Instance prompt : trumpuni
   - Class prompt : 
2. Set required paths
   - Training Images: /home/ubuntu/user_data/kohya/image/trump
   - Destination directory : /home/ubuntu/user_data/kohya/output/trump
   - ![output folder](image-16.png)
3. Set the number of repeats : 20   
4. Click on Prepare training data 
5. Click on Copy info to respective fields.
    - new folder in output/img ![new folder in output/img](image-17.png)
6. Upload config file
    - ![alt text](image-18.png)
    - /home/ubuntu/user_data/kohya/configs/kohya48gbvram.json
7. Configuration tab
   - ![alt text](image-19.png)
   - ![1 & 2](image-20.png)
8. Start training
   - ![alt text](image-21.png)
9. Check progress
    - /home/ubuntu/user_data/kohya/logs/francoisbonnard-clipskip-30090550-logs.txt
10. Result
    -  .safetensors files
11. TensorBoard
    - Start tensorboard ![alt text](image-22.png)
    - Open tensorboard ![alt text](image-23.png)
    - 11h39 ![alt text](image-24.png)
    - ![alt text](image-27.png)