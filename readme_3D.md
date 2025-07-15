## Source

[Image to 3D to ComfyUI: The Easy Way!](https://www.youtube.com/watch?v=PIiHqOdlzYA)

[3D or Live View ComfyUI Workflow](https://civitai.com/models/1156226/3d-or-live-view-comfyui-workflow)

[Image to 3D Asset with Trellis](https://trellis3d.net/models/trellis3d)

[stability ai](https://stability.ai/stable-3d)

[stability ai stable-point-aware-3d](https://huggingface.co/stabilityai/stable-point-aware-3d)

## Test

[Run Hunyuan3D in the Cloud - No Installation Required](https://www.thinkdiffusion.com/hunyuan3D)

[github source](https://github.com/Tencent-Hunyuan/Hunyuan3D-1)

ComfyUI Manager V3.31.8 -> V3.34

Installed missing nodes

Hy3DModelLoader
Hy3DVAEDecode
Hy3DDiffusersSchedulerConfig
Hy3DDelightImage
Hy3DMeshUVWrap
Hy3DCameraConfig
Hy3DSampleMultiView
DownloadAndLoadHy3DPaintModel
Hy3DPostprocessMesh
Hy3DRenderMultiView
DownloadAndLoadHy3DDelightModel
Hy3DApplyTexture
Hy3DMeshVerticeInpaintTexture
CV2InpaintTexture
Hy3DBakeFromMultiview
Hy3DGenerateMesh
Hy3DExportMesh


Prompt outputs failed validation:
Hy3DModelLoader:
- Value not in list: model: 'hunyuan3d-dit-v2-0-fp16.safetensors' not in ['Wan2.1-Fun-Control-14B_fp8_e4m3fn.safetensors', 'Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors', 'Wan2_1-I2V-14B-720P_fp8_e4m3fn.safetensors', 'Wan_2.1_Fun_Control_1.3B.safetensors', 'flux1-canny-dev.safetensors', 'flux1-depth-dev.safetensors', 'flux1-dev-fp8.safetensors', 'flux1-dev.safetensors', 'flux1-fill-dev.safetensors', 'flux1-schnell-fp8.safetensors', 'flux1CannyDevFp8_v10.safetensors', 'flux1DepthDevFp8_v10.safetensors', 'hunyuan_video_I2V_fp8_e4m3fn.safetensors', 'hunyuan_video_v2_replace_image_to_video_720p_bf16.safetensors', 'wan2.1_i2v_720p_14B_fp16.safetensors', 'wan2.1_i2v_720p_14B_fp8_scaled.safetensors', 'wan2.1_t2v_1.3B_fp16.safetensors']


[hunyuan3d-dit-v2-0-fp16.safetensors](https://huggingface.co/Kijai/Hunyuan3D-2_safetensors/blob/main/hunyuan3d-dit-v2-0-fp16.safetensors)

Copy in diffusion_models

[Flow control topic](https://www.reddit.com/r/comfyui/comments/1icdbvx/comment/m9soga7/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button)