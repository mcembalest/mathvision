import modal
from pathlib import Path

GPU_CONFIG = f"l40s:1"
MINUTES_TIMEOUT = 20
MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
MODEL_TOKENIZER_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_CHAT_TEMPLATE = "qwen2-vl"
MODEL_VOL_PATH = Path("/models")
MODEL_VOL = modal.Volume.from_name("sgl-cache", create_if_missing=True)
volumes = {MODEL_VOL_PATH: MODEL_VOL}


def download_model():
    from huggingface_hub import snapshot_download

    snapshot_download(
        MODEL_PATH,
        local_dir=str(MODEL_VOL_PATH / MODEL_PATH),
        revision=MODEL_REVISION,
        ignore_patterns=["*.pt", "*.bin"],
    )

cuda_version = "12.8.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

vlm_image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .entrypoint([])
    .apt_install("libnuma-dev")  # NUMA library for sgl_kernel
    .uv_pip_install(
        "transformers==4.54.1",
        "numpy<2",
        "fastapi[standard]==0.115.4",
        "pydantic==2.9.2",
        "requests==2.32.3",
        "starlette==0.41.2",
        "torch==2.7.1",
        "sglang[all]==0.4.10.post2",
        "sgl-kernel==0.2.8",
        "hf-xet==1.1.5",
        pre=True,
    )
    .env(
        {
            "HF_HOME": str(MODEL_VOL_PATH),
            "HF_XET_HIGH_PERFORMANCE": "1",
        }
    )
    .run_function(
        download_model, volumes=volumes
    )
)

app = modal.App("sgl-vlm")

@app.cls(
    gpu=GPU_CONFIG,
    timeout=MINUTES_TIMEOUT * 60,
    scaledown_window=MINUTES_TIMEOUT * 60,
    image=vlm_image,
    volumes=volumes,
)
@modal.concurrent(max_inputs=100)
class Model:
    @modal.enter()
    def start_runtime(self):
        import sglang as sgl

        self.runtime = sgl.Runtime(model_path=MODEL_PATH,tokenizer_path=MODEL_TOKENIZER_PATH)
        self.runtime.endpoint.chat_template = sgl.lang.chat_template.get_chat_template(
            MODEL_CHAT_TEMPLATE
        )
        sgl.set_default_backend(self.runtime)

    @modal.fastapi_endpoint(method="POST", docs=True)
    def generate(self, request: dict) -> str:
        from pathlib import Path
        import requests
        import sglang as sgl

        image_url = request.get("image_url")
        image_response = requests.get(image_url)
        image_response.raise_for_status()
        image_filename = image_url.split("/")[-1]
        image_path = Path(f"/tmp/{image_filename}")
        image_path.write_bytes(image_response.content)

        @sgl.function
        def forward(s, image_path, text):
            s += sgl.user(text + sgl.image(str(image_path)))
            s += sgl.assistant(sgl.gen("response"))

        text = request.get("text")
        return forward.run(
            image_path=image_path, 
            text=text, 
            max_new_tokens=request.get("max_new_tokens", 1024)
        )["response"]

    @modal.exit()
    def shutdown_runtime(self):
        self.runtime.shutdown()
