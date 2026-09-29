#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import io
import math
import os
import time
import uuid
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple
from urllib.request import urlopen

import torch
import torchvision.transforms as T
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MODEL_LAYER_COUNTS = {
    "InternVL2_5-1B": 24,
    "InternVL2_5-2B": 24,
    "InternVL2_5-4B": 36,
    "InternVL2_5-8B": 32,
    "InternVL2_5-26B": 48,
    "InternVL2_5-38B": 64,
    "InternVL2_5-78B": 80,
}


class ChatMessage(BaseModel):
    role: str
    content: Any


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    max_tokens: Optional[int] = Field(default=2048, alias="max_tokens")
    stream: Optional[bool] = False


def str_to_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def build_transform(input_size: int):
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios,
    width: int,
    height: int,
    image_size: int,
):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = True,
) -> List[Image.Image]:
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def image_from_url(url: str) -> Image.Image:
    if url.startswith("data:"):
        _, encoded = url.split(",", 1)
        payload = base64.b64decode(encoded)
        return Image.open(io.BytesIO(payload)).convert("RGB")
    if url.startswith("http://") or url.startswith("https://"):
        with urlopen(url, timeout=30) as response:
            payload = response.read()
        return Image.open(io.BytesIO(payload)).convert("RGB")
    if url.startswith("file://"):
        return Image.open(url[len("file://") :]).convert("RGB")
    return Image.open(url).convert("RGB")


def load_image_from_url(url: str, input_size: int, max_num: int) -> torch.Tensor:
    image = image_from_url(url)
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, max_num=max_num, use_thumbnail=True)
    pixel_values = [transform(tile) for tile in images]
    return torch.stack(pixel_values)


def split_model(model_name: str) -> Dict[str, int]:
    if model_name not in MODEL_LAYER_COUNTS:
        raise ValueError(f"unsupported model name for split device map: {model_name}")
    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("CUDA device is required for InternVL2.5-38B inference.")

    num_layers = MODEL_LAYER_COUNTS[model_name]
    device_map: Dict[str, int] = {}
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)

    layer_cnt = 0
    for gpu_idx, layer_num in enumerate(num_layers_per_gpu):
        for _ in range(layer_num):
            if layer_cnt >= num_layers:
                break
            device_map[f"language_model.model.layers.{layer_cnt}"] = gpu_idx
            layer_cnt += 1

    device_map["vision_model"] = 0
    device_map["mlp1"] = 0
    device_map["language_model.model.tok_embeddings"] = 0
    device_map["language_model.model.embed_tokens"] = 0
    device_map["language_model.output"] = 0
    device_map["language_model.model.norm"] = 0
    device_map["language_model.model.rotary_emb"] = 0
    device_map["language_model.lm_head"] = 0
    device_map[f"language_model.model.layers.{num_layers - 1}"] = 0
    return device_map


def content_text_and_images(content: Any) -> Tuple[str, List[str]]:
    if isinstance(content, str):
        return content, []
    if not isinstance(content, list):
        return str(content), []

    text_parts: List[str] = []
    image_urls: List[str] = []
    for item in content:
        if not isinstance(item, dict):
            text_parts.append(str(item))
            continue
        item_type = item.get("type")
        if item_type == "text":
            text_parts.append(str(item.get("text", "")))
        elif item_type == "image_url":
            image_url = item.get("image_url") or {}
            if isinstance(image_url, dict):
                image_urls.append(str(image_url.get("url", "")))
            else:
                image_urls.append(str(image_url))
        elif item_type == "image":
            image_url = item.get("url") or item.get("path") or item.get("image")
            if image_url:
                image_urls.append(str(image_url))
    return "\n".join(part for part in text_parts if part).strip(), [url for url in image_urls if url]


def build_question(messages: List[ChatMessage], image_count: int) -> str:
    system_parts: List[str] = []
    user_parts: List[str] = []
    assistant_parts: List[str] = []
    for message in messages:
        text, _ = content_text_and_images(message.content)
        if not text:
            continue
        if message.role == "system":
            system_parts.append(text)
        elif message.role == "assistant":
            assistant_parts.append(text)
        else:
            user_parts.append(text)

    text_parts: List[str] = []
    if system_parts:
        text_parts.append("\n".join(system_parts).strip())
    if assistant_parts:
        text_parts.append("\n".join(f"Previous assistant: {part}" for part in assistant_parts).strip())
    if user_parts:
        text_parts.append("\n".join(user_parts).strip())
    question_text = "\n\n".join(part for part in text_parts if part).strip()

    if image_count == 0:
        return question_text
    if image_count == 1:
        return f"<image>\n{question_text}".strip()
    image_prefix = "".join(f"Image-{idx + 1}: <image>\n" for idx in range(image_count))
    return f"{image_prefix}{question_text}".strip()


def collect_image_urls(messages: List[ChatMessage]) -> List[str]:
    image_urls: List[str] = []
    for message in messages:
        _, urls = content_text_and_images(message.content)
        image_urls.extend(urls)
    return image_urls


def parse_args():
    parser = argparse.ArgumentParser(description="OpenAI-compatible InternVL2.5-38B server using transformers.")
    parser.add_argument("--model-path", default=os.environ.get("MODEL_PATH", "OpenGVLab/InternVL2_5-38B"))
    parser.add_argument("--served-model-name", default=os.environ.get("SERVED_MODEL_NAME", "internvl25-38b"))
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    parser.add_argument("--model-name-for-split", default=os.environ.get("MODEL_NAME_FOR_SPLIT", "InternVL2_5-38B"))
    parser.add_argument("--max-image-tiles", type=int, default=int(os.environ.get("MAX_IMAGE_TILES", "12")))
    parser.add_argument("--image-size", type=int, default=int(os.environ.get("IMAGE_SIZE", "448")))
    parser.add_argument("--default-max-new-tokens", type=int, default=int(os.environ.get("POLICY_MAX_TOKENS", "2048")))
    parser.add_argument("--use-flash-attn", action="store_true", default=str_to_bool(os.environ.get("USE_FLASH_ATTN", "1")))
    parser.add_argument("--load-in-8bit", action="store_true", default=str_to_bool(os.environ.get("LOAD_IN_8BIT", "0")))
    parser.add_argument("--device-map", choices=["split", "auto"], default=os.environ.get("DEVICE_MAP", "split"))
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default=os.environ.get("DTYPE", "bf16"))
    return parser.parse_args()


def create_app(args) -> FastAPI:
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    device_map: Any = split_model(args.model_name_for_split) if args.device_map == "split" else "auto"
    model_kwargs: Dict[str, Any] = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "use_flash_attn": args.use_flash_attn,
        "trust_remote_code": True,
        "device_map": device_map,
    }
    if args.load_in_8bit:
        model_kwargs["load_in_8bit"] = True

    print(f"Loading {args.model_path} with transformers", flush=True)
    print(f"served_model_name={args.served_model_name}", flush=True)
    print(f"device_map={args.device_map} cuda_devices={torch.cuda.device_count()}", flush=True)
    print(f"dtype={args.dtype} load_in_8bit={args.load_in_8bit} use_flash_attn={args.use_flash_attn}", flush=True)

    model = AutoModel.from_pretrained(args.model_path, **model_kwargs).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
    infer_lock = Lock()

    app = FastAPI(title="InternVL2.5-38B Transformers OpenAI-Compatible Server")

    @app.get("/v1/models")
    def list_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": args.served_model_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "local",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatCompletionRequest):
        if request.stream:
            raise HTTPException(status_code=400, detail="stream=true is not supported by this lightweight server.")

        image_urls = collect_image_urls(request.messages)
        question = build_question(request.messages, len(image_urls))
        if not question:
            raise HTTPException(status_code=400, detail="empty prompt")

        pixel_values = None
        num_patches_list = None
        if image_urls:
            pixels = [
                load_image_from_url(url, input_size=args.image_size, max_num=args.max_image_tiles)
                for url in image_urls
            ]
            num_patches_list = [pixel.size(0) for pixel in pixels]
            pixel_values = torch.cat(pixels, dim=0).to(dtype).cuda()

        max_new_tokens = request.max_tokens or args.default_max_new_tokens
        temperature = 0.0 if request.temperature is None else float(request.temperature)
        generation_config = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": temperature > 0,
            "temperature": max(temperature, 1e-6),
            "top_p": 1.0 if request.top_p is None else float(request.top_p),
        }

        try:
            with infer_lock:
                with torch.inference_mode():
                    if pixel_values is not None and len(image_urls) > 1:
                        response = model.chat(
                            tokenizer,
                            pixel_values,
                            question,
                            generation_config,
                            num_patches_list=num_patches_list,
                            history=None,
                            return_history=False,
                        )
                    else:
                        response = model.chat(
                            tokenizer,
                            pixel_values,
                            question,
                            generation_config,
                            history=None,
                            return_history=False,
                        )
            if isinstance(response, tuple):
                response = response[0]
        except Exception as exc:
            raise HTTPException(status_code=500, detail=repr(exc)) from exc

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": str(response)},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }

    return app


def main():
    args = parse_args()
    app = create_app(args)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
