"""Process referenced Markdown images with multimodal summaries and MinIO URLs."""

import os
import re
import sys
import base64
from pathlib import Path
from typing import Dict, List, Tuple
from collections import deque

from minio import Minio
from minio.deleteobjects import DeleteObject

from app.clients.minio_utils import get_minio_client
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_running_task
from app.lm.lm_utils import get_llm_client
from langchain.messages import HumanMessage
from langchain_core.exceptions import LangChainException
from app.conf.minio_config import minio_config
from app.conf.lm_config import lm_config
from app.core.logger import logger
from app.utils.rate_limit_utils import apply_api_rate_limit
from app.core.load_prompt import load_prompt

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}


def step_1_get_content(state: ImportGraphState) -> Tuple[str, Path, Path]:
    """Load Markdown content and resolve its companion image directory."""
    md_file_path = state["md_path"]
    if not md_file_path:
        raise FileNotFoundError("md_path is required")

    path_obj = Path(md_file_path)
    if not state["md_content"]:
        with open(path_obj, "r", encoding="utf-8") as f:
            md_content = f.read()
        logger.debug(f"Markdown loaded from file: length={len(md_content)}")
    else:
        md_content = state["md_content"]
        logger.debug(f"Markdown loaded from state: length={len(md_content)}")

    images_dir = path_obj.parent / "images"
    return md_content, path_obj, images_dir


def is_supported_image(filename: str) -> bool:
    """Return whether the filename has a supported image extension."""
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS

def find_image_in_md(md_content: str, image_filename: str, context_len: int = 100) -> List[Tuple[str, str]]:
    """Return surrounding text for each Markdown reference to an image."""
    pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_filename) + r".*?\)")
    results = []

    for m in pattern.finditer(md_content):
        start, end = m.span()
        pre_text = md_content[max(0, start - context_len):start]
        post_text = md_content[end:min(len(md_content), end + context_len)]
        logger.debug(f"Markdown image reference found: filename={image_filename}")
        results.append((pre_text, post_text))
    if not results:
        logger.debug(f"No Markdown reference found: filename={image_filename}")
    return results

def step_2_scan_images(md_content: str, images_dir: Path) -> List[Tuple[str, str, Tuple[str, str]]]:
    """Collect supported images that are referenced by the Markdown document."""
    targets = []
    for image_file in os.listdir(images_dir):
        if not is_supported_image(image_file):
            logger.debug(f"Unsupported image extension: filename={image_file}")
            continue
        img_path = str(images_dir / image_file)
        context_list = find_image_in_md(md_content, image_file)
        if not context_list:
            logger.warning(f"Image is not referenced by Markdown: filename={image_file}")
            continue
        targets.append((image_file, img_path, context_list[0]))
        logger.info(f"Image queued for processing: filename={image_file}")
    logger.info(f"Image scan completed: targets={len(targets)}")
    return targets


def encode_image_to_base64(image_path: str) -> str:
    """Encode a local image as a Base64 string for multimodal inference."""
    with open(image_path, "rb") as img_file:
        base64_str = base64.b64encode(img_file.read()).decode("utf-8")
    logger.debug(f"Image encoded: path={image_path}, length={len(base64_str)}")
    return base64_str

def summarize_image(image_path: str, root_folder: str, image_content: Tuple[str, str]) -> str:
    """Generate a concise image summary using the configured vision model."""
    base64_image = encode_image_to_base64(image_path)
    try:
        lvm_client = get_llm_client(model=lm_config.lv_model)

        prompt_text = load_prompt(
            name="image_summary",
            root_folder=root_folder,
            image_content=image_content,
        )

        messages = [
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": prompt_text
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            )
        ]

        response = lvm_client.invoke(messages)

        summary = response.content.strip().replace("\n", "")
        logger.info(f"Image summary generated: path={image_path}")
        return summary

    except LangChainException as e:
        logger.error(f"Image summary failed with LangChain error: path={image_path}, error={e}")
        return "图片描述"
    except Exception as e:
        logger.error(f"Image summary failed: path={image_path}, error={e}")
        return "图片描述"

def step_3_generate_summaries(doc_stem: str, targets: List[Tuple[str, str, Tuple[str, str]]],
                              requests_per_minute: int = 9) -> Dict[str, str]:
    """Generate summaries for referenced images under a request-rate limit."""
    summaries = {}
    request_times = deque()

    for img_file, image_path, context in targets:
        apply_api_rate_limit(request_times, requests_per_minute, window_seconds=60)
        logger.debug(f"Generating image summary: path={image_path}")
        summaries[img_file] = summarize_image(image_path, root_folder=doc_stem, image_content=context)

    logger.info(f"Image summary generation completed: images={len(summaries)}")
    return summaries


def clean_minio_directory(minio_client: Minio, prefix: str) -> None:
    """Remove objects under a document prefix before uploading refreshed images."""
    try:
        objects_to_delete = minio_client.list_objects(
            bucket_name=minio_config.bucket_name,
            prefix=prefix,
            recursive=True
        )
        delete_list = [DeleteObject(obj.object_name) for obj in objects_to_delete]

        if delete_list:
            logger.info(f"Removing stale MinIO objects: prefix={prefix}, objects={len(delete_list)}")
            errors = minio_client.remove_objects(minio_config.bucket_name, delete_list)
            for error in errors:
                logger.error(f"MinIO object removal failed: {error}")
        else:
            logger.debug(f"No stale MinIO objects found: prefix={prefix}")
    except Exception as e:
        logger.error(f"MinIO cleanup failed: prefix={prefix}, error={e}")


def upload_images_batch(minio_client: Minio, upload_dir: str, targets: List[Tuple[str, str, Tuple[str, str]]]) -> Dict[
    str, str]:
    """Upload referenced images and return filename-to-URL mappings."""
    urls = {}
    for img_file, img_path, _ in targets:
        object_name = f"{upload_dir}/{img_file}"
        logger.debug(f"MinIO object path prepared: {object_name}")
        if img_url := upload_to_minio(minio_client, img_path, object_name):
            urls[img_file] = img_url
    logger.info(f"Image upload completed: uploaded={len(urls)}, requested={len(targets)}")
    return urls

def upload_to_minio(minio_client: Minio, local_path: str, object_name: str) -> str | None:
    """Upload one image to MinIO and return its access URL."""
    try:
        logger.info(f"Uploading image to MinIO: path={local_path}, object={object_name}")
        minio_client.fput_object(
            bucket_name=minio_config.bucket_name,
            object_name=object_name,
            file_path=local_path,
            content_type=f"image/{os.path.splitext(local_path)[1][1:]}"
        )

        object_name = object_name.replace("\\", "%5C")
        protocol = "https" if minio_config.minio_secure else "http"
        base_url = f"{protocol}://{minio_config.endpoint}/{minio_config.bucket_name}"
        img_url = f"{base_url}{object_name}"
        logger.info(f"Image uploaded to MinIO: object={object_name}")
        return img_url
    except Exception as e:
        logger.error(f"MinIO image upload failed: path={local_path}, error={e}")
        return None

def merge_summary_and_url(summaries: Dict[str, str], urls: Dict[str, str]) -> Dict[str, Tuple[str, str]]:
    """Join image summaries with uploaded URLs and discard incomplete records."""
    image_info = {}
    for image_file, summary in summaries.items():
        if url := urls.get(image_file):
            image_info[image_file] = (summary, url)
    logger.info(f"Image metadata assembled: images={len(image_info)}")
    return image_info


def process_md_file(md_content: str, image_info: Dict[str, Tuple[str, str]]) -> str:
    """Replace local Markdown image references with summarized MinIO URLs."""
    for img_filename, (summary, new_url) in image_info.items():
        pattern = re.compile(
            r"!\[.*?\]\(.*?" + re.escape(img_filename) + r".*?\)",
            re.IGNORECASE
        )
        replacement_text = f"![{summary}]({new_url})"
        md_content = pattern.sub(lambda _: replacement_text, md_content)
        logger.debug(f"Markdown image reference replaced: filename={img_filename}")

    logger.info(f"Markdown image references replaced: images={len(image_info)}")
    return md_content

def step_4_upload_and_replace(minio_client: Minio, doc_stem: str, targets: List[Tuple[str, str, Tuple[str, str]]],
                              summaries: Dict[str, str], md_content: str) -> str:
    """Upload images, assemble metadata, and rewrite Markdown references."""
    minio_img_dir = minio_config.minio_img_dir
    upload_dir = f"{minio_img_dir}/{doc_stem}".replace(" ", "")

    clean_minio_directory(minio_client, upload_dir)
    urls = upload_images_batch(minio_client, upload_dir, targets)
    image_info = merge_summary_and_url(summaries, urls)
    if image_info:
        md_content = process_md_file(md_content, image_info)

    return md_content


def step_5_backup_new_md_file(origin_md_path: str, md_content: str) -> str:
    """Write processed Markdown to a sibling file with the ``_new`` suffix."""
    new_md_file_name = os.path.splitext(origin_md_path)[0] + "_new.md"

    with open(new_md_file_name, "w", encoding="utf-8") as f:
        f.write(md_content)

    logger.info(f"Processed Markdown written: path={new_md_file_name}")
    return new_md_file_name


def node_md_img(state: ImportGraphState) -> ImportGraphState:
    """Summarize referenced images, upload them, and update Markdown state."""
    add_running_task(state["task_id"], sys._getframe().f_code.co_name)

    md_content, path_obj, images_dir = step_1_get_content(state)
    state["md_content"] = md_content

    if not images_dir.exists():
        logger.info(f"Image directory not found; skipping image processing: {images_dir.absolute()}")
        return state

    minio_client = get_minio_client()
    if not minio_client:
        logger.warning("MinIO client unavailable; image processing skipped")
        return state
    
    targets = step_2_scan_images(md_content, images_dir)
    if not targets:
        logger.info("No supported Markdown image references found")
        return state

    summaries = step_3_generate_summaries(path_obj.stem, targets)

    new_md_content = step_4_upload_and_replace(minio_client, path_obj.stem, targets, summaries, md_content)
    state["md_content"] = new_md_content

    new_md_file_name = step_5_backup_new_md_file(state['md_path'], new_md_content)
    state["md_path"] = new_md_file_name
    logger.info(f"Markdown image processing completed: path={new_md_file_name}")

    return state
