import shutil
import time
import zipfile
from pathlib import Path

import requests

from app.core.logger import logger, node_log, step_log
from app.import_process.agent.state import ImportGraphState
from app.utils.path_util import PROJECT_ROOT
from app.utils.task_utils import add_done_task, add_running_task
from app.conf.mineru_config import mineru_config
"""Convert PDF documents to Markdown through the MinerU extraction service."""


@step_log("step_1_validate_paths")
def step_1_validate_paths(state):
    """Validate the PDF path and initialize the extraction output directory."""
    pdf_path = state.get("pdf_path", "").strip()
    local_dir = state.get("local_dir", "").strip()
    if not pdf_path:
        raise ValueError("pdf_path is required")
    if not local_dir:
        local_dir = PROJECT_ROOT / "output"
        state["local_dir"] = str(local_dir)
        logger.warning(f"Output directory not configured; using {local_dir}")
    pdf_path_obj = Path(pdf_path)
    local_dir_obj = Path(local_dir)
    if not pdf_path_obj.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path_obj}")
    if not local_dir_obj.exists():
        logger.warning(f"Creating extraction output directory: {local_dir_obj}")
        local_dir_obj.mkdir(parents=True, exist_ok=True)
    return pdf_path_obj, local_dir_obj


@step_log("step_2_upload_and_poll")
def step_2_upload_and_poll(pdf_path_obj, output_dir_obj):
    """Upload a PDF to MinerU and poll until an extraction archive is ready."""
    if not mineru_config.base_url or not mineru_config.api_key:
        raise ValueError("MinerU base URL and API key are required")

    request_headers = {
        "Authorization": f"Bearer {mineru_config.api_key}",
        "Content-Type": "application/json"
    }
    url_get_upload = f"{mineru_config.base_url}/file-urls/batch"
    req_data = {
        "files": [{"name": pdf_path_obj.name}],
        "model_version": "vlm"
    }

    response = requests.post(url_get_upload, json=req_data, headers=request_headers)
    if response.status_code != 200:
        raise RuntimeError(f"MinerU upload URL request failed: status={response.status_code}")

    resp_data = response.json()
    if resp_data["code"] != 0:
        raise RuntimeError(f"MinerU upload URL request rejected: code={resp_data['code']}, message={resp_data['msg']}")

    signed_url = resp_data["data"]["file_urls"][0]
    batch_id = resp_data["data"]["batch_id"]

    file_data = pdf_path_obj.read_bytes()

    with requests.Session() as session:
        session.trust_env = False
        put_response = session.put(signed_url, data=file_data, timeout=60)

        if put_response.status_code != 200:
            raise RuntimeError(f"PDF upload failed: status={put_response.status_code}")
    poll_url = f"{mineru_config.base_url}/extract-results/batch/{batch_id}"
    timeout_seconds = 600
    poll_interval = 3
    start_time = time.time()

    logger.debug("Polling MinerU extraction status")

    while True:
        if time.time() - start_time > timeout_seconds:
            raise TimeoutError(f"MinerU extraction timed out after {timeout_seconds} seconds")
        try:
            poll_response = requests.get(poll_url, headers=request_headers, timeout=10)
        except Exception as e:
            logger.warning(f"MinerU polling request failed; retrying: {e}")
            time.sleep(poll_interval)
            continue
        status_code = poll_response.status_code
        if status_code != 200:
            if 500 <= status_code < 600:
                logger.warning(f"MinerU returned server error {status_code}; retrying")
                time.sleep(poll_interval)
                continue
            else:
                raise RuntimeError(f"MinerU polling request failed: status={status_code}")
        poll_data = poll_response.json()
        if poll_data["code"] != 0:
            raise RuntimeError(f"MinerU polling rejected: code={poll_data['code']}, message={poll_data['msg']}")
        extract_results = poll_data["data"]["extract_result"]
        if not extract_results:
            logger.debug("MinerU extraction is still pending")
            time.sleep(poll_interval)
            continue
        task_state = extract_results[0]["state"]
        if task_state == "done":
            full_zip_url = extract_results[0]["full_zip_url"]
            if not full_zip_url:
                raise RuntimeError("MinerU completed without a valid result archive URL")
            logger.info("MinerU extraction completed; downloading result archive")
            return full_zip_url
        elif task_state == "failed":
            raise RuntimeError("MinerU extraction failed")
        else:
            logger.debug(f"MinerU extraction state={task_state}; continuing to poll")
            time.sleep(poll_interval)




@step_log("step_3_download_and_extract")
def step_3_download_and_extract(zip_url:str, output_dir_pbj:Path, stem:str):
    """Download the MinerU archive, extract it, and return the normalized Markdown path."""
    response = requests.get(zip_url, timeout=120)
    if response.status_code != 200:
        raise RuntimeError(f"MinerU result archive download failed: status={response.status_code}")

    zip_save_path = output_dir_pbj / f"{stem}_result.zip"
    zip_save_path.write_bytes(response.content)

    extract_target_dir = output_dir_pbj / stem
    if extract_target_dir.exists():
        shutil.rmtree(extract_target_dir)

    extract_target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_save_path, 'r') as zip_ref:
        zip_ref.extractall(extract_target_dir)

    md_file_list = list(extract_target_dir.rglob("*.md"))
    if not md_file_list or len(md_file_list) == 0:
        raise FileNotFoundError("MinerU archive did not contain a Markdown file")

    target_md_file = None
    for md_file in md_file_list:
        if md_file.stem == stem:
            target_md_file = md_file
            break
    if  not target_md_file :
        for md_file in md_file_list:
            if md_file.name.lower() == "full.md":
                target_md_file = md_file
                break

    if not target_md_file:
        target_md_file = md_file_list[0]

    if target_md_file.stem != stem:
        target_md_file = target_md_file.rename(target_md_file.with_name(f"{stem}.md"))

    final_md_str_path = str(target_md_file.resolve())
    return final_md_str_path



@node_log("node_pdf_to_md")
def node_pdf_to_md(state: ImportGraphState) -> ImportGraphState:
    """Convert the input PDF to Markdown and store the content in import state."""
    add_running_task(state['task_id'], 'node_pdf_to_md')
    pdf_path_obj , output_dir_pbj = step_1_validate_paths(state)
    zip_url = step_2_upload_and_poll(pdf_path_obj, output_dir_pbj)
    md_path = step_3_download_and_extract(zip_url, output_dir_pbj, pdf_path_obj.stem)
    state['md_path'] = md_path
    with open(md_path, 'r', encoding='utf-8') as f:
        state['md_content'] = f.read()
    add_done_task(state['task_id'], 'node_pdf_to_md')
    return state



