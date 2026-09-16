"""Markdown chunking pipeline for document ingestion."""

import re
import json
import os
import sys
from typing import List, Dict, Any, Tuple

from app.utils.task_utils import add_running_task
from app.import_process.agent.state import ImportGraphState
from app.core.logger import logger

# Chunking defaults.
DEFAULT_MAX_CONTENT_LENGTH = 2000
MIN_CONTENT_LENGTH = 500


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _chunk_strategy() -> str:
    return str(os.environ.get("EVAL_CHUNK_STRATEGY", "semantic_title") or "semantic_title").strip().lower()


def step_1_get_inputs(state: ImportGraphState) -> Tuple[Any, str, int, str]:
    """Load and normalize Markdown content and chunking parameters."""
    content = state.get("md_content")
    if not content:
        logger.warning("Document chunking skipped because Markdown content is empty")
        return None, None, None, None

    content = content.replace("\r\n", "\n").replace("\r", "\n")
    file_title = state.get("file_title", "Unknown File")
    doc_id = state.get("doc_id", "")
    max_len = _env_int("EVAL_CHUNK_MAX_LEN", DEFAULT_MAX_CONTENT_LENGTH)

    logger.info(f"Chunking inputs loaded: file_title={file_title}, doc_id={doc_id or '-'}, max_length={max_len}")
    return content, file_title, max_len, doc_id


def step_2_split_by_titles(content: str, file_title: str, doc_id: str) -> Tuple[List[Dict[str, Any]], int, int]:
    """Split Markdown into title-based sections while respecting code fences."""
    title_pattern = r'^\s*#{1,6}\s+.+'

    lines = content.split("\n")
    sections = []
    current_title = ""
    current_lines = []
    title_count = 0
    in_code_block = False

    def _flush_section():
        """Append the current section when it contains content."""
        if not current_lines:
            return
        sections.append({
            "title": current_title,
            "content": "\n".join(current_lines),
            "file_title": file_title,
            "doc_id": doc_id,
        })

    for line in lines:
        stripped_line = line.strip()
        if stripped_line.startswith("```") or stripped_line.startswith("~~~"):
            in_code_block = not in_code_block
            current_lines.append(line)
            continue

        is_valid_title = (not in_code_block) and re.match(title_pattern, line)
        if is_valid_title:
            _flush_section()
            current_title = line.strip()
            current_lines = [current_title]
            title_count += 1
            logger.debug(f"Markdown heading detected: {current_title}")
        else:
            current_lines.append(line)

    _flush_section()
    logger.info(f"Title-based chunking completed: headings={title_count}, lines={len(lines)}")
    return sections, title_count, len(lines)


def step_3_handle_no_title(
    content: str,
    sections: List[Dict[str, Any]],
    title_count: int,
    file_title: str,
    doc_id: str
) -> List[Dict[str, Any]]:
    """Return a single document section when no Markdown headings are found."""
    if title_count == 0:
        logger.warning(f"No Markdown headings found; using one section: file_title={file_title}")
        return [{"title": "无标题", "content": content, "file_title": file_title, "doc_id": doc_id}]
    return sections

def _split_long_section(section: Dict[str, Any], max_length: int = DEFAULT_MAX_CONTENT_LENGTH) -> List[Dict[str, Any]]:
    """Split an oversized section with an ordered separator strategy."""
    content = section.get("content", "") or ""
    if len(content) <= max_length:
        return [section]

    content = content.replace("\r\n", "\n").replace("\r", "\n")
    title = section.get("title", "") or ""
    prefix = f"{title}\n\n" if title else ""
    available_len = max_length - len(prefix)
    if available_len <= 0:
        logger.warning(f"Section title exceeds chunk budget: {title[:20]}...")
        return [section]

    body = content
    if title and body.lstrip().startswith(title):
        body = body[body.find(title) + len(title):].lstrip()

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except Exception as e:
        logger.error(f"Failed to load RecursiveCharacterTextSplitter: {e}", exc_info=False)
        raise

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=available_len,
        chunk_overlap=_env_int("EVAL_CHUNK_OVERLAP", 0),
        separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " "],
    )

    sub_sections = []
    for idx, chunk in enumerate(splitter.split_text(body), start=1):
        text = chunk.strip()
        if not text:
            continue
        full_text = (prefix + text).strip()
        sub_sections.append({
            "title": f"{title}-{idx}" if title else f"chunk-{idx}",
            "content": full_text,
            "parent_title": title,
            "part": idx,
            "file_title": section.get("file_title"),
            "doc_id": section.get("doc_id", ""),
        })

    logger.debug(f"Oversized section split: title={title}, chunks={len(sub_sections)}")
    return sub_sections

def _merge_short_sections(sections: List[Dict[str, Any]], min_length: int = MIN_CONTENT_LENGTH) -> List[Dict[str, Any]]:
    """Merge adjacent short sections that share the same parent heading."""
    if not sections:
        logger.debug("Short-section merge skipped because input is empty")
        return []

    merged_sections = []
    current_chunk = None

    for sec in sections:
        if current_chunk is None:
            current_chunk = sec
            continue

        is_current_short = len(current_chunk["content"]) < min_length
        is_same_parent = current_chunk.get("parent_title") == sec.get("parent_title")

        if is_current_short and is_same_parent:
            parent_title = sec.get("parent_title", "")
            next_content = sec["content"]
            if parent_title and next_content.startswith(parent_title):
                next_content = next_content[len(parent_title):].lstrip()
            current_chunk["content"] += "\n\n" + next_content
            if "part" in sec:
                current_chunk["part"] = sec["part"]
            logger.debug(
                f"Short sections merged: parent={current_chunk.get('parent_title')}, "
                f"length={len(current_chunk['content'])}"
            )
        else:
            merged_sections.append(current_chunk)
            current_chunk = sec

    if current_chunk is not None:
        merged_sections.append(current_chunk)

    logger.debug(f"Short-section merge completed: input={len(sections)}, output={len(merged_sections)}")
    return merged_sections

def step_4_refine_chunks(sections: List[Dict[str, Any]], max_len: int) -> List[Dict[str, Any]]:
    """Refine sections by splitting oversized content and merging short siblings."""
    if not max_len or max_len <= 0:
        logger.warning(f"Chunk refinement skipped because max_length={max_len} is invalid")
        return sections

    refined_split = []
    for sec in sections:
        refined_split.extend(_split_long_section(sec, max_len))
    logger.info(f"Oversized section splitting completed: chunks={len(refined_split)}")

    if os.environ.get("EVAL_CHUNK_DISABLE_SHORT_MERGE", "").strip().lower() in {"1", "true", "yes", "on"}:
        final_sections = refined_split
        logger.info(f"Short-section merge disabled: chunks={len(final_sections)}")
    else:
        final_sections = _merge_short_sections(refined_split)
        logger.info(f"Short-section merge completed: chunks={len(final_sections)}")

    for sec in final_sections:
        if not isinstance(sec, dict):
            continue
        
        if "part" not in sec:
            sec["part"] = 0
            
        if not sec.get("parent_title"):
            sec["parent_title"] = sec.get("title") or ""
    logger.debug("Chunk metadata normalization completed")

    return final_sections


def step_fixed_window_split(
    content: str,
    file_title: str,
    doc_id: str,
    chunk_size: int,
    chunk_overlap: int,
) -> List[Dict[str, Any]]:
    """Split content with the fixed-window evaluation strategy."""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except Exception as e:
        logger.error(f"Failed to load RecursiveCharacterTextSplitter: {e}", exc_info=False)
        raise

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max(chunk_size, 100),
        chunk_overlap=max(0, min(chunk_overlap, chunk_size // 2)),
        separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " "],
    )
    chunks = []
    for idx, text in enumerate(splitter.split_text(content), start=1):
        text = text.strip()
        if not text:
            continue
        title = f"{file_title}-fixed-{idx}"
        chunks.append({
            "title": title,
            "content": text,
            "parent_title": file_title,
            "part": idx,
            "file_title": file_title,
            "doc_id": doc_id,
        })
    logger.info(f"Fixed-window chunking completed: size={chunk_size}, overlap={chunk_overlap}, chunks={len(chunks)}")
    return chunks

def step_5_print_stats(lines_count: int, sections: List[Dict[str, Any]]) -> None:
    """Log chunking statistics for operational visibility."""
    chunk_num = len(sections)
    logger.info(f"Chunking statistics: lines={lines_count}, chunks={chunk_num}")
    if sections:
        first_title = sections[0].get("title", "无标题")
        logger.info(f"First chunk title: {first_title}")

def step_6_backup(state: ImportGraphState, sections: List[Dict[str, Any]]) -> None:
    """Persist chunk metadata to the configured local output directory."""
    local_dir = state.get("local_dir")
    if not local_dir:
        logger.warning("Chunk backup skipped because local_dir is not configured")
        return

    try:
        os.makedirs(local_dir, exist_ok=True)
        backup_path = os.path.join(local_dir, "chunks.json")
        with open(backup_path, "w", encoding="utf-8") as f:
            json.dump(
                sections,
                f,
                ensure_ascii=False,
                indent=2,
            )
        logger.info(f"Chunk backup written: path={backup_path}")
    except Exception as e:
        logger.error(f"Chunk backup failed: {e}", exc_info=False)

def node_document_split(state: ImportGraphState) -> ImportGraphState:
    """Split Markdown content into searchable chunks and update import state."""
    node_name = sys._getframe().f_code.co_name
    logger.info(f"{node_name} started")
    add_running_task(state["task_id"], node_name)

    try:
        content, file_title, max_len, doc_id = step_1_get_inputs(state)
        if content is None:
            logger.info(f"{node_name} stopped: Markdown content is empty")
            return state

        strategy = _chunk_strategy()
        if strategy == "fixed":
            sections = step_fixed_window_split(
                content=content,
                file_title=file_title,
                doc_id=doc_id,
                chunk_size=max_len,
                chunk_overlap=_env_int("EVAL_CHUNK_OVERLAP", 0),
            )
            step_5_print_stats(len(content.split("\n")), sections)
            state["chunks"] = sections
            step_6_backup(state, sections)
            logger.info(
                f"{node_name} completed: strategy=fixed, chunks={len(sections)}"
            )
            return state

        sections, title_count, lines_count = step_2_split_by_titles(content, file_title, doc_id)

        sections = step_3_handle_no_title(content, sections, title_count, file_title, doc_id)

        sections = step_4_refine_chunks(sections, max_len)

        step_5_print_stats(lines_count, sections)

        state["chunks"] = sections
        step_6_backup(state, sections)

        logger.info(f"{node_name} completed: chunks={len(sections)}")

    except Exception as e:
        logger.error(f"{node_name} failed: {e}", exc_info=True)

    return state
