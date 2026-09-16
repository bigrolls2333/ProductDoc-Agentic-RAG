"""Public task-progress facade used by the HTTP and graph layers."""

from typing import Dict, List
from .sse_utils import push_to_session
from .task_store import get_task_store

TASK_STATUS_PENDING = "pending"
TASK_STATUS_PROCESSING = "processing"
TASK_STATUS_COMPLETED = "completed"
TASK_STATUS_FAILED = "failed"

# Graph node identifiers mapped to UI labels.
_NODE_NAME_TO_CN: Dict[str, str] = {
    "upload_file": "开始上传文件",
    "node_entry": "检查文件",
    "node_pdf_to_md": "PDF转Markdown",
    "node_md_img": "Markdown图片处理",
    "node_item_name_recognition": "主体名称识别",
    "node_document_split": "文档切分",
    "node_bge_embedding": "向量生成",
    "node_import_milvus": "导入向量库",
    "__end__": "处理完成",
    "END": "处理完成",
    # Query graph nodes
    "node_query_intent": "识别查询意图",
    "node_direct_answer": "直接回复",
    "node_item_name_confirm": "确认问题产品",
    "node_doc_need_router": "判断是否查文档",
    "node_retrieval_plan": "生成检索计划",
    "node_answer_output": "生成答案",
    "node_rerank": "重排序",
    "node_rrf": "倒排融合",
    "node_web_search_mcp": "网络搜索",
    "node_search_embedding": "切片搜索",
    "node_search_embedding_hyde": "切片搜索(假设性文档)",
    "node_multi_search": "多路搜索",
    "node_join": "多路搜索合并",
}


def _to_cn(node_name: str) -> str:
    """Return the UI label for a node, falling back to its identifier."""
    return _NODE_NAME_TO_CN.get(node_name, node_name)


def add_running_task(task_id: str, node_name: str, is_stream: bool = False) -> None:
    """Mark a task stage as active and optionally publish progress."""
    get_task_store().add_running(task_id, node_name)

    if is_stream:
        task_push_queue(task_id)


def add_done_task(task_id: str, node_name: str, is_stream: bool = False) -> None:
    """Mark a task stage as complete and optionally publish progress."""
    get_task_store().add_done(task_id, node_name)

    if is_stream:
        task_push_queue(task_id)


def set_task_result(task_id: str, key: str, value: str) -> None:
    """Store a result field for a task."""
    get_task_store().set_result(task_id, key, value)


def get_task_result(task_id: str, key: str, default: str = "") -> str:
    """Return a result field for a task."""
    return get_task_store().get_result(task_id, key, default)


def get_task_status(task_id: str) -> str:
    """Return the current status for a task."""
    return get_task_store().get_status(task_id)


def get_done_task_list(task_id: str) -> List[str]:
    """Return completed task stages using UI labels."""
    done = get_task_store().get_done(task_id)
    return [_to_cn(n) for n in done]


def get_running_task_list(task_id: str) -> List[str]:
    """Return active task stages using UI labels."""
    running = get_task_store().get_running(task_id)
    return [_to_cn(n) for n in running]


def update_task_status(task_id: str, status_name: str, push_queue: bool = False) -> None:
    """Update task status and optionally publish a progress event."""
    get_task_store().update_status(task_id, status_name)
    if push_queue:
        task_push_queue(task_id)


def task_push_queue(task_id: str):
    push_to_session(task_id, "progress", {
        "status": get_task_status(task_id),
        "done_list": get_done_task_list(task_id),
        "running_list": get_running_task_list(task_id),
    })


def clear_task(task_id: str):
    """Remove all stored state for a task."""
    get_task_store().clear(task_id)
