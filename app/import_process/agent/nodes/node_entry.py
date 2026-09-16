from pathlib import Path
from app.core.logger import logger, node_log
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_running_task, add_done_task
from app.utils.file_hash_utils import build_doc_id

@node_log("node_entry")
def node_entry(state: ImportGraphState) -> ImportGraphState:
    """Validate the input file and initialize format-specific state fields."""
    add_running_task(state['task_id'],'node_entry')

    local_file_path = state['local_file_path']
    if not local_file_path:
        logger.warning("Import skipped because no input file was provided")
        add_done_task(state['task_id'], 'node_entry')
        return state

    if local_file_path.endswith(".md"):
        state['is_md_read_enabled'] = True
        state['md_path'] = local_file_path
    elif local_file_path.endswith(".pdf"):
        state['is_pdf_read_enabled'] = True
        state['pdf_path'] = local_file_path
    else:
        logger.warning(f"Unsupported import file type: path={local_file_path}")
        add_done_task(state['task_id'], 'node_entry')
        return state

    file_title = Path(local_file_path).stem
    state['file_title']= file_title
    if not state.get("doc_id"):
        state["doc_id"] = build_doc_id(local_file_path)
    logger.info(f"Import metadata initialized: file_title={file_title}, doc_id={state['doc_id']}")

    add_done_task(state['task_id'], 'node_entry')
    return state

