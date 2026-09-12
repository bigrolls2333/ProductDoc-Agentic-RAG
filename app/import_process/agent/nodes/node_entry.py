import os

from pathlib import Path
from app.core.logger import logger, node_log
from app.import_process.agent.state import ImportGraphState, create_default_state
from app.utils.task_utils import add_running_task, add_done_task
from app.utils.file_hash_utils import build_doc_id

@node_log("node_entry")
def node_entry(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 入口节点 (node_entry)
    为什么叫这个名字: 作为图的 Entry Point，负责接收外部输入并决定流程走向。
    未来要实现:
        1. 进行任务状态记录,开始和结束列表记录
        2. 根据state中 local_file_path属性判断数据类型进而修改
           相关参数, is_md_read_enabled 或者 is_pdf_read_enabled
                    md_path 或者 pdf_path
        3. 不可解析结果类型不可用,直接输出对应警告日志! 逻辑路由节点会自动处理
        4. 获取file_tile标识,用于后期识别pdf对应的主体(item_name)进行兜底
    """

    # 1. 任务状态记录处理
    add_running_task(state['task_id'],'node_entry')

    # 2. 判断文件类型
    local_file_path = state['local_file_path']
    if not local_file_path:
        logger.warning(f"没有输入文件地址,无法处理,直接跳转到结束节点!")
        add_done_task(state['task_id'], 'node_entry')
        return state

    if local_file_path.endswith(".md"):
        state['is_md_read_enabled'] = True
        state['md_path'] = local_file_path
    elif local_file_path.endswith(".pdf"):
        state['is_pdf_read_enabled'] = True
        state['pdf_path'] = local_file_path
    else:
        logger.warning(f"虽然输出了loclal_file_path,但是无法识别文件类型,请检查输入文件类型是否正确,目前只支持md和pdf文件,请检查! {local_file_path}")
        add_done_task(state['task_id'], 'node_entry')
        return state

    # 3. 获取文件标识
    # 基于os.path处理
    file_title_os = os.path.basename(local_file_path).split(".")[0]
    # 基于pathlib处理
    file_title = Path(local_file_path).stem # 文件名 .name  文件夹名 .parent   文件后缀 .suffix
    state['file_title']= file_title
    if not state.get("doc_id"):
        state["doc_id"] = build_doc_id(local_file_path)
    logger.info(f"入口节点已生成文档标识：file_title={file_title} | doc_id={state['doc_id']}")

    add_done_task(state['task_id'], 'node_entry')
    return state

