from app.core.logger import logger


def _debug_step(message: str, enabled: bool):
    if enabled:
        print(message, flush=True)

def _load_graph_dependencies(debug: bool = False):
    """
    延后导入所有重依赖，避免模块导入期直接触发原生层初始化。
    """
    _debug_step("[debug] import dotenv.load_dotenv - start", debug)
    from dotenv import load_dotenv
    _debug_step("[debug] import dotenv.load_dotenv - ok", debug)

    _debug_step("[debug] import langgraph.graph - start", debug)
    from langgraph.graph import StateGraph, END
    _debug_step("[debug] import langgraph.graph - ok", debug)

    _debug_step("[debug] import ImportGraphState - start", debug)
    from app.import_process.agent.state import ImportGraphState
    _debug_step("[debug] import ImportGraphState - ok", debug)

    _debug_step("[debug] import node_entry - start", debug)
    from app.import_process.agent.nodes.node_entry import node_entry
    _debug_step("[debug] import node_entry - ok", debug)

    _debug_step("[debug] import node_pdf_to_md - start", debug)
    from app.import_process.agent.nodes.node_pdf_to_md import node_pdf_to_md
    _debug_step("[debug] import node_pdf_to_md - ok", debug)

    _debug_step("[debug] import node_md_img - start", debug)
    from app.import_process.agent.nodes.node_md_img import node_md_img
    _debug_step("[debug] import node_md_img - ok", debug)

    _debug_step("[debug] import node_document_split - start", debug)
    from app.import_process.agent.nodes.node_document_split import node_document_split
    _debug_step("[debug] import node_document_split - ok", debug)

    _debug_step("[debug] import node_item_name_recognition - start", debug)
    from app.import_process.agent.nodes.node_item_name_recognition import node_item_name_recognition
    _debug_step("[debug] import node_item_name_recognition - ok", debug)

    _debug_step("[debug] import node_bge_embedding - start", debug)
    from app.import_process.agent.nodes.node_bge_embedding import node_bge_embedding
    _debug_step("[debug] import node_bge_embedding - ok", debug)

    _debug_step("[debug] import node_import_milvus - start", debug)
    from app.import_process.agent.nodes.node_import_milvus import node_import_milvus
    _debug_step("[debug] import node_import_milvus - ok", debug)

    _debug_step("[debug] load_dotenv - start", debug)
    load_dotenv()
    _debug_step("[debug] load_dotenv - ok", debug)

    return {
        "StateGraph": StateGraph,
        "END": END,
        "ImportGraphState": ImportGraphState,
        "node_entry": node_entry,
        "node_pdf_to_md": node_pdf_to_md,
        "node_md_img": node_md_img,
        "node_document_split": node_document_split,
        "node_item_name_recognition": node_item_name_recognition,
        "node_bge_embedding": node_bge_embedding,
        "node_import_milvus": node_import_milvus,
    }


def debug_load_graph_dependencies():
    """
    分步导入诊断工具。
    用于定位哪一个依赖导入会触发解释器静默崩溃。
    """
    return _load_graph_dependencies(debug=True)


def _build_workflow():
    """
    构建导入图工作流。
    这里延后创建 StateGraph 和注册边，避免模块导入期直接触发 LangGraph 原生层初始化。
    """
    # ===================== 1. 初始化LangGraph状态图 =====================
    # 核心：StateGraph是LangGraph的核心类，用于构建有状态的工作流
    # 参数ImportGraphState：自定义TypedDict类型，定义了工作流的**全量状态字段**
    # 作用：所有节点的入参都是该状态对象，节点返回的键值对会自动合并回状态，实现节点间数据共享
    deps = _load_graph_dependencies()
    StateGraph = deps["StateGraph"]
    ImportGraphState = deps["ImportGraphState"]
    END = deps["END"]

    workflow = StateGraph(ImportGraphState)

    # ===================== 2. 注册所有业务节点 =====================
    # 语法：add_node("节点唯一标识", 节点函数)
    # 要求：节点函数必须接收「状态对象」作为入参，返回字典（用于更新状态）
    # 所有节点按「知识库导入流程」先后顺序注册，节点标识与函数名保持一致，便于维护
    workflow.add_node("node_entry", deps["node_entry"])  # 流程入口：参数初始化、输入校验
    workflow.add_node("node_pdf_to_md", deps["node_pdf_to_md"])  # PDF转MD：非MD格式文件的前置处理
    workflow.add_node("node_md_img", deps["node_md_img"])  # MD图片处理：保证文档中图片的可访问性
    workflow.add_node("node_document_split", deps["node_document_split"])  # 文档分块：解决大文本无法向量化/推理的问题
    workflow.add_node("node_item_name_recognition", deps["node_item_name_recognition"])  # 项目名识别：业务定制化步骤，提取核心业务标识
    workflow.add_node("node_bge_embedding", deps["node_bge_embedding"])  # BGE向量化：文本→向量，为Milvus存储做准备
    workflow.add_node("node_import_milvus", deps["node_import_milvus"])  # 向量入库：将向量数据持久化到Milvus

    # ===================== 3. 设置工作流入口节点 =====================
    # 语法：set_entry_point("节点标识") → 推荐写法，直接指定流程起始节点
    # 作用：指定工作流执行的第一个节点
    workflow.set_entry_point("node_entry")

    # ===================== 4. 注册条件边 =====================
    workflow.add_conditional_edges(
        "node_entry",
        route_after_entry,
        {
            "node_md_img": "node_md_img",
            "node_pdf_to_md": "node_pdf_to_md",
            END: END
        }
    )

    # ===================== 5. 注册静态顺序边 =====================
    workflow.add_edge("node_pdf_to_md", "node_md_img")  # PDF转MD完成 → MD图片处理
    workflow.add_edge("node_md_img", "node_document_split")  # MD处理完成 → 文档分块
    workflow.add_edge("node_document_split", "node_item_name_recognition")  # 分块完成 → 项目名识别
    workflow.add_edge("node_item_name_recognition", "node_bge_embedding")  # 项目名识别完成 → BGE向量化
    workflow.add_edge("node_bge_embedding", "node_import_milvus")  # 向量化完成 → 导入Milvus向量库
    workflow.add_edge("node_import_milvus", END)  # Milvus入库完成 → 工作流执行结束（END是内置结束节点）

    return workflow

# ===================== 4. 定义条件路由函数（入口节点后的分支逻辑） =====================
# 核心：根据状态中的配置项，动态决定后续执行路径，实现「PDF导入」/「MD直接导入」分支
# 要求：接收状态对象为入参，返回「目标节点标识」或END（内置结束常量）
def route_after_entry(state) -> str:
    """
    入口节点后的条件路由逻辑
    :param state: 工作流全量状态对象，包含所有配置项和中间结果
    :return: 目标节点标识/END，LangGraph会自动跳转到对应节点
    """
    # 分支1：开启MD直接导入 → 跳过PDF转MD，直接执行MD图片处理
    if state.get("is_md_read_enabled"):
        return "node_md_img"
    # 分支2：开启PDF导入 → 执行PDF转MD，再走后续流程
    elif state.get("is_pdf_read_enabled"):
        return "node_pdf_to_md"
    # 分支3：未开启任何导入配置 → 直接终止工作流（END是LangGraph内置结束常量）
    else:
        from langgraph.graph import END
        return END

# ===================== 6. 懒加载编译工作流 =====================
# 核心：避免模块导入阶段直接 compile()，把启动期原生崩溃与运行期业务错误拆开
# 作用：服务真正执行导入任务时才编译工作流，便于排障，也降低启动副作用
_kb_import_app = None
_workflow = None


def get_kb_import_app():
    """
    获取导入图的可执行实例（首次调用时编译，后续复用）
    """
    global _kb_import_app

    global _workflow

    if _kb_import_app is None:
        logger.info("开始构建导入图工作流（首次懒加载）")
        _workflow = _build_workflow()
        logger.info("导入图工作流构建完成，开始编译")
        _kb_import_app = _workflow.compile()
        logger.info("导入图工作流编译完成")

    return _kb_import_app
