"""MCP-backed web retrieval node with structured result normalization."""

import sys
import json
import asyncio
from app.utils.task_utils import add_done_task, add_running_task
from app.utils.eval_flag_utils import is_eval_enabled
from app.conf.bailian_mcp_config import mcp_config
from agents.mcp import MCPServerStreamableHttp
from app.core.logger import logger

async def mcp_call(query):
    """Call the configured WebSearch MCP endpoint and return its raw result."""
    search_mcp = MCPServerStreamableHttp(
        name="search_mcp",
        params={
            "url": mcp_config.mcp_base_url,
            "headers": {"Authorization": mcp_config.api_key},
            "timeout": 300
        }
    )

    try:
        logger.info(f"MCP WebSearch connecting: endpoint={mcp_config.mcp_base_url}")
        await search_mcp.connect()
        
        logger.info(f"MCP WebSearch connected: query={query}")
        result = await search_mcp.call_tool(
            tool_name="bailian_web_search",
            arguments={"query": query, "count": 5}
        )
        logger.info("MCP WebSearch call completed")
        return result
        
    except Exception as e:
        logger.error(f"MCP WebSearch call failed: {e}", exc_info=True)
        return None
        
    finally:
        await search_mcp.cleanup()


def node_web_search_mcp(state):
    """Run WebSearch MCP and return normalized web evidence."""
    logger.info("node_web_search_mcp started")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    retrieval_plan = state.get("retrieval_plan") or {}
    if state.get("_no_web_override") or is_eval_enabled("EVAL_DISABLE_WEB") or not retrieval_plan.get("use_web", False):
        logger.info("Web retrieval skipped by retrieval plan")
        add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
        return {"web_search_docs": []}

    query = state.get("rewritten_query", "")
    if not query:
        query = state.get("original_query", "")
        
    docs = []
    
    if query:
        try:
            logger.info(f"MCP WebSearch request: query={query}")
            result = asyncio.run(mcp_call(query))
            
            if result and not result.isError and result.content:
                raw_text = result.content[0].text
                try:
                    data = json.loads(raw_text)
                    pages = data.get("pages") or []
                    
                    logger.debug(f"MCP WebSearch pages received: {len(pages)}")
                    
                    for item in pages:
                        snippet = (item.get("snippet") or "").strip()
                        url = (item.get("url") or "").strip()
                        title = (item.get("title") or "").strip()
                        
                        if not snippet:
                            continue
                            
                        docs.append({"title": title, "url": url, "snippet": snippet})
                        
                except json.JSONDecodeError:
                    logger.error(f"MCP WebSearch response is not valid JSON: {raw_text[:100]}...")
            else:
                if result and result.isError:
                    logger.error(f"MCP WebSearch returned an error: {result}")
                else:
                    logger.warning("MCP WebSearch returned no usable content")

            logger.info(f"MCP WebSearch normalized results: {len(docs)}")
            
        except Exception as e:
            logger.error(f"MCP WebSearch node failed: {e}", exc_info=True)
    else:
        logger.warning("MCP WebSearch skipped because query is empty")

    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    logger.info(f"node_web_search_mcp completed: results={len(docs)}")

    if docs:
        return {"web_search_docs": docs}
    return {}
