"""Smoke test for the document-ingestion workflow."""

import json

from app.import_process.agent.main_graph import get_kb_import_app
from app.import_process.agent.state import create_default_state
from app.core.logger import logger

logger.info("Ingestion graph smoke test started")

initial_state = create_default_state(local_file_path="万用表RS-12的使用.pdf")
final_state = None

kb_import_app = get_kb_import_app()

for event in kb_import_app.stream(initial_state):
    for key, value in event.items():
        logger.info(f"Completed stage: {key}")
        final_state = value

logger.info(f"Final state: {json.dumps(final_state, indent=4, ensure_ascii=False)}")

logger.info("Workflow graph:")
kb_import_app.get_graph().print_ascii()

logger.info("Ingestion graph smoke test completed")

