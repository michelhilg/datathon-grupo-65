"""Handlers de telemetria LangChain — captura de contextos RAG e integração Langfuse."""
import logging
import os
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger(__name__)

# Separador usado por rag_pipeline.retrieve() entre chunks
_CHUNK_SEPARATOR = "\n\n---\n\n"


class ContextAccumulatorHandler(BaseCallbackHandler):
    """Captura os chunks retornados pela tool retention_knowledge por requisição.

    Instanciar um handler por request e passar via RunnableConfig(callbacks=[...]).
    """

    def __init__(self) -> None:
        super().__init__()
        self.captured_contexts: list[str] = []

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,  # noqa: ARG002
        parent_run_id: UUID | None = None,  # noqa: ARG002
        **kwargs: Any,
    ) -> None:
        output_str = str(output) if not isinstance(output, str) else output
        chunks = [c.strip() for c in output_str.split(_CHUNK_SEPARATOR) if c.strip()]
        if chunks:
            self.captured_contexts.extend(chunks)


def get_langfuse_handler() -> Any:
    """Retorna uma nova instância do CallbackHandler oficial do Langfuse.
    
    Por request, instancia o handler oficial da biblioteca que interceptará
    automaticamente os eventos do LangChain (tools, agents, chains).
    """
    pk = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    sk = os.getenv("LANGFUSE_SECRET_KEY", "")
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    
    # Validação simples para evitar que a inicialização levante erro se as chaves forem ausentes/inválidas
    if not pk or not sk or pk.startswith("your-") or sk.startswith("your-"):
        logger.info("Langfuse credentials ausentes ou inválidas — telemetria desabilitada.")
        return None
        
    try:
        from langfuse import Langfuse  # noqa: PLC0415
        from langfuse.langchain import CallbackHandler  # noqa: PLC0415
        
        Langfuse(public_key=pk, secret_key=sk, host=host)
        handler = CallbackHandler()
        logger.info("Langfuse telemetria habilitada com sucesso para esta requisição.")
        return handler
    except ImportError:
        logger.info("langfuse não instalado — telemetria LLM desabilitada")
        return None
    except Exception as e:
        logger.warning("Erro ao inicializar Langfuse CallbackHandler: %s", e)
        return None
