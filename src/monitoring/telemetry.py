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
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        output_str = str(output) if not isinstance(output, str) else output
        # retrieve() separa chunks com _CHUNK_SEPARATOR — sinal exclusivo de retention_knowledge
        if _CHUNK_SEPARATOR in output_str:
            chunks = [c.strip() for c in output_str.split(_CHUNK_SEPARATOR) if c.strip()]
            self.captured_contexts.extend(chunks)


class LangfuseCallbackHandler(BaseCallbackHandler):
    """Handler de telemetria Langfuse. No-op se LANGFUSE_PUBLIC_KEY não configurada ou pacote ausente."""

    def __init__(self) -> None:
        super().__init__()
        self._client = None
        self._trace = None
        self._enabled = False
        self._generations: dict[str, Any] = {}

        try:
            from langfuse import Langfuse  # noqa: PLC0415

            pk = os.getenv("LANGFUSE_PUBLIC_KEY", "")
            sk = os.getenv("LANGFUSE_SECRET_KEY", "")
            if pk and sk and not pk.startswith("your-"):
                self._client = Langfuse(
                    public_key=pk,
                    secret_key=sk,
                    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
                )
                self._enabled = True
                logger.info("Langfuse telemetria habilitada")
        except ImportError:
            logger.info("langfuse não instalado — telemetria LLM desabilitada")

    def on_chain_start(self, serialized: dict, inputs: dict, *, run_id: UUID, **kwargs: Any) -> None:
        if not self._enabled or self._client is None:
            return
        try:
            self._trace = self._client.trace(
                name="churn-agent-call",
                input=str(inputs)[:2000],
                metadata={"run_id": str(run_id)},
            )
        except Exception as exc:
            logger.debug("Langfuse on_chain_start error: %s", exc)

    def on_chat_model_start(
        self, serialized: dict, messages: list, *, run_id: UUID, **kwargs: Any
    ) -> None:
        if not self._enabled or self._trace is None:
            return
        try:
            gen = self._trace.generation(
                name="llm-call",
                input=str(messages)[:2000],
                model=serialized.get("kwargs", {}).get("model_name", "unknown"),
            )
            self._generations[str(run_id)] = gen
        except Exception as exc:
            logger.debug("Langfuse on_chat_model_start error: %s", exc)

    def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        if not self._enabled:
            return
        gen = self._generations.pop(str(run_id), None)
        if gen is None:
            return
        try:
            output_text = ""
            if hasattr(response, "generations") and response.generations:
                output_text = response.generations[0][0].text if response.generations[0] else ""
            gen.end(output=output_text[:2000])
        except Exception as exc:
            logger.debug("Langfuse on_llm_end error: %s", exc)

    def on_tool_start(
        self, serialized: dict, input_str: str, *, run_id: UUID, **kwargs: Any
    ) -> None:
        if not self._enabled or self._trace is None:
            return
        try:
            tool_name = serialized.get("name", "unknown_tool")
            self._trace.span(
                name=f"tool:{tool_name}",
                input=input_str[:500],
            )
        except Exception as exc:
            logger.debug("Langfuse on_tool_start error: %s", exc)

    def on_chain_end(self, outputs: dict, *, run_id: UUID, **kwargs: Any) -> None:
        if not self._enabled or self._client is None:
            return
        try:
            self._client.flush()
        except Exception as exc:
            logger.debug("Langfuse flush error: %s", exc)


def get_langfuse_handler() -> LangfuseCallbackHandler:
    """Retorna uma nova instância de LangfuseCallbackHandler por request."""
    return LangfuseCallbackHandler()
