"""Guardrails de segurança para input e output do agente.

Referência: OWASP Top 10 for LLM Applications (2025)
            https://owasp.org/www-project-top-10-for-large-language-model-applications/

Nota: OutputGuardrail requer modelo spacy instalado:
      python -m spacy download en_core_web_lg
"""
import logging
import re

logger = logging.getLogger(__name__)

# Importação em nível de módulo para permitir mock nos testes
try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    _PRESIDIO_AVAILABLE = True
except ImportError:
    _PRESIDIO_AVAILABLE = False


class InputGuardrail:
    """Valida input do usuário antes de enviar ao LLM (LLM01 — Prompt Injection)."""

    INJECTION_PATTERNS = [
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"you\s+are\s+now\s+a",
        r"system:\s*",
        r"<\|im_start\|>",
        r"\[INST\]",
        r"forget\s+(everything|all|your\s+instructions)",
    ]

    # LLM06 — Sensitive Information Disclosure: bloqueia tentativas de extração de PII
    PII_EXTRACTION_PATTERNS = [
        r"\bcpf\b",
        r"\bcnpj\b",
        r"\brg\b",
        r"\bpassport\b",
        r"\bpassaporte\b",
        r"(me\s+d[iê]|inform[ae]|mostre?|revele?|liste?|exib[ae])\s+.{0,30}(cpf|cnpj|rg|senha|password|credit.?card|cartão)",
        r"(qual|o\s+que|me\s+d[iê]).{0,20}(senha|password|token|secret|chave\s+pix)",
    ]

    MAX_INPUT_LENGTH = 4096

    def __init__(self, allowed_topics: list[str] | None = None):
        self.allowed_topics = allowed_topics or []
        self._compiled = [re.compile(p, re.IGNORECASE) for p in self.INJECTION_PATTERNS]
        self._pii_compiled = [re.compile(p, re.IGNORECASE) for p in self.PII_EXTRACTION_PATTERNS]

    def validate(self, user_input: str) -> tuple[bool, str]:
        """Valida input do usuário.

        Args:
            user_input: Texto livre enviado pelo usuário.

        Returns:
            Tupla (is_valid, reason).
        """
        for pattern in self._compiled:
            if pattern.search(user_input):
                logger.warning("Prompt injection detectado: %.100s", user_input)
                return False, "Input bloqueado: padrão suspeito detectado."

        for pattern in self._pii_compiled:
            if pattern.search(user_input):
                logger.warning("Tentativa de extração de PII detectada: %.100s", user_input)
                return False, "Input bloqueado: solicitação de dados pessoais não permitida."

        if len(user_input) > self.MAX_INPUT_LENGTH:
            return False, f"Input bloqueado: excede tamanho máximo ({self.MAX_INPUT_LENGTH} chars)."

        if self.allowed_topics:
            lower = user_input.lower()
            if not any(topic in lower for topic in self.allowed_topics):
                logger.info("Input fora do escopo: %.100s", user_input)
                return False, "Input bloqueado: fora do escopo do assistente."

        return True, "OK"


class OutputGuardrail:
    """Sanitiza output do LLM antes de retornar ao usuário (LLM02 + LLM06)."""

    def __init__(self, language: str = "en"):
        self._available = False
        self.language = language
        if _PRESIDIO_AVAILABLE:
            try:
                self._analyzer = AnalyzerEngine()
                self._anonymizer = AnonymizerEngine()
                self._available = True
                logger.info("OutputGuardrail inicializado com Presidio.")
            except Exception as exc:
                logger.warning("Presidio não inicializável — output sanitization desativada: %s", exc)
        else:
            logger.warning("presidio-analyzer não instalado — output sanitization desativada.")

    def sanitize(self, llm_output: str) -> str:
        """Remove PII do output do LLM.

        Args:
            llm_output: Texto gerado pelo LLM.

        Returns:
            Texto sanitizado (ou original se Presidio indisponível).
        """
        if not self._available:
            return llm_output

        results = self._analyzer.analyze(
            text=llm_output,
            language=self.language,
            entities=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"],
        )

        if results:
            logger.warning("PII detectado no output: %d entidades", len(results))
            anonymized = self._anonymizer.anonymize(
                text=llm_output,
                analyzer_results=results,
            )
            return anonymized.text

        return llm_output
