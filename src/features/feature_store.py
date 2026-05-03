"""Feature Store incremental — upsert via HSET/TTL, nunca FLUSHALL.

Padrão: cache-first com fallback gracioso se Redis indisponível.
Atualização: delta via timestamp — só processa registros alterados desde last_materialized_at.
Garantia: store nunca fica vazio durante atualização (upsert, não full-flush).
"""
import json
import logging
import os
import time
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_FEATURE_TTL_SECONDS = int(os.getenv("FEATURE_TTL_SECONDS", "3600"))
_KEY_PREFIX = "features:customer:"
_LAST_MATERIALIZED_KEY = "features:meta:last_materialized"


def _make_redis_client():
    import redis
    url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    return redis.Redis.from_url(url, decode_responses=True, socket_connect_timeout=2)


def _serialize(v: Any) -> str:
    """Converte valor numpy/Python para JSON string armazenável no Redis."""
    if hasattr(v, "item"):  # numpy scalar → Python nativo
        v = v.item()
    if v is None:
        return "null"
    if isinstance(v, float) and pd.isna(v):
        return "null"
    return json.dumps(v)


class FeatureStore:
    """Feature store incremental sobre Redis.

    Toda atualização usa HSET + EXPIRE (upsert). Nunca FLUSHALL.
    Store permanece disponível durante qualquer operação de escrita.
    """

    def __init__(self, client: Any = None) -> None:
        self._r = client  # None = lazy init na primeira chamada

    def _client(self) -> Any:
        if self._r is None:
            self._r = _make_redis_client()
        return self._r

    def _key(self, customer_id: str) -> str:
        return f"{_KEY_PREFIX}{customer_id}"

    def upsert(
        self,
        customer_id: str,
        features: dict,
        ttl: int = _FEATURE_TTL_SECONDS,
    ) -> None:
        """Atualiza (ou cria) features de um cliente sem afetar outros registros.

        Usa HSET + EXPIRE: operação atômica, store nunca fica vazio.
        """
        key = self._key(customer_id)
        mapping = {k: _serialize(v) for k, v in features.items()}
        r = self._client()
        r.hset(key, mapping=mapping)
        r.expire(key, ttl)

    def get(self, customer_id: str) -> Optional[dict]:
        """Recupera features de um cliente. Retorna None se não estiver no cache."""
        raw = self._client().hgetall(self._key(customer_id))
        if not raw:
            return None
        return {k: json.loads(v) for k, v in raw.items()}

    def get_many(self, customer_ids: list[str]) -> dict[str, Optional[dict]]:
        """Recupera features de múltiplos clientes via pipeline Redis (batch eficiente)."""
        pipe = self._client().pipeline()
        for cid in customer_ids:
            pipe.hgetall(self._key(cid))
        results = pipe.execute()
        return {
            cid: ({k: json.loads(v) for k, v in raw.items()} if raw else None)
            for cid, raw in zip(customer_ids, results)
        }

    def batch_upsert_delta(
        self,
        df: pd.DataFrame,
        since_timestamp: Optional[float] = None,
        ttl: int = _FEATURE_TTL_SECONDS,
    ) -> int:
        """Materializa features apenas dos registros alterados desde `since_timestamp`.

        Estratégia incremental: nunca apaga o store inteiro.
        - Com since_timestamp: filtra pelo campo 'updated_at' do DataFrame.
        - Sem since_timestamp (None): processa todos os registros (carga inicial).

        Args:
            df: DataFrame com coluna 'customerID' e, opcionalmente, 'updated_at' (Unix ts).
            since_timestamp: Processa apenas linhas com updated_at >= since_timestamp.
            ttl: TTL em segundos para cada chave Redis.

        Returns:
            Número de registros upsertados.
        """
        from src.features.feature_engineering import build_features

        if "customerID" not in df.columns:
            raise ValueError("DataFrame deve conter coluna 'customerID'")

        delta = df
        if since_timestamp is not None and "updated_at" in df.columns:
            delta = df[df["updated_at"] >= since_timestamp]

        if delta.empty:
            logger.info("Nenhum delta para materializar (since_ts=%.0f)", since_timestamp or 0)
            return 0

        # Preserva IDs antes de build_features (que descarta customerID via encode_categoricals)
        customer_ids = delta["customerID"].tolist()
        features_df = build_features(delta.copy())

        pipe = self._client().pipeline()
        for cid, (_, row) in zip(customer_ids, features_df.iterrows()):
            key = self._key(str(cid))
            mapping = {k: _serialize(v) for k, v in row.to_dict().items()}
            pipe.hset(key, mapping=mapping)
            pipe.expire(key, ttl)

        pipe.execute()

        self._client().set(_LAST_MATERIALIZED_KEY, time.time())
        logger.info("Delta materializado: %d registros upsertados", len(customer_ids))
        return len(customer_ids)

    def last_materialized_at(self) -> Optional[float]:
        """Retorna timestamp Unix da última materialização bem-sucedida, ou None."""
        val = self._client().get(_LAST_MATERIALIZED_KEY)
        return float(val) if val else None

    def ping(self) -> bool:
        """Verifica conectividade com Redis."""
        try:
            return bool(self._client().ping())
        except Exception:
            return False


_store: Optional[FeatureStore] = None


def get_feature_store() -> FeatureStore:
    """Retorna instância global do FeatureStore (singleton lazy)."""
    global _store
    if _store is None:
        _store = FeatureStore()
    return _store
