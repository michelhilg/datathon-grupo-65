"""Testes do Feature Store incremental — usa fakeredis para isolar Redis."""
import time

import fakeredis
import pandas as pd
import pytest

from src.features.feature_store import FeatureStore, _serialize


@pytest.fixture
def store():
    r = fakeredis.FakeRedis(decode_responses=True)
    return FeatureStore(client=r)


@pytest.fixture
def sample_features():
    return {"tenure": 12.0, "MonthlyCharges": 65.5, "avg_monthly_spend": 65.5, "services_count": 3}


# ── upsert + get ─────────────────────────────────────────────────────────────

def test_upsert_and_get_roundtrip(store, sample_features):
    store.upsert("CUST-001", sample_features)
    result = store.get("CUST-001")
    assert result is not None
    assert result["tenure"] == pytest.approx(12.0)
    assert result["MonthlyCharges"] == pytest.approx(65.5)


def test_get_miss_returns_none(store):
    assert store.get("CUST-NONEXISTENT") is None


def test_upsert_overwrites_existing(store, sample_features):
    store.upsert("CUST-001", sample_features)
    updated = {**sample_features, "tenure": 24.0}
    store.upsert("CUST-001", updated)
    result = store.get("CUST-001")
    assert result["tenure"] == pytest.approx(24.0)


def test_upsert_sets_ttl(store, sample_features):
    store.upsert("CUST-001", sample_features, ttl=300)
    ttl = store._client().ttl("features:customer:CUST-001")
    assert 0 < ttl <= 300


# ── get_many ──────────────────────────────────────────────────────────────────

def test_get_many_returns_all_hits(store, sample_features):
    store.upsert("CUST-001", sample_features)
    store.upsert("CUST-002", {**sample_features, "tenure": 36.0})
    results = store.get_many(["CUST-001", "CUST-002", "CUST-999"])
    assert results["CUST-001"] is not None
    assert results["CUST-002"]["tenure"] == pytest.approx(36.0)
    assert results["CUST-999"] is None


# ── batch_upsert_delta ────────────────────────────────────────────────────────

def test_batch_upsert_full_materialization(store, telco_raw):
    count = store.batch_upsert_delta(telco_raw)
    assert count == len(telco_raw)

    result = store.get("CUST-0000")
    assert result is not None
    assert "avg_monthly_spend" in result


def test_batch_upsert_delta_filters_by_timestamp(store, telco_raw):
    now = time.time()

    # Divide em dois grupos temporais: 8 antigos, 8 novos
    df = telco_raw.copy()
    df["updated_at"] = [now - 200] * 8 + [now] * 8

    count = store.batch_upsert_delta(df, since_timestamp=now - 10)
    assert count == 8  # apenas os 8 novos


def test_batch_upsert_delta_empty_returns_zero(store, telco_raw):
    future = time.time() + 9999
    df = telco_raw.copy()
    df["updated_at"] = time.time() - 100  # todos no passado

    count = store.batch_upsert_delta(df, since_timestamp=future)
    assert count == 0


def test_batch_upsert_raises_without_customer_id(store):
    df = pd.DataFrame([{"tenure": 12, "MonthlyCharges": 50}])
    with pytest.raises(ValueError, match="customerID"):
        store.batch_upsert_delta(df)


# ── garantia: store nunca fica vazio (anti-FLUSHALL) ─────────────────────────

def test_existing_keys_survive_batch_upsert(store, telco_raw, sample_features):
    # Insere uma chave "canário" antes da operação em lote
    store.upsert("CANARY-KEY", sample_features)

    store.batch_upsert_delta(telco_raw)

    # Canário deve sobreviver — sem FLUSHALL
    assert store.get("CANARY-KEY") is not None


# ── last_materialized_at ──────────────────────────────────────────────────────

def test_last_materialized_at_none_before_first_run(store):
    assert store.last_materialized_at() is None


def test_last_materialized_at_set_after_batch(store, telco_raw):
    before = time.time()
    store.batch_upsert_delta(telco_raw)
    after = time.time()

    ts = store.last_materialized_at()
    assert ts is not None
    assert before <= ts <= after


# ── ping ─────────────────────────────────────────────────────────────────────

def test_ping_returns_true(store):
    assert store.ping() is True


# ── _serialize helper ─────────────────────────────────────────────────────────

def test_serialize_numpy_types():
    import numpy as np
    assert _serialize(np.int64(42)) == "42"
    assert _serialize(np.float64(3.14)) == "3.14"
    assert _serialize(float("nan")) == "null"
    assert _serialize(None) == "null"
    assert _serialize("text") == '"text"'
    assert _serialize(True) == "true"
