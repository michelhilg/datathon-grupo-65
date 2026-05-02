#!/bin/sh
set -e

# Se MODEL_PATH está definido e o modelo ainda não foi exportado, exporta agora.
# Funciona porque mlruns/ e mlflow.db já estão montados como volumes.
if [ -n "$MODEL_PATH" ] && [ ! -f "$MODEL_PATH/MLmodel" ]; then
    echo "[entrypoint] Modelo não encontrado em $MODEL_PATH. Exportando do MLflow..."
    uv run python scripts/export_model.py
    echo "[entrypoint] Modelo pronto."
fi

exec "$@"
