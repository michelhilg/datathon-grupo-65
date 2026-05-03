"""Stage: (re)constrói índice ChromaDB a partir da knowledge base.

Disparado pelo DVC sempre que data/knowledge_base/ ou params.rag mudam,
garantindo consistência entre a KB e o índice de vetores.
"""
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parents[1]))
from src.agent.rag_pipeline import build_index


def main() -> None:
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    rag = params["rag"]
    collection = build_index(
        knowledge_base_dir=Path(rag["knowledge_base_dir"]),
        persist_dir=Path(rag["persist_dir"]),
        model_name=rag["model_name"],
        chunk_size=rag["chunk_size"],
        chunk_overlap=rag["chunk_overlap"],
        force_rebuild=True,
    )
    print(f"Índice construído: {collection.count()} chunks em '{rag['persist_dir']}'")
    print(f"  knowledge_base_dir : {rag['knowledge_base_dir']}")
    print(f"  embedding_model    : {rag['model_name']}")
    print(f"  chunk_size         : {rag['chunk_size']} palavras (overlap={rag['chunk_overlap']})")


if __name__ == "__main__":
    main()
