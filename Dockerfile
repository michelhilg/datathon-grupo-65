# Usa uma imagem oficial leve do Python
FROM python:3.11-slim

# Evita que o Python gere arquivos .pyc e força o output a ir direto para o terminal
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Instala ferramentas básicas, dependências de sistema (se necessário) e o uv
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Instala o uv (gerenciador de pacotes ultrarrápido)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Configura o diretório de trabalho
WORKDIR /app

# Copia os arquivos de configuração de dependências primeiro
COPY pyproject.toml uv.lock ./

# Instala as dependências usando o uv (sem instalar dependências de desenvolvimento)
RUN uv sync --no-dev

# Copia todo o restante do projeto (código, pipeline do dvc, etc)
COPY . .

# Expõe a porta do mlflow caso queiramos testar a UI pelo container
EXPOSE 5000

# Comando default: rodar o pipeline através do DVC
CMD ["uv", "run", "dvc", "repro"]
