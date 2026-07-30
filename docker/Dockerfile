# Obraz produkcyjny API. Multi-stage: warstwa budująca ma uv i nagłówki
# kompilacji, warstwa końcowa tylko środowisko uruchomieniowe.
#
# Model embeddingów (~1.1 GB) NIE jest wbudowany w obraz. Pobiera się przy
# pierwszym starcie do wolumenu wskazanego przez HF_HOME. Powód: obraz z wagami
# to ~2 GB do wypchnięcia przy każdej zmianie jednej linijki kodu, a wagi
# zmieniają się raz na kilka miesięcy. Wolumen przeżywa restart poda, więc
# koszt pobrania płaci się raz, nie przy każdym deployu.

FROM python:3.11-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

WORKDIR /app

# Zależności instalowane PRZED skopiowaniem kodu — warstwa cache'uje się
# i nie unieważnia przy każdej zmianie w src/. --no-dev pomija ruff, mypy,
# pytest i resztę narzędzi deweloperskich, których produkcja nie uruchamia.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src/ ./src/
COPY README.md ./
RUN uv sync --frozen --no-dev


FROM python:3.11-slim AS runtime

# Użytkownik bez uprawnień roota. Wymagane przez restrykcyjne polityki
# PodSecurity, a poza tym proces czytający PDF-y i wołający sieć nie ma
# powodu działać jako root.
RUN groupadd --system --gid 1001 tsl \
    && useradd --system --uid 1001 --gid tsl --create-home tsl

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Katalog na wagi modelu embeddingów — montowany jako wolumen.
    HF_HOME=/home/tsl/.cache/huggingface

WORKDIR /app

COPY --from=builder --chown=tsl:tsl /app/.venv /app/.venv
COPY --from=builder --chown=tsl:tsl /app/src /app/src

# Bramka promocji jedzie w TYM SAMYM obrazie co API — nie w osobnym obrazie
# ewaluacyjnym. Drugi obraz to drugi tag do zsynchronizowania, czyli druga okazja
# do rozjazdu między bramką a kodem, który ona ocenia. Kilkadziesiąt kB (wyniki
# przebiegów wycina .dockerignore) nie jest ceną wartą tego ryzyka.
#
# Kopiowane z kontekstu budowania, nie z warstwy builder: `uv sync` instaluje
# wyłącznie pakiet tsl_rag ze src/, a evals/ nie jest pakietem instalowanym.
# Dlatego `python -m evals.run_retrieval_evals` znajduje go WYŁĄCZNIE przez cwd
# i Job bramki musi mieć workingDir /app.
COPY --chown=tsl:tsl evals/ /app/evals/

USER tsl

EXPOSE 8000

# Probe wskazuje /health (liveness), nie /ready: /ready odpytuje bazę
# i provider embeddingów, więc jako HEALTHCHECK Dockera restartowałby
# kontener przy chwilowo niedostępnej bazie. Readiness należy do orkiestratora.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status==200 else 1)"

CMD ["uvicorn", "tsl_rag.api.app:create_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8000"]
