# ===== Makefile para LLM Tecnoquímicas =====
# Requiere: uv (https://docs.astral.sh/uv/), Git Bash/WSL (o usa los comandos uv equivalentes en PowerShell)
# Variables (puedes sobreescribir en la línea de comandos, p. ej.: make scrape-headless MAX_PAGES=50)
LINKS ?= links.txt
OUT_TMP ?= src/tecnoquimicas_kb/data/tmp/
CLEAN_DIR ?= src/tecnoquimicas_kb/data/clean
RAW_DIR ?= src/tecnoquimicas_kb/data/raw
MAX_PAGES ?= 25           # Páginas por dominio
MAX_TOTAL_PAGES ?= 200    # Límite global de páginas
EVAL_OUT ?= qa_eval_results.jsonl

.PHONY: init ollama-pull scrape-headless data-organize \
        build-index app clean-tmp 

init:
	uv sync

# Web scraping sin ver navegador 
scrape-headless:
	uv run python -m tecnoquimicas_kb.apps.cli.scrape_cli \
		--links $(LINKS) \
		--out $(OUT_TMP) \
		--crawl \
		--max-pages-per-domain $(MAX_PAGES) \
		--max-total-pages $(MAX_TOTAL_PAGES)

# Web scraping "headful" 
scrape-headful:
	uv run python -m tecnoquimicas_kb.apps.cli.scrape_cli \
		--links $(LINKS) \
		--out $(OUT_TMP) \
		--crawl \
		--max-pages-per-domain $(MAX_PAGES) \
		--max-total-pages $(MAX_TOTAL_PAGES) \
		--headful

# Reubicar salidas del scraping:
#   - OUT_TMP/raw_html   -> RAW_DIR/raw_html
#   - OUT_TMP/clean_text -> CLEAN_DIR/clean_text
#   - OUT_TMP/chunks     -> CLEAN_DIR/chunks
data-organize:
	uv run python scripts/organize_data_dirs.py \
		--in $(OUT_TMP) \
		--clean-dir $(CLEAN_DIR) \
		--raw-dir $(RAW_DIR)

# Construir índice FAISS
build-index:
	set PYTHONPATH=src && uv run python -m tecnoquimicas_kb.app.cli.build_faiss

# Ejecutar la app 
app:
	set PYTHONPATH=src && uv run streamlit run src/tecnoquimicas_kb/app/app.py

# Ejecutar API FastAPI
api:
	set PYTHONPATH=src && uv run uvicorn tecnoquimicas_kb.api.server:app --reload --host 0.0.0.0 --port 8000

# Ejecutar API FastAPI y ngrok juntos
api-ngrok:
	(set PYTHONPATH=src && uv run uvicorn tecnoquimicas_kb.api.server:app --reload --host 0.0.0.0 --port 8000 &) && sleep 5 && ngrok http 8000

clean-tmp:
	uv run python -c "import shutil,sys; shutil.rmtree('$(OUT_TMP)', ignore_errors=True)"

# Modelos locales (opcional)
ollama-pull:
	ollama pull gpt-oss:20b
	ollama pull gemma3:4b
	ollama pull gemma3:270m
