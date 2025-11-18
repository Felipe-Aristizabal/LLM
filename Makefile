# ===== Makefile para LLM Tecnoquímicas =====
# Requiere: uv (https://docs.astral.sh/uv/), Git Bash/WSL (o usa los comandos uv equivalentes en PowerShell)
# Variables (puedes sobreescribir en la línea de comandos, p. ej.: make scrape-headless MAX_PAGES=50)
LINKS ?= links.txt
OUT_TMP ?= src/tecnoquimicas_kb/data/tmp/
CLEAN_DIR ?= src/tecnoquimicas_kb/data/clean
RAW_DIR ?= src/tecnoquimicas_kb/data/raw
INDEX_DIR ?= src/tecnoquimicas_kb/index/faiss
MAX_PAGES ?= 25           # Páginas por dominio
MAX_TOTAL_PAGES ?= 200    # Límite global de páginas
EVAL_OUT ?= qa_eval_results.jsonl

.PHONY: init scrape-headless scrape-headful data-organize qa-app \
        build-index eval-qa clean-tmp ollama-pull

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

# Construir / reconstruir el índice FAISS a partir de CLEAN_DIR
build-index:
	uv run python -m tecnoquimicas_kb.app.cli.index_cli \
		--data-dir $(CLEAN_DIR) \
		--index-dir $(INDEX_DIR)

# Ejecutar la app de QA full-context (modo stuffing, sin FAISS)
app:
	uv run streamlit run src/tecnoquimicas_kb/app/app.py

# Correr evaluación de QA usando el dataset por defecto (o el que definas en el CLI)
eval-qa:
	uv run python -m tecnoquimicas_kb.apps.cli.eval_cli \
		--output-file $(EVAL_OUT)

clean-tmp:
	uv run python -c "import shutil,sys; shutil.rmtree('$(OUT_TMP)', ignore_errors=True)"

# Modelos locales (opcional)
ollama-pull:
	gpt-oss:20b
	ollama pull gemma3:4b
	ollama pull gemma3:270m
