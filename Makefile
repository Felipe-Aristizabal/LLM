# ===== Makefile para LLM Tecnoquímicas =====
# Requiere: uv (https://docs.astral.sh/uv/), Git Bash/WSL (o usa los comandos uv equivalentes en PowerShell)
# Variables (puedes sobreescribir en la línea de comandos, p. ej.: make scrape-headless MAX_PAGES=50)
LINKS ?= links.txt
OUT_TMP ?= src/tecnoquimicas_kb/data/tmp/
CLEAN_DIR ?= src/tecnoquimicas_kb/data/clean
RAW_DIR ?= src/tecnoquimicas_kb/data/raw
MAX_PAGES ?= 25

.PHONY: init scrape-headless data-organize stuffing-app clean-tmp ollama-pull

init:
	uv sync

# Web scraping sin ver navegador
scrape-headless:
	uv run tq-kb --links $(LINKS) --out $(OUT_TMP) --crawl --max-pages $(MAX_PAGES)

# Web scraping viendo el navegador
scrape-headful:
	uv run tq-kb --links $(LINKS) --out $(OUT_TMP) --crawl --max-pages $(MAX_PAGES) --headful

# Reubicar salidas del scraping:
#   - OUT_TMP/raw_html   -> RAW_DIR/raw_html
#   - OUT_TMP/clean_text -> CLEAN_DIR/clean_text
#   - OUT_TMP/chunks     -> CLEAN_DIR/chunks
data-organize:
	uv run python src/scripts/organize_data.py --in $(OUT_TMP) --clean-dir $(CLEAN_DIR) --raw-dir $(RAW_DIR)

# Ejecutar la app (modo Stuffing, sin FAISS)
stuffing-app:
	uv run streamlit run src/tecnoquimicas_kb/app/app_stuffing.py

clean-tmp:
	uv run python -c "import shutil,sys; shutil.rmtree('$(OUT_TMP)', ignore_errors=True)"

# Modelos locales (opcional)
ollama-pull:
	ollama pull gemma3:4b
	ollama pull gemma3:270m

# Inicialización de el agente
agent-app:
	uv run --isolated streamlit run src/tecnoquimicas_kb/app/app_agent.py
