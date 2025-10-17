# LLM Tecnoquímicas — Base de Conocimiento Semántico y Sistema Q&A (Módulo 1)

> **Objetivo:** Construir el núcleo de conocimiento (knowledge base) de la empresa **Tecnoquímicas** para alimentar un asistente virtual. El proyecto incluye: extracción de información pública (**web scraping**), limpieza y **chunking** orquestrado mediante **LangChain** y visualizado en una **UI** basada en **Streamlit** para demostrar el **Q&A** básico empleando **Prompt Engineering** donde el texto limpio obtenido en el Punto 2 se consolida en el prompt como memoria del asistente.

---

## Tabla de contenidos
- [Arquitectura (Stuffing)](#arquitectura-stuffing)
- [Estructura de datos requerida](#estructura-de-datos-requerida)
- [Variables de entorno (`.env`)](#variables-de-entorno-env)
- [Instalación y entorno](#instalación-y-entorno)
- [Web scraping (headless)](#web-scraping-headless)
- [Web scraping (headful)](#web-scraping-headful)
- [Organizar datos a la estructura canónica](#organizar-datos-a-la-estructura-canónica)
- [Ejecutar la app de prueba (Stuffing)](#ejecutar-la-app-de-prueba-stuffing)
- [Cuotas de Gemini y consejo práctico](#cuotas-de-gemini-y-consejo-práctico)

---

## Arquitectura (Stuffing)

```
[data/links.txt] -> [scraper -> raw_html / clean_text / chunks] -> [organizar a estructura canónica]
                                                                              │
                                                                              ▼
                                                        [stuffing: contexto construido desde archivos]
                                                                              │
                                                                              ▼
                                                [Prompts con guardrails (Resumen | FAQ | Q&A)]
                                                                              │
                                        ┌───────────────┬─────────────────────┘
                                        ▼               ▼
                               [Gemini 2.5 Pro]   [Ollama: gemma3:4b]
```

- **Targeted stuffing**: se arma el contexto con **k archivos más relevantes** (por coincidencia de términos), para reducir tokens.  
- **Guardrails + Prompt Engineering**: si no hay evidencia en el contexto, el asistente **debe** decirlo.

---

## Estructura de datos requerida

Estructura **base** donde debe quedar la salida del web scraping :

```
src/tecnoquimicas_kb/data/
  raw/
    raw_html/           # HTML crudo
  clean/
    clean_text/         # .txt limpios
    chunks/             # .jsonl/.json con metadatos (text, url, section, date, chunk_id)
```

> La salida del web scraping  esta ubicada en otra ruta (p. ej. `src/tecnoquimicas_kb/data/tmp/out/`), usa el paso de **Organizar datos** para mover a la estructura base.

---

## Instalación y entorno

1) Instala dependencias (con `uv`):  
```bash
uv sync
```

2) Crea `.env` a partir de `.env.example` y completa (especialmente `GOOGLE_API_KEY` si usas Gemini):  
```bash
cp .env.example .env
```

---

## Variables de entorno (`.env`)

Ejemplo (ver también `.env.example`):

```dotenv
# === LLM Provider ===
MODEL_PROVIDER=gemini          # "gemini" | "ollama"
GEN_MODEL_ID=gemini-2.5-pro
OLLAMA_MODEL_ID=gemma3:4b

# === Stuffing (límites y top-k) ===
MAX_CONTEXT_CHARS_ALL=60000    # Resumen/FAQ
MAX_CONTEXT_CHARS_QA=40000     # Q&A
DEFAULT_K_FILES_SUM=12         # archivos para Resumen
DEFAULT_K_FILES_FAQ=15         # archivos para FAQ  (¡ojo con el nombre!)
DEFAULT_K_FILES_QA=15          # archivos para Q&A

# === Gemini (si usas provider=gemini) ===
GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY
```
---
## Web scraping (headless)

Con **Makefile** (recomendado):

```bash
make init
make scrape-headless              # usa data/links.txt y deja la salida en src/tecnoquimicas_kb/data/tmp/out
```

Comando **equivalente (make scrape-headless)** sin supervision :
```bash
uv run tq-kb --links data/links.txt --out src/tecnoquimicas_kb/data/tmp/out --crawl --max-pages 25
```
## Web scraping (headful)
Comando **equivalente (scrape-headful)** con supervision:
```bash
uv run tq-kb --links data/links.txt --out src/tecnoquimicas_kb/data/tmp/out --crawl --max-pages 25 --headful
```

> Ajusta páginas con `MAX_PAGES=50`:
```bash
make scrape-headless MAX_PAGES=50
```


---

## Organizar datos a la estructura canónica

Después del scraping, mueve todo a las carpetas canónicas:

```bash
make data-organize
```

Equivalente sin make:
```bash
uv run python scripts/organize_data.py \
  --in src/tecnoquimicas_kb/data/tmp/out \
  --clean-dir src/tecnoquimicas_kb/data/clean \
  --raw-dir src/tecnoquimicas_kb/data/raw
```

Esto deja:
```
src/tecnoquimicas_kb/data/
  raw/raw_html/
  clean/clean_text/
  clean/chunks/
```


---

## Ejecutar la app de prueba (Stuffing)

```bash
make stuffing-app
# equivalente:
# uv run streamlit run src/tecnoquimicas_kb/app/app_stuffing.py
```

En la **sidebar**:
- Cambia `Provider` **gemini** ↔ **ollama**.
- Ajusta **Top archivos** y **límites de contexto** si te topas la cuota de Gemini.  
- Para local: `ollama pull gemma3:4b` (mejor que `270m` u otro modelo de Ollama).

---

## Cuotas de Gemini y consejo práctico

- El **free tier** limita los **tokens de entrada por minuto** (~125k). Si concatenas demasiado texto o das clics seguidos, puedes recibir `ResourceExhausted (429)`.
- **Mitigación**: usa “targeted stuffing” (ya viene en el código), baja los límites en la sidebar, y considera `MODEL_PROVIDER=ollama` mientras se restablece la cuota.


---
