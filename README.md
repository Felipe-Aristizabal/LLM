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
                                                       [stuffing: contexto top-k desde archivos]
                                                                              │
                                                                              ▼
      ┌──────────────────────────────────────────────────────────────────────────────────────────┐
      │              Agente Conversacional  [Gemini 2.5 Pro]   [Ollama: gemma3:4b]               │
      │  (memoria por sesión + prompts con guardrails + enrutamiento estructurado/documental)    │
      └──────────────────────────────────────────────────────────────────────────────────────────┘
                                  │                                               │
                                  ▼                                               ▼
                          [TOOL_STRUCT]                                     [TOOL_DOCS (RAG)]
                      (catálogo determinista)                       (LLM + contexto documental)
                                  │                                               │
                                  ▼                                               ▼
                      Respuesta concreta (tel, NIT,                    Respuesta explicativa
                       horarios, correos, sedes…)                      basada en el contexto
                              
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

## Ejecutar el **Agente Q&A**

A continuación se detalla la forma recomendada para levantar la app, alternar proveedor (Gemini/Ollama) y verificar el estado del backend de modelo. 

```
make agent-app
```

## ¿Cómo decide el agente? (Routing)

El agente enruta cada consulta a la mejor fuente de verdad según su **intención**.

---

### 1) `TOOL_STRUCT` — Datos deterministas (catálogo)

**Cuándo**  
El usuario pide un **dato concreto**:
- Teléfono, NIT, horario, correo, sedes, sitio web, etc.

**Qué hace**  
1. **Normaliza** la consulta (minúsculas, sin acentos, limpia ruido).  
2. Aplica **reglas simples de intención** (keywords) → campo del **JSON estructurado** (cacheado por *mtime*).  
3. Devuelve respuesta **exacta** y **rápida**, **sin** intervención del LLM.

**Ventajas**  
- **Cero alucinaciones** (la fuente es determinista).  
- **Baja latencia** y **costo cero** (no llama al LLM).

**Ejemplos**  
- “¿Cuál es el teléfono de servicio al cliente?”  
- “Dime el NIT de la empresa.”  
- “Horarios de atención en Cali.”

> Si hay **match**, el agente **no** consulta el LLM y responde de inmediato.  
> Si **no** hay coincidencia o el campo está vacío ⇒ **pasa a `TOOL_DOCS`**.

---

### 2) `TOOL_DOCS` — Documental / RAG (Stuffing + LLM)

**Cuándo**  
Preguntas **explicativas** o de **contexto**:
- Historia, políticas, procesos, iniciativas, cultura, sostenibilidad, innovación, etc.

**Qué hace**  
1. Construye un **contexto compacto** (*targeted stuffing*) con los **k archivos más relevantes** (p. ej., `k=12–15`), a partir de `clean_text/chunks` con metadatos (texto, URL, sección, fecha, id de chunk).  
2. Inyecta **guardrails** en el prompt:  
   - “Si el contexto no es suficiente, dilo; **no inventes**.”  
   - “Resume y responde en **5–8 oraciones**.”  
3. Invoca el **LLM** (Gemini u Ollama) seleccionado en la **UI**.  
4. Integra **memoria por sesión** para **coherencia** en *follow-ups*.

**Ventajas**  
- Respuestas **contextuales** y **explicativas**.  
- **Control de alucinación** vía guardrails + contexto.

**Ejemplos**  
- “¿Qué iniciativas de sostenibilidad lidera TQ?”  
- “Explica la historia y valores de Tecnoquímicas.”  
- “¿Cómo se relaciona el área de innovación con las marcas X e Y?”

> Si el **contexto no respalda** la respuesta, el agente **lo declara** y sugiere **próximos pasos** (fuentes, áreas, enlaces).
