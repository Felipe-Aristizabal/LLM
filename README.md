
# Entrega Final — Asistente Conversacional para Tecnoquímicas (Módulo 3)

> **Objetivo:** Desplegar un asistente conversacional robusto para Tecnoquímicas, integrando técnicas avanzadas de agentes (Function Calling), una API REST, y automatización vía WhatsApp/N8N. Incluye análisis de conversaciones y visualización de clústeres.

---

## Tabla de Contenidos
- [Resumen de la Entrega Final](#resumen-de-la-entrega-final)
- [Flujo de Ejecución](#flujo-de-ejecución)
- [Despliegue y Ejecución](#despliegue-y-ejecución)
- [API REST y Function Calling](#api-rest-y-function-calling)
- [Integración con WhatsApp y N8N](#integración-con-whatsapp-y-n8n)
- [Análisis de Conversaciones (t-SNE)](#análisis-de-conversaciones-t-sne)
- [Decisiones de Diseño](#decisiones-de-diseño)
- [Entrega Previa: Módulo 1 y 2](#entrega-previa-módulo-1-y-2)

---

## Resumen de la Entrega Final

Este repositorio implementa un asistente conversacional para Tecnoquímicas, capaz de interactuar con usuarios vía WhatsApp, utilizando técnicas avanzadas de agentes (Function Calling), memoria conversacional y herramientas externas. El sistema está expuesto como una API REST y automatizado mediante N8N para integración con WhatsApp Business API.



## Flujo de Ejecución

1. **Usuario** envía mensaje por WhatsApp.
2. **WhatsApp Business API** recibe y reenvía el mensaje a **N8N** (Webhook).
3. **N8N** procesa el mensaje y realiza una petición HTTP POST al endpoint `/chat` de la **API**.
4. **API REST (FastAPI)** gestiona la sesión, pasa el mensaje al **Agente**.
5. **Agente** utiliza Function Calling para decidir y ejecutar herramientas según el mensaje.
6. **Respuesta** del agente es devuelta a N8N y enviada al usuario por WhatsApp.
7. **Conversaciones** se almacenan para análisis posterior (t-SNE).

---

## Despliegue y Ejecución

1. Instala dependencias:
   ```bash
   uv sync
   ```
2. Configura variables en `.env` (ver ejemplo más abajo).
3. Levanta la API REST:
   ```bash
   uvicorn src/tecnoquimicas_kb/api/server.py --reload
   ```
4. Configura N8N y WhatsApp Business API (ver sección específica).
5. (Opcional) Ejecuta el análisis de conversaciones:
   ```bash
   python scripts/analyze_conversations.py
   ```

---

## API REST y Function Calling

- **Endpoint principal:** `/chat` (POST)
- **Payload:**
    ```json
    {
      "session_id": "<whatsapp_number>",
      "message": "<mensaje del usuario>"
    }
    ```
- **Respuesta:**
    ```json
    {
      "response": "<respuesta del agente>",
      "tool_used": "<herramienta>",
      "error": null
    }
    ```
- **Function Calling:**
    - El agente utiliza un esquema JSON estricto para invocar herramientas.
    - Manejo robusto de errores: respuestas amables si una herramienta falla.

---

## Integración con WhatsApp y N8N

- **N8N** actúa como orquestador:
    1. Recibe mensajes de WhatsApp vía Webhook.
    2. Llama a la API REST del agente.
    3. Envía la respuesta de vuelta al usuario.
- **Configuración básica:**
    - Importa el workflow de ejemplo (`n8n_workflow.json`).
    - Configura credenciales de WhatsApp y API.
    - Ajusta nodos según tus endpoints y lógica.

---

## Análisis de Conversaciones (t-SNE)

1. **Logging:** Cada conversación se almacena (JSON/DB).
2. **Embeddings:** Se generan vectores para cada conversación usando un modelo de embeddings.
3. **Visualización:** Se aplica t-SNE para proyectar los vectores y visualizar clústeres (ejemplo: quejas, consultas, fallos).
4. **Script/Notebook:** Incluido en `scripts/analyze_conversations.py` o notebook equivalente.

---

## Decisiones de Diseño

- **N8N vs Servidor Propio:** Se eligió N8N por su facilidad de integración visual y rápida iteración.
- **Function Calling:** Aumenta la fiabilidad y control sobre las herramientas usadas por el agente.
- **Gestión de Errores:** El agente responde de forma cortés ante fallos o falta de información.


## Entrega Previa: Módulo 1 y 2

<details>
<summary>Haz clic para ver la documentación de la entrega previa (Módulo 1 y 2)</summary>

# LLM Tecnoquímicas — Base de Conocimiento Semántico y Sistema Q&A (Módulo 1) RAG (Módulo 2)

> **Objetivo:** Construir el núcleo de conocimiento (knowledge base) de la empresa **Tecnoquímicas** para alimentar un asistente virtual. El proyecto incluye: extracción de información pública (**web scraping**), limpieza y **chunking** orquestrado mediante **LangChain** y visualizado en una **UI** basada en **Streamlit** para demostrar el **Q&A** básico empleando **Prompt Engineering** donde el texto limpio obtenido en el Punto 2 se consolida en el prompt como memoria del asistente.
</details>
