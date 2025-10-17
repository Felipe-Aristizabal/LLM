# LLM Tecnoquímicas — Base de Conocimiento Semántico y Sistema Q&A (Módulo 1)

> **Objetivo:** Construir el núcleo de conocimiento (knowledge base) de la empresa **Tecnoquímicas** para alimentar un asistente virtual. El proyecto incluye: extracción de información pública, limpieza y **chunking** con **LangChain**, **prompts** y una **UI** base en **Streamlit** para demostrar el **Q&A** básico basado en **RAG/Prompt Engineering**.

---

## Tabla de contenidos
- [Arquitectura (Módulo 1)](#arquitectura-módulo-1)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Variables de entorno (`.env`)](#variables-de-entorno-env)
- [Datos de entrada (links y chunks)](#datos-de-entrada-links-y-chunks)
- [Construcción del índice (FAISS)](#construcción-del-índice-faiss)
- [Ejecución de la app (Streamlit)](#ejecución-de-la-app-streamlit)
- [Conmutar de modelo (Gemini ↔ Ollama)](#conmutar-de-modelo-gemini--ollama)
- [Tareas implementadas (rúbrica)](#tareas-implementadas-rúbrica)
- [Evaluación rápida (20 preguntas)](#evaluación-rápida-20-preguntas)
- [Buenas prácticas, ética y reproducibilidad](#buenas-prácticas-ética-y-reproducibilidad)
- [Enlaces útiles y referencias](#enlaces-útiles-y-referencias)
- [Licencia de datos y descargo](#licencia-de-datos-y-descargo)

