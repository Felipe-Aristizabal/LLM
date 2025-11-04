from typing import Dict, Any, Callable
from tecnoquimicas_kb.agent.memory import InMemorySessionStore
from tecnoquimicas_kb.agent.tools import get_structured_answer
from tecnoquimicas_kb.agent.prompts import AGENT_SYSTEM_PROMPT, DOCS_QA_INSTRUCTION
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configuración del LLM
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "gemini")

def _get_llm():
    """Retorna la instancia del LLM según el provider configurado."""
    if MODEL_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        model_id = os.getenv("GEN_MODEL_ID", "gemini-2.5-pro")
        api_key = os.getenv("GOOGLE_API_KEY")
        return ChatGoogleGenerativeAI(model=model_id, temperature=0.2, google_api_key=api_key)
    else:  # ollama
        from langchain_ollama import ChatOllama
        model_id = os.getenv("OLLAMA_MODEL_ID", "gemma3:4b")
        return ChatOllama(model=model_id, temperature=0.2)

def agent_dispatch(
    session_store: InMemorySessionStore,
    session_id: str,
    user_msg: str,
    docs_context_builder: Callable[[str], str]
) -> Dict[str, Any]:
    """
    Router principal del agente:
    1) Intenta obtener respuesta de datos estructurados (TOOL_STRUCT)
    2) Si no hay match, usa TOOL_DOCS (stuffing + LLM)
    3) Mantiene memoria de la sesión
    
    Returns:
        dict con keys: 'answer', 'route', 'thought'
    """
    # Paso 1: Intentar respuesta estructurada
    structured_answer = get_structured_answer(user_msg)
    
    if structured_answer:
        return {
            "answer": structured_answer,
            "route": "TOOL_STRUCT",
            "thought": "Pregunta sobre datos estructurados deterministas."
        }
    
    # Paso 2: No hay match en datos estructurados → usar TOOL_DOCS (RAG/Stuffing)
    llm = _get_llm()
    
    # Construir contexto documental
    docs_context = docs_context_builder(user_msg)
    
    # Obtener historial de la sesión
    history = session_store.get_history(session_id)
    
    # Construir mensajes para el LLM
    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT}
    ]
    
    # Agregar historial previo (últimos N mensajes para no saturar)
    max_history = 10
    for turn in history[-max_history:]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    
    # Agregar contexto documental e instrucción
    messages.append({
        "role": "system",
        "content": f"{DOCS_QA_INSTRUCTION}\n\nCONTEXTO DOCUMENTAL:\n{docs_context}"
    })
    
    # Invocar el LLM
    try:
        response = llm.invoke(messages)
        answer = response.content if hasattr(response, 'content') else str(response)
    except Exception as e:
        answer = f"Error al procesar la consulta: {str(e)}"
    
    return {
        "answer": answer,
        "route": "TOOL_DOCS",
        "thought": "Consulta documental usando stuffing + LLM con memoria de sesión."
    }