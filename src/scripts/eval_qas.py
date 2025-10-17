# scripts/eval_qas.py
import json, time
from tecnoquimicas_kb.rag.chains import load_index, answer
from dotenv import load_dotenv
load_dotenv()

QUESTIONS = [
 "¿Quién es Tecnoquímicas y en qué sectores opera?",
 "¿Dónde puedo encontrar los puntos de contacto de TQ?",
 "¿Qué categorías de productos comercializa TQ?",
 "¿Tienen presencia en todo Colombia? ¿Cobertura?",
 "¿Cómo reporto un evento adverso de un medicamento?",
 "¿Qué programas de sostenibilidad o planeta maneja TQ?",
 "¿Cuál es la historia/resumen de TQ?",
 "¿Cómo encuentro el producto X en el catálogo?",
 "¿Qué marcas propias maneja TQ?",
 "¿Cómo postulo una PQRS?",
 "¿Cómo contactar ventas institucionales?",
 "¿Qué políticas de calidad declaran?",
 "¿Qué alianzas con hospitales o universidades tiene TQ?",
 "¿Dónde veo noticias/actualizaciones de TQ?",
 "¿Tienen FAQ oficiales para consumidores?",
 "¿Cómo ubico sedes o canales físicos?",
 "¿Qué dicen los medios sobre participación de mercado?",
 "¿Qué iniciativas de donación/RSI han realizado?",
 "¿Dónde encuentro la línea ética o canal de transparencia?",
 "¿Cómo solicitar información para prensa?"
]

if __name__ == "__main__":
    vs = load_index()
    out = []
    for q in QUESTIONS:
        ans, _ = answer(vs, q)
        out.append({"q": q, "a": ans, "ts": time.time()})
        print("OK:", q)
    with open("qa_eval_20.jsonl", "w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("Guardado qa_eval_20.jsonl")
