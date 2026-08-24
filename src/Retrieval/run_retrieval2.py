import json
import requests
from tqdm import tqdm

ANYTHINGLLM_API = "http://localhost:3001/api/v1/workspace/meu-workspace/chat"
API_KEY = "XV060WE-0G9M47P-GATDFKY-S2DC8D1"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

with open("runtime/rag_questions_EXATAS.json", "r", encoding="utf-8") as f:
    questions = json.load(f)

results = []

for item in tqdm(questions):

    payload = {
        "message": item["question"]
    }

    response = requests.post(ANYTHINGLLM_API, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()

    sources = data.get("sources", [])

    retrieval_list = []
    for s in sources:
        retrieval_list.append({
            "text": s["text"]
        })

    # Usa todos os gold_chunks (fatos individuais) quando disponíveis,
    # em vez de colapsar tudo no ground_truth único. Isso preserva o
    # número real de fatos relevantes para perguntas com múltiplos gold_chunks,
    # o que é essencial para o cálculo correto de MAP/nDCG/Hits.
    gold_chunks = item.get("gold_chunks")
    if gold_chunks:
        gold_list = [{"fact": g} for g in gold_chunks]
    else:
        gold_list = [{"fact": item["ground_truth"]}]

    results.append({
        "question_type": "normal",
        "retrieval_list": retrieval_list,
        "gold_list": gold_list
    })

with open("runtime/retrieval_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("Retrieval file saved.")