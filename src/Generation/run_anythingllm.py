
import json #para ler e gravar os arquivos que no modelo do data set
import requests #faz requisições diretamente na API no anythingLLM
from tqdm import tqdm #mostra barra de conclusão, unicamente visual

ANYTHINGLLM_API = "http://localhost:3001/api/v1/workspace/meu-workspace/chat" # AQUI O WORKSPACE SLUG NÃO É O MESMO QUE O NOME DO WORSKPACE
API_KEY = "XV060WE-0G9M47P-GATDFKY-S2DC8D1"

headers = {
    "Authorization": f"Bearer {API_KEY}", #meu login no anything pela key
    "Content-Type": "application/json" #modelo do arquivo enviado JSON
}

with open("runtime/rag_questions_EXATAS.json", "r", encoding="utf-8") as f: #leitura do data set no modelo CRUD dentro da pasta DATA
    questions = json.load(f)

results = []

for item in tqdm(questions):
    payload = {
        "message": item["question"],
        "mode": "chat",
        "reset": True,
        "--think": False
    }

    response = requests.post(ANYTHINGLLM_API, headers=headers, json=payload)
    response.raise_for_status()


    answer = response.json().get("textResponse")## MUDOU AQUI
    print(answer) # ESSA LINHA É SÓ PRA VERIFICAÇÃO 

    print(json.dumps(response.json(), indent=2)) #mostra os chunks relevantes de cada questão


    
    results.append({ #salva os resultados parciais na lista results
         "id": item["id"],
         "question": item["question"], #pergunta do prompt
         "ground_truth": item["ground_truth"], #resposta esperada
         "generated_answer": answer #resposta gerada
     })

with open("runtime/crud_results.json", "w", encoding="utf-8") as f: #write dos resultados parciais no arquivo JSON results
    json.dump(results, f, ensure_ascii=False, indent=2)