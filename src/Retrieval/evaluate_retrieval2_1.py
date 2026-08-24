import os
import json
import glob
import argparse
import math
import re
import requests
import numpy as np

OLLAMA_URL = "http://localhost:11434/api/embed"

EMBEDDING_MODEL = "bge-m3:567m"

SIMILARITY_THRESHOLD = 0.60

# Cache de embeddings
embedding_cache = {} #salva os ground truth para não recupera-los mais vezes do que o necessário

#remove o cabeçalho de metadata (<document_metadata>...</document_metadata>) e o prefixo
#"passage:" que o AnythingLLM grava junto ao texto do chunk. Sem essa limpeza, esse
#ruído entra no embedding e reduz artificialmente a similaridade de cosseno contra o
#ground truth, mesmo quando o chunk é de fato relevante.
def clean_chunk_text(text): #text são os chunks recuperados, após o embbeding do anything(vetor PERGUNTA X vetor chunks DOCUMENTO), salvos no retrieval_results -> eles sofrerão embeddings novamente para serem comparados agora com os chnks gold e comparados pelo cosseno (vetor DOCUMENTOS RECUPERADO RELEVANTE (text) X vetor GOLD LIST (fact) )
    text = re.sub(r'<document_metadata>.*?</document_metadata>', '', text, flags=re.DOTALL)
    text = re.sub(r'^\s*passage:\s*', '', text.strip(), flags=re.IGNORECASE)
    return text

#normalização para formatar todos os textos da mesma forma e retirar espaços, pontuações e erros de pdf
def normalize(text):
    text = clean_chunk_text(text)
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

#embedding
def get_embedding(text):

    text = normalize(text)

    if text in embedding_cache:
        return embedding_cache[text]

    payload = {
        "model": EMBEDDING_MODEL,
        "input": text
    }

    try:

        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=60
        )

        response.raise_for_status()

    except requests.exceptions.RequestException as e:

        raise RuntimeError(f"Erro ao obter embedding: {e}")

    embedding = np.array(response.json()["embeddings"][0])

    embedding_cache[text] = embedding

    return embedding

#similaridade agora com cosseno
def cosine_similarity(vec1, vec2): #mudou bastante aqui, antes era similaridade apenas textual, agora por cosseno 
    denominator = (
        np.linalg.norm(vec1) *
        np.linalg.norm(vec2)
    )
    if denominator == 0:
        return 0
    return np.dot(vec1, vec2) / denominator

# melhor correspondência 
def best_match(chunk, gold_list): #é oq vai relacionar a avaliação com mais de um chunk como ground_truth

    chunk_embedding = get_embedding(chunk)

    best_gold = None
    best_score = float("-inf")

    for gold in gold_list:

        gold_embedding = get_embedding(gold)

        score = cosine_similarity(
            chunk_embedding,
            gold_embedding
        )

        if score > best_score:

            best_score = score
            best_gold = gold

    return best_gold, best_score

def calculate_metrics(retrieved_lists, gold_lists, K=10):
    hits_k = 0
    similarity_scores = []


    map_k_list = []
    mrr_list = []
    ndcg_k_list = []

    for retrieved, gold in zip(retrieved_lists, gold_lists):
        gold = [normalize(g) for g in gold]
        retrieved = [normalize(r) for r in retrieved]
        retrieved_k = retrieved[:K]
        relevant_count = len(gold)
        relevant_retrieved = 0
        average_precision_sum = 0
        first_relevant_rank = None
        dcg = 0
        found_golds = set()

        # Pré-calcula embeddings dos golds
        for g in gold:
            get_embedding(g)

        # Avaliação Top-K
        for rank, chunk in enumerate(retrieved_k, start=1):
            matched_gold, similarity_score = best_match(
                chunk,
                gold
            )
            similarity_scores.append(similarity_score)
            if (
                matched_gold is not None
                and similarity_score >= SIMILARITY_THRESHOLD #usa o threshold do começo ver se não vai falhar com o modelo tentanto recuperar e a métrica usando como condição, pois no Anything não recuperava se era maior que 0,25
                and matched_gold not in found_golds
            ):
                found_golds.add(matched_gold)
                relevant_retrieved += 1

                if first_relevant_rank is None:
                    first_relevant_rank = rank
                precision_at_rank = relevant_retrieved / rank
                average_precision_sum += precision_at_rank
                dcg += 1 / math.log2(rank + 1)

        # nDCG
        ideal_relevant = min(
            relevant_count,
            K
        )
        idcg = sum(
            1 / math.log2(i + 1)
            for i in range(1, ideal_relevant + 1)
        )
        ndcg = dcg / idcg if idcg > 0 else 0

        # MAP
        if relevant_count == 0:
            map_k = 0
        else:
            map_k = (
                average_precision_sum /
                min(relevant_count, K)
            )

        # MRR
        if first_relevant_rank is None:
            mrr = 0
        else:
            mrr = 1 / first_relevant_rank

        # Hits
        if relevant_retrieved > 0:
            hits_k += 1

        map_k_list.append(map_k)
        mrr_list.append(mrr)
        ndcg_k_list.append(ndcg)

    total = len(gold_lists)
    if similarity_scores: #garante que não há erro caso nenhum chunk for recuperado
        print(f"Average cosine similarity: {np.mean(similarity_scores):.4f}")
        print(f"Std cosine similarity: {np.std(similarity_scores):.4f}")
        print(f"Max cosine similarity: {np.max(similarity_scores):.4f}")
        print(f"Min cosine similarity: {np.min(similarity_scores):.4f}")
    return {

        f"Hits@{K}":
            hits_k / total,

        f"MAP@{K}":
            sum(map_k_list) / total,

        f"MRR@{K}":
            sum(mrr_list) / total,

        f"nDCG@{K}":
            sum(ndcg_k_list) / total,
    }

def main_eval(file_name, K=10):
    print(f'\nEvaluating file: {file_name}')

    with open(file_name, 'r', encoding='utf-8') as file:
        data = json.load(file)

    retrieved_lists = []
    gold_lists = []

    for d in data:
        if d.get('question_type') == 'null_query':
            continue
        retrieved_lists.append([m['text'] for m in d['retrieval_list']])
        gold_lists.append([m['fact'] for m in d['gold_list']])

    metrics = calculate_metrics(retrieved_lists, gold_lists, K)

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    print('-' * 30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=False)
    parser.add_argument('--path', type=str, default="output")
    parser.add_argument('--k', type=int, default=10)

    args = parser.parse_args()

    if args.file:
        main_eval(args.file, args.k)
    else:
        json_files = glob.glob(os.path.join(args.path, '*.json'))
        for file in json_files:
            main_eval(file, args.k)
            #python src/Retrieval/evaluate_retrieval2_1.py --file runtime/retrieval_results.json --k 10 