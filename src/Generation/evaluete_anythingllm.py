#Necessário baixar pip install rouge-score bert-score
#Necessário baixar pip install sacrebleu
import json
import sacrebleu
from rouge_score import rouge_scorer
from bert_score import score
import re
import string
from collections import Counter

#ROUGE SCORE E BERTSCORE
def normalize_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

#EXATC MATCH
def exact_match_score(prediction, ground_truth):
    return int(
        normalize_text(prediction) == normalize_text(ground_truth) #compara as duas strings normalizadas extritamente em igualdade- extremamente rígida 
    )
#F1 SCORE
def f1_score(prediction, ground_truth):
    pred_tokens = normalize_text(prediction).split()
    gt_tokens = normalize_text(ground_truth).split()

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens) #quantos itens relevantes foram recuperados
    recall = num_same / len(gt_tokens) #quantos itens recuperados são relevantes
#mede a "recuperação", mas dentre os próprios tokens do ground truth
    return 2 * precision * recall / (precision + recall) #calculo final do f1

#LEITURA DO ARQUIVO DE RESULTADOS
with open("runtime/crud_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

rouge_scores = []
em_scores = []
f1_scores = []
bleu_hyps = []
bleu_refs = []
refs = []
hyps = []

for item in data:
    gt = item["ground_truth"]
    gen = item["generated_answer"]

    rouge_scores.append(rouge.score(gt, gen)["rougeL"].fmeasure)
    em_scores.append(exact_match_score(gen, gt))
    f1_scores.append(f1_score(gen, gt))

#BLEU SCORE
    bleu_hyps.append(gen)
    bleu_refs.append([gt])
    refs.append(gt)
    hyps.append(gen)
P, R, F1 = score(hyps, refs, lang="en", verbose=True) #utilização do preicision, recall e f1 para avaliação.
bleu = sacrebleu.corpus_bleu(bleu_hyps, list(zip(*bleu_refs)))

print("Exact Match (avg):", sum(em_scores) / len(em_scores))
print("F1-score (avg):", sum(f1_scores) / len(f1_scores))
print("ROUGE-L (avg):", sum(rouge_scores) / len(rouge_scores))
print("BLEU-4:", bleu.score)
print("BERTScore (avg):", F1.mean().item())