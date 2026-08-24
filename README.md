# Avalia-o-de-Small-Language-Models-com-Retrieval-Augmented-Generation-em-contextos-educacionais.
Scripts:
  Geração: 
    -Run: Envia o payload via requisição para a API do AnythingLLM -> armazena as respostas geradas pelo modelo em cada question.
    -Evaluate: realiza comparações de sequências semânticas através do RougeL e similaridade semântica a partir da métrica BertScore.
    
  Recuperação:
    -Run: Envia o Payload para o AnythingLLM -> realiza o embedding das questions e realiza um comparação em similarity threshold com embeddings vetorizados dos chunks do documento de contextualização pré vetorizado no AnythingLLM -> armazena os chunks recuperados.
    -Evaluate: Realiza um segundo embedding, porém agora entre os chunks recuperados e os ground_truth / gold_lists diretamente -> realiza a comparação por cosseno de cada dupla de vetores -> usa o limiar definido pela comparação por cosseno para definir se o chunk é considerado relevante ou não -> esses chunk então são armazenados nas listas usadas como bases para o calculo das métricas Hit@k (taxa de sucesso - binário) e nDCG (ranqueamento - pesos ponderados e posição) -> print dos resultados.
