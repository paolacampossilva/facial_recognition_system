# D6: estratégia de comparação

- **Status:** Aceita
- **Data:** 2026-10-18
- **Dona:** Paola

## Contexto

Na aplicação, um rosto novo é comparado com a galeria de pessoas cadastradas, que têm várias fotos cada. O enunciado lista distância euclidiana, cosseno, k-NN, SVM e protótipos como estratégias a investigar (seção 28). O E07 escolhe a estratégia na galeria privada, e cada estratégia precisa de um escore de rejeição para o threshold de "pessoa desconhecida" (seção 32 e D8).

## Decisão

Quatro estratégias, comparadas no E07 com o melhor modelo de E01 a E06:

| Estratégia | Como decide | Escore de rejeição |
|---|---|---|
| **Cosseno com o vizinho mais próximo** | pessoa da foto da galeria mais parecida | maior similaridade |
| **Protótipo** | média dos embeddings de cada pessoa, renormalizada | similaridade com o protótipo vencedor |
| **k-NN** | k = 3, cosseno, votos ponderados pela similaridade | similaridade média dos vizinhos da classe vencedora |
| **SVM linear** | C escolhido na validação entre 0,1, 1 e 10 | probabilidade máxima, calibrada por Platt |

A sessão 1 é a galeria (cadastro e treino do k-NN e do SVM). A estratégia é escolhida na validação privada e reportada no teste privado (D9).

## Alternativas consideradas

- **Distância euclidiana separada:** com embeddings L2-normalizados, d² = 2 − 2·cos, então ela ordena igual ao cosseno e daria o mesmo resultado.
- **k = 1:** é o mesmo que o vizinho mais próximo.
- **Outros classificadores** (regressão logística, MLP): ficam fora do escopo pelo prazo. O SVM já representa a família dos classificadores treinados.

## Consequências

- Os escores de rejeição não estão na mesma escala (similaridade nas três primeiras, probabilidade no SVM), então cada estratégia tem o próprio threshold, escolhido na validação pela D8.
- Com cerca de 10 fotos por pessoa na galeria, a calibração de Platt do SVM pode ser instável. Se for, isso entra na análise do E07.
- As métricas de open-set e a escolha do threshold já existem em `src/facial_recognition_system/evaluation/` (D7 e D8). As estratégias entram com o E07 (#27).
