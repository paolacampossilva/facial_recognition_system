# D6: estratégia de comparação

- **Status:** Aceita, com o critério de escolha do E07 e a calibração do SVM definidos na implementação (ver Decisão e Consequências)
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

**Critério do E07:** cada estratégia (e cada valor de C do SVM) recebe o próprio threshold da D8 na validação, e vence a de maior **F1 macro na validação** nesse threshold, com "desconhecido" contado como classe. Em caso de empate, vence a mais simples, na ordem da tabela.

## Alternativas consideradas

- **Distância euclidiana separada:** com embeddings L2-normalizados, d² = 2 − 2·cos, então ela ordena igual ao cosseno e daria o mesmo resultado.
- **k = 1:** é o mesmo que o vizinho mais próximo.
- **Outros classificadores** (regressão logística, MLP): ficam fora do escopo pelo prazo. O SVM já representa a família dos classificadores treinados.
- **Escolher a estratégia pelo EER do escore de rejeição:** não depende de threshold, mas ignora as identificações erradas entre pessoas da galeria, que o F1 macro conta.
- **Escolher pela accuracy no threshold da D8:** os cerca de 200 desconhecidos do LFW dominariam a conta, e uma estratégia que só rejeita bem pareceria boa.

## Consequências

- Os escores de rejeição não estão na mesma escala (similaridade nas três primeiras, probabilidade no SVM), então cada estratégia tem o próprio threshold, escolhido na validação pela D8.
- Com cerca de 10 fotos por pessoa na galeria, a calibração de Platt do SVM pode ser instável. Se for, isso entra na análise do E07.
- A calibração de Platt usa `CalibratedClassifierCV(SVC(kernel="linear"), method="sigmoid", ensemble=False)`, porque `SVC(probability=True)` está obsoleto no scikit-learn 1.9. O número de folds da calibração é 5, ou o número de fotos da pessoa com menos fotos na galeria, se for menor (mínimo de 2).
- Implementado em [src/facial_recognition_system/recognition/strategies.py](../../src/facial_recognition_system/recognition/strategies.py), com a mesma interface (`fit` na galeria, `predict` devolvendo identidade e escore) que a aplicação vai usar na Parte 2. O E07 e o E08 rodam com [scripts/run_open_set.py](../../scripts/run_open_set.py) e [configs/E07_E08.yaml](../../configs/E07_E08.yaml), e o critério de escolha fica em [src/facial_recognition_system/experiments/open_set.py](../../src/facial_recognition_system/experiments/open_set.py) (#27).
