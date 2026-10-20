# D5: funções de perda e fine-tuning parcial

- **Status:** Aceita
- **Data:** 2026-10-18
- **Dona:** Valentina

## Contexto

A seção 28 pede para investigar o fine-tuning parcial e funções de perda de metric learning (Triplet, ArcFace e outras). O treino usa o subconjunto da D2: 433 identidades e 3.961 fotos do split de treino do LFW. É pouco para treinar uma rede inteira, e o FaceNet do E01 já parte de um EER de validação de 0,76% (D4). O treino roda na RTX 4060, com 8 GB (D10). A validação e o teste usam identidades que **não** estão no treino (D9), então o modelo precisa generalizar para pessoas novas, não classificar as 433 conhecidas.

## Decisão

**ArcFace (E04) e Triplet Loss (E06), com a mesma configuração de fine-tuning parcial do FaceNet.**

- **ArcFace:** cabeça de classificação sobre as 433 identidades, com escala s = 32 e margem angular m = 0,5. Cada identidade tem um vetor central aprendido, e o embedding só reduz a perda se ficar pelo menos m radianos mais perto do próprio centro do que dos outros. A cabeça só existe no treino; depois, o FaceNet segue sozinho.
- **Triplet:** margem 0,2, *batch-hard mining* (para cada âncora, o positivo mais distante e o negativo mais próximo do lote), amostragem P×K com P = 32 identidades e K = 4 fotos por lote.
- **Comum às duas**, para a comparação ser controlada:
  - só `block8`, `last_linear` e `last_bn` treinam: **2,5 M de 23,5 M parâmetros** (11%). O resto fica congelado, e as **107 BatchNorm da parte congelada ficam em modo `eval`**, para as estatísticas delas não mudarem com os lotes do LFW. O dropout (p = 0,6) continua ativo, como no treino original;
  - AdamW, learning rate de 1e-4 no backbone e 1e-3 na cabeça ArcFace, weight decay de 5e-4;
  - batch de 128, até 30 épocas;
  - flip horizontal e color jitter leve (brilho, contraste e saturação de 0,2) **só no treino**;
  - **early stopping pelo EER nos pares de validação**, com paciência 5. O EER é medido também antes de treinar (época 0), como referência.
- Só as épocas treinadas disputam o melhor checkpoint. Assim, o resultado é sempre um modelo com fine-tuning, e a comparação com a época 0 responde à hipótese.

## Alternativas consideradas

- **Early stopping pela loss de validação:** as identidades da validação não existem na cabeça ArcFace (o split é por identidade), então não há loss de classificação para elas. O EER dos pares mede exatamente o que a aplicação faz (comparar dois rostos) e é a mesma métrica para as duas perdas.
- **Fine-tuning completo (E03 e E05):** com 3.961 fotos, treinar 23,5 M de parâmetros tende a decorar as 433 identidades e a esquecer o que o VGGFace2 ensinou. Ficou de fora por prazo e só volta se sobrar tempo (division.md).
- **Cross-entropy simples (softmax sem margem):** separa as classes do treino, mas não força a compactação de cada identidade, que é o que importa para pessoas novas. O ArcFace é a versão com margem dessa mesma ideia.
- **Contrastive Loss:** usa pares em vez de trios e aproveita pior cada lote. O Triplet com batch-hard já cobre o metric learning por distância.
- **Escolher a melhor época incluindo a época 0:** se nenhuma época treinada superasse o pré-treinado, o E04 viraria uma cópia do E01 e esconderia o efeito do fine-tuning.

## Consequências

- **E04 (ArcFace), primeira leitura:**
  - o early stopping parou na época 6 e ficou com a **época 1**;
  - o EER de validação caiu de 0,76% (época 0) para 0,48%;
  - **no teste, o E04 empatou com o E01**: EER de 1,18% contra 1,11%, com IC de 95% praticamente iguais (0,76–1,59).

  A validação fica otimista porque é nela que a melhor época é escolhida. A loss de treino caiu de 19 para 3, enquanto o EER de validação voltou a subir a partir da época 4: é o overfitting nas 433 identidades que a D2 previa. A configuração não foi mudada depois de olhar o teste. As métricas oficiais vêm da avaliação (#20).
- O treino é barato: cerca de 20 s por época na 4060 e 2 minutos no total (D10). Repetir com outros hiperparâmetros caberia no prazo, mas qualquer ajuste precisa ser escolhido na validação, nunca no teste.
- O E06 (Triplet, #22) reutiliza o mesmo loop, trocando só a cabeça e o amostrador P×K.
- **Limitação:** uma seed por treino (D11), então diferenças pequenas entre E01, E04 e E06 precisam ser lidas junto com os intervalos de confiança da D7.
- Implementado em [src/facial_recognition_system/training/](../../src/facial_recognition_system/training/) (PR #74): `arcface.py`, `freezing.py`, `data.py` e `loop.py`, além de [scripts/train.py](../../scripts/train.py). Config em [configs/E04.yaml](../../configs/E04.yaml).
