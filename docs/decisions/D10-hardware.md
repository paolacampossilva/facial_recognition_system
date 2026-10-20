# D10: hardware

- **Status:** Aceita
- **Data:** 2026-10-18
- **Dona:** Valentina

## Contexto

O grupo tem duas máquinas: o notebook da Valentina, com uma RTX 4060 Laptop de 8 GB (CUDA 12.8), e o Mac da Paola, com chip M5 e sem CUDA. Os experimentos precisam registrar o hardware e ser comparáveis entre si, inclusive no custo computacional (seções 33 e 34). Os dados privados precisam ficar no computador do grupo, em diretório protegido e com acesso restrito (seção 22), e modelos e embeddings derivados deles também são privados (seções 19 e 20).

## Decisão

**O notebook da RTX 4060 roda todos os experimentos**: o `--smoke`, E01 a E06 e, com as fotos privadas, E07 e E08.

- O fine-tuning da D5 é leve: só o último bloco do FaceNet treina, em algumas centenas de identidades, com recortes de 160×160. Isso cabe nos 8 GB com folga.
- As fotos privadas e tudo que deriva delas (recortes, embeddings, checkpoints e resultados por pessoa) ficam num cofre VeraCrypt nesse notebook, aberto só durante o uso, ligado ao projeto por junções do Windows (`data/private/`, `faces/private/`, `embeddings/private/` e `checkpoints/private/`). Só a Valentina acessa.
- O Mac da Paola serve para desenvolver e testar com dados sintéticos ou do LFW. Os testes unitários não dependem de GPU, e os scripts aceitam `--device cpu`.

## Alternativas consideradas

- **Dividir os treinos entre as máquinas:** os tempos de treino deixariam de ser comparáveis (seção 33), e haveria dois ambientes para manter iguais. O MPS do Mac também não tem todas as operações do PyTorch e daria números ligeiramente diferentes.
- **GPU na nuvem (Colab, Kaggle):** seria preciso subir as fotos privadas para um serviço externo, o que a seção 23 proíbe sem consentimento. Para os dados públicos não haveria problema, mas a 4060 já dá conta.
- **Treinar no CPU:** possível para o `--smoke`, mas bem mais lento nos treinos completos e sem ganho nenhum.

## Consequências

- Medido no E04, na 4060: **pico de 660 MB de memória da GPU**, cerca de 20 s por época incluindo a validação (3.961 fotos de treino e 2.651 de validação) e **2 minutos no total**. Extrair os embeddings da validação e do teste leva cerca de 15 a 30 s por modelo. O orçamento de 8 GB e de tempo não limita os experimentos.
- Os tempos dependem do cache de disco: a primeira leitura dos recortes é mais lenta que as seguintes. Por isso o custo computacional da seção 33 usa o tempo de treino por época, não o tempo de extração.
- O ambiente da 4060 é o de referência: Python 3.11.9, torch 2.7.1+cu128, torchvision 0.22.1+cu128, CUDA 12.8. Cada JSON em `experiments/results/` registra versões e hardware (D11).
- O notebook é um ponto único de falha: se ele parar, os experimentos com dados privados param. Os dados públicos e o código podem ser refeitos em outra máquina a partir do Git e do `scripts/download_datasets.py`.
- Testado com `pytest -m gpu -s` ([tests/test_environment.py](../../tests/test_environment.py)). O cofre está descrito no [division.md](../division.md), em "Cofre".
