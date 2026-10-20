# D11: registro de experimentos

- **Status:** Aceita
- **Data:** 2026-10-18
- **Dona:** Valentina

## Contexto

Cada experimento precisa de hipótese, configuração, dataset, modelo, estratégia, hiperparâmetros, métricas, resultado e interpretação (seção 27), e precisa ser reproduzível: seed, versões, hiperparâmetros, arquitetura, épocas, optimizer, learning rate, batch size, transformações e hardware (seção 34). As duas integrantes trabalham em fatias separadas (a Valentina gera embeddings e treina, a Paola avalia), então o registro também é o contrato entre as fatias. Nada que derive do dataset privado pode ir para o GitHub, nem resultado por pessoa (seções 18 a 20).

## Decisão

**Um YAML por experimento em `configs/` e um JSON por experimento em `experiments/results/`, ambos versionados, sem ferramenta externa.**

- **Config:** `configs/<EXP>.yaml` (`E01.yaml`, ...), validado por `ExperimentConfig` (pydantic com `extra="forbid"`, então um campo com erro de digitação falha em vez de ser ignorado). Tem a hipótese, a estratégia, o backbone, os caminhos dos dados, o batch e, nos experimentos com fine-tuning, a seção `training` com **todos** os hiperparâmetros da D5 escritos por extenso, mesmo os que coincidem com o padrão.
- **Resultado:** `experiments/results/<EXP>.json` com `experiment`, `hypothesis`, `model`, `strategy`, `dataset`, `seed`, `config` (cópia do YAML), `libraries` (Python, torch, torchvision, facenet-pytorch, scikit-learn, numpy), `hardware` (dispositivo, GPU, CUDA, sistema e CPU, sem nome de máquina nem de usuário), `embeddings` (fotos, identidades, dimensão e tempo por split), `training` (quando há treino), `threshold` e `metrics`. Quem extrai os embeddings grava `threshold` e `metrics` como `null`, e a avaliação os preenche.
- **Seed 42** em Python, NumPy e PyTorch (`seed_everything`), com o autotuning do cuDNN desligado. Também usam a seed os sorteios do split (D9), do subconjunto de fine-tuning (D2) e do embaralhamento do DataLoader.
- **Uma seed por treino**, por falta de tempo. A variação é medida pelo bootstrap da D7 nos pares, não por várias execuções do treino.
- Embeddings (`embeddings/`) e checkpoints (`checkpoints/`) ficam **fora do Git** (`.gitignore` e hook de pre-commit). No privado, ficam no cofre (D10). Em `experiments/` só entram números agregados.

## Alternativas consideradas

- **MLflow ou Weights & Biases:** mais uma dependência e mais um servidor, ou um serviço externo. O W&B mandaria métricas e configs para fora da máquina, o que complica a regra de nada do dataset privado sair do notebook da 4060 (seção 23). Para seis experimentos, YAML e JSON no Git bastam e já ficam no histórico revisado.
- **Hiperparâmetros como argumentos de linha de comando:** o comando exato se perde. O YAML versionado é o registro.
- **Várias seeds por experimento (por exemplo, 3 treinos de E04):** daria a variância do treino, mas triplicaria o tempo da 4060 numa semana que já está cheia.

## Consequências

- Reproduzir um experimento é rodar um comando com o YAML dele (`scripts/extract_embeddings.py` ou `scripts/train.py`), e o README lista esses comandos.
- Rodar de novo **substitui** o JSON inteiro, incluindo `threshold` e `metrics`, para nunca sobrar métrica calculada com embeddings antigos. Depois de refazer a extração, a avaliação precisa rodar de novo.
- O seed fixo e o cuDNN sem autotuning tornam o resultado repetível na mesma máquina (o `--smoke` com a mesma configuração deu os mesmos números em duas execuções), mas não garantem bits idênticos em outra GPU ou outra versão do CUDA. Por isso o hardware e as versões ficam no JSON.
- Implementado em [src/facial_recognition_system/experiments/](../../src/facial_recognition_system/experiments/) (`config.py`, `environment.py`, `extraction.py` e `embeddings.py`; PRs #68 e #74).
