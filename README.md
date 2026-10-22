<details open>
<summary><strong>🇧🇷 Português</strong></summary>

# Reconhecimento Facial

Sistema de **reconhecimento facial** desenvolvido para a disciplina de Inteligência Artificial da **UFMS**, em duas partes: **experimentação e fine-tuning** de modelos de embedding facial e uma **aplicação** de cadastro e reconhecimento construída com o melhor modelo encontrado.

**Integrantes:**

- Paola Campos da Silva
- Valentina Campos Soares

## Sumário

- [Visão geral](#visão-geral)
  - [Pipeline previsto](#pipeline-previsto)
- [Tecnologias](#tecnologias)
- [Instalação](#instalação)
  - [Hook de privacidade](#hook-de-privacidade)
  - [Testes](#testes)
- [Conjuntos de dados](#conjuntos-de-dados)
  - [Dataset público](#dataset-público)
  - [Dataset privado](#dataset-privado)
- [Experimentos (Parte 1)](#experimentos-parte-1)
  - [Preparação dos dados](#preparação-dos-dados)
  - [Protocolo de avaliação](#protocolo-de-avaliação)
  - [Métricas](#métricas)
  - [Threshold](#threshold)
  - [Reprodutibilidade](#reprodutibilidade)
- [Privacidade e LGPD](#privacidade-e-lgpd)
- [Estrutura de pastas](#estrutura-de-pastas)
- [Tipos de commit](#tipos-de-commit)

## Visão geral

> **Status:** em desenvolvimento. O ambiente e os testes estão documentados em [Instalação](#instalação); a execução dos experimentos e da aplicação será documentada quando existir.

O enunciado completo está em [docs/assignment.md](docs/assignment.md).

| Parte | Objetivo | Entrega |
|---|---|---|
| Parte 1, Experimentação | Comparar modelos, estratégias de treinamento/fine-tuning, funções de perda e thresholds para descobrir a melhor configuração | Código, experimentos, métricas e vídeo com os resultados |
| Parte 2, Aplicação | Usar a melhor configuração da Parte 1 numa aplicação de cadastro, reconhecimento e exclusão de pessoas | Aplicação funcional e vídeo de demonstração |

### Pipeline previsto

```text
Imagem / câmera
      ↓
Detecção da face
      ↓
Alinhamento / pré-processamento
      ↓
Rede neural
      ↓
Embedding facial
      ↓
Comparação com os cadastrados
      ↓
Threshold
      ↓
Reconhecido / desconhecido
```

Uma face com similaridade abaixo do threshold é classificada como **pessoa desconhecida**. O valor do threshold será escolhido experimentalmente na Parte 1.

## Tecnologias

| Camada | Tecnologia |
|---|---|
| Linguagem | Python 3.11 |
| Modelo de embedding facial | A definir na Parte 1 |
| Framework de deep learning | PyTorch 2.7.1 (CUDA 12.8) |
| Interface da aplicação | A definir na Parte 2 |

## Instalação

Requisitos: **Python 3.11.4 ou mais recente na série 3.11** (o `tarfile.extractall(filter=...)` usado em [scripts/download_datasets.py](scripts/download_datasets.py) não existe no 3.11.0 a 3.11.3) e, para treinar, uma GPU NVIDIA com driver compatível com CUDA 12.8. Máquinas do projeto: notebook com RTX 4060 (experimentos e treino) e Mac com M5 (desenvolvimento, sem CUDA).

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install facenet-pytorch==2.6.0 --no-deps
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
pre-commit install
```

A ordem importa:

- o **torch vem do índice cu128**, o build CUDA 12.8 do torch 2.7.1. No macOS, que não tem CUDA, instale sem o `--index-url` (`python -m pip install torch==2.7.1 torchvision==0.22.1`), e o resto é igual;
- o **facenet-pytorch é instalado com `--no-deps`**, porque a versão 2.6.0 declara `torch<2.3`. As dependências que ele realmente usa estão fixadas no [requirements.txt](requirements.txt);
- o `pip` vai avisar que o facenet-pytorch pede outras versões de torch, numpy e Pillow. O aviso é esperado e os testes abaixo confirmam que o pipeline funciona;
- o **`pip install -e .`** instala o pacote do projeto (`src/facial_recognition_system`) em modo editável, para os scripts em `scripts/` conseguirem importá-lo. O `--no-deps` evita que o pip mexa nas versões já instaladas. Sem esse passo, o `pytest` funciona, mas os scripts falham com `ModuleNotFoundError: No module named 'facial_recognition_system'`.

### Hook de privacidade

O `pre-commit install` ativa, neste clone, os hooks do [.pre-commit-config.yaml](.pre-commit-config.yaml), entre eles o [scripts/check_private_files.py](scripts/check_private_files.py), que bloqueia fotos, vídeos, embeddings, checkpoints e bancos em qualquer commit. É a segunda barreira depois do `.gitignore` e precisa ser rodado **uma vez em cada clone**, porque o Git não versiona os hooks. Para conferir se está ativo:

```bash
pre-commit run --all-files
```

Nunca use `git commit --no-verify`, que pula o hook.

### Testes

```bash
pytest              # testes em CPU (padrão)
pytest -m gpu -s    # testes de GPU; -s mostra qual placa e qual versão do CUDA foram usadas
```

Os testes marcados com `gpu` ficam fora da execução padrão e **falham** (não são pulados) se a GPU não conseguir rodar o modelo. Rode os dois comandos em toda máquina nova com GPU NVIDIA; no Mac, só o primeiro.

## Conjuntos de dados

### Dataset público

Nenhuma cópia de dataset público é versionada. O `python scripts/download_datasets.py` baixa cada um da fonte e confere o checksum. Detalhes, citação e instruções em [data/README.md](data/README.md).

| Dataset | Fonte oficial | Finalidade | Licença / termos de uso | Como foi usado |
|---|---|---|---|---|
| LFW (Labeled Faces in the Wild) | <http://vis-www.cs.umass.edu/lfw/> (fora do ar; baixado da cópia no figshare usada pelo scikit-learn) | benchmark de verificação e subconjunto para fine-tuning | sem licença formal, distribuído para pesquisa com pedido de citação; não redistribuído aqui | alinhado com MTCNN e dividido por identidade em treino, validação e teste |

### Dataset privado

Fotos dos integrantes e de pessoas próximas, maiores de 18 anos, coletadas diretamente pela dupla e com Termo de Consentimento assinado. Esse dataset **nunca** é versionado nem enviado a serviços externos; a única exceção é o envio das fotos pelo WhatsApp, só para quem autorizou no termo, com a conversa apagada em até 24 horas (ver [Privacidade e LGPD](#privacidade-e-lgpd) e [data/README.md](data/README.md)).

## Experimentos (Parte 1)

Resultados envolvendo o dataset privado aparecem apenas de forma agregada, sem identificar participantes.

Cada experimento tem um YAML em [configs/](configs/) e grava o resultado em `experiments/results/<EXP>.json`. Os embeddings (`embeddings/`) e os checkpoints (`checkpoints/`) ficam fora do Git.

```bash
# sem fine-tuning (E01, E02): só extrai os embeddings da validação e do teste
python scripts/extract_embeddings.py --config configs/E01.yaml

# com fine-tuning (E04, E06): rode o --smoke antes (2 épocas curtas, alguns segundos)
python scripts/train.py --config configs/E04.yaml --smoke
python scripts/train.py --config configs/E04.yaml

# privado (E07 e E08): só no notebook da 4060, com o cofre aberto; grava só resultados agregados
python scripts/preprocess_faces.py --input-dir data/private --output-dir faces/private
python scripts/run_open_set.py --config configs/E07_E08.yaml
```

| Experimento | Modelo | Estratégia | Dataset | Accuracy | F1 | Observação |
|---|---|---|---|---|---|---|
| A preencher | | | | | | |

### Preparação dos dados

Todos os experimentos usam o mesmo pipeline de detecção (D3): MTCNN do `facenet-pytorch`, recorte de 160×160, probabilidade mínima de 0,9. A ordem para recriar os dados públicos do zero:

```bash
python scripts/download_datasets.py                       # LFW da fonte, com SHA-256 conferido
python scripts/preprocess_faces.py --input-dir data/public/lfw --output-dir data/public/lfw_crop160 --selection center
python scripts/make_splits.py --originals-dir data/public/lfw --crops-dir data/public/lfw_crop160 --output-dir data/splits
python scripts/check_leakage.py --originals-dir data/public/lfw --crops-dir data/public/lfw_crop160 --splits-dir data/splits
```

Os splits já estão versionados em `data/splits/`, então os dois últimos comandos só são necessários para auditar ou regenerar. No LFW, 1 das 13.233 fotos ficou sem rosto acima de 0,9 e saiu dos splits. Detalhes de cada etapa em [data/README.md](data/README.md).

O fine-tuning (E04 e E06) usa só o treino: identidades com pelo menos 4 fotos, no máximo 40 por identidade, o que dá **433 identidades e 3.961 fotos** (D2).

### Protocolo de avaliação

**Público (E01 a E06), verificação de pares do LFW** (D9):

- Split **por identidade**, 70/15/15 com seed 42, estratificado pelo número de fotos (2–4, 5–9 e 10+). Nenhuma pessoa aparece em dois splits, então o fine-tuning nunca vê quem está na validação ou no teste.
- Pares: todas as combinações de duas fotos da mesma pessoa (até 20 por identidade) e o mesmo número de pares negativos. Todos os experimentos usam **os mesmos pares**.
- Os pares oficiais do LFW (View 2) não são usados, porque têm identidades que entram no treino do fine-tuning.
- Leakage: dHash em duas etapas (originais e recortes) encontrou 2 fotos repetidas entre splits, que foram removidas.

| Split | Identidades | Fotos | Pares (positivos + negativos) |
|---|---|---|---|
| treino | 1.176 | 6.439 | — |
| validação | 2.286 | 3.489 | 1.447 + 1.447 |
| teste | 2.285 | 3.302 | 1.445 + 1.443 |

**Privado (E07 e E08), identificação open-set** (D9):

- A sessão 1 de cada participante é a galeria (cadastro). As fotos da sessão 2 vão metade para a validação e metade para o teste.
- 2 participantes ficam fora da galeria como desconhecidas, uma na validação e outra no teste, com todas as fotos.
- Cada split recebe também 200 rostos de identidades do LFW com uma foto só, para haver tentativas de desconhecidos suficientes para medir FAR de 1%.
- A mesma checagem de leakage roda entre a sessão 1 e a sessão 2.

**Regras que valem para tudo:** o teste nunca escolhe nada (seção 29). O melhor modelo de E01 a E06, que segue para o E07, é escolhido pelo **EER da validação**. A estratégia do E07 e o threshold do E08 são escolhidos na validação privada. O teste é rodado uma vez por experimento, depois que tudo está fixado.

### Métricas

**Verificação** (E01 a E06, D7):

- ROC-AUC, EER e TAR@FAR=1%, que não dependem de threshold;
- accuracy e F1 no threshold escolhido na validação;
- histogramas de similaridade de mesma pessoa e de pessoas diferentes (`experiments/figures/<EXP>_hist.png`) e curvas ROC (`<EXP>_roc.png`, `comparison_roc.png`);
- **intervalo de confiança de 95% por bootstrap**, com 1.000 reamostragens dos pares, positivos e negativos reamostrados separadamente.

**Identificação open-set** (E07 e E08, D7):

- accuracy e F1 macro, com "desconhecido" contado como classe;
- FAR (desconhecidos aceitos como alguém), FRR (conhecidos rejeitados) e taxa de identificação errada (conhecido atribuído à pessoa errada);
- matriz de confusão agregada, só com IDs anônimos (`experiments/figures/E08_confusion.png`).

### Threshold

Uma face é aceita quando `similaridade >= threshold`, e o threshold é sempre escolhido **na validação** e aplicado no teste **sem reajuste** (D8).

| Onde | Critério | Valor |
|---|---|---|
| Público (E01 a E06) | threshold do EER nos pares de validação, onde FAR = FRR. É o critério neutro para comparar modelos | _a preencher_ |
| Privado e aplicação (E08) | menor threshold com **FAR ≤ 1%** na validação. Se o FRR de validação passar de 20%, usa-se o do EER | _a preencher_ |

A falsa aceitação pesa mais que a falsa rejeição: reconhecer um estranho como alguém cadastrado é pior do que pedir para a pessoa tentar de novo. Com cerca de 200 tentativas de desconhecidos por split, o FAR do privado tem resolução de 0,5% (1 aceitação em 200), por isso a meta é 1% e não 0,1%. No E07, cada estratégia de comparação tem o próprio threshold, porque os escores não estão na mesma escala (similaridade no cosseno, no protótipo e no k-NN, probabilidade no SVM).

### Reprodutibilidade

- **Configuração:** um YAML por experimento em [configs/](configs/), com todos os hiperparâmetros escritos por extenso (D11). O resultado vai para `experiments/results/<EXP>.json`, com hipótese, cópia do YAML, seed, versões das bibliotecas, hardware, tempos, threshold e métricas.
- **Seed 42** em Python, NumPy e PyTorch, com o autotuning do cuDNN desligado. Também usam a seed os sorteios do split, do subconjunto de fine-tuning e do DataLoader. Uma seed por treino; a variação é medida pelo bootstrap.
- **Versões:** Python 3.11, PyTorch 2.7.1 com CUDA 12.8, facenet-pytorch 2.6.0, scikit-learn 1.9.1, todas fixadas no [requirements.txt](requirements.txt). O JSON de cada experimento registra as versões com que ele rodou.
- **Hardware:** todos os experimentos rodam no notebook com RTX 4060 (8 GB), o que deixa os tempos de treino comparáveis (D10). O resultado é repetível na mesma máquina, mas não com bits idênticos em outra GPU ou outra versão do CUDA.

Comando de cada experimento:

| Experimento | O que testa | Comando |
|---|---|---|
| E01 | FaceNet (VGGFace2) pré-treinado, sem fine-tuning | `python scripts/extract_embeddings.py --config configs/E01.yaml` |
| E02 | ResNet-50 (ImageNet), sem fine-tuning | `python scripts/extract_embeddings.py --config configs/E02.yaml` |
| E04 | FaceNet com fine-tuning parcial e ArcFace | `python scripts/train.py --config configs/E04.yaml` |
| E06 | FaceNet com fine-tuning parcial e Triplet Loss | `python scripts/train.py --config configs/E06.yaml` |
| Avaliação de E01 a E06 | threshold na validação, métricas com IC e gráficos | `python scripts/evaluate.py --config configs/E01.yaml configs/E02.yaml configs/E04.yaml configs/E06.yaml` |
| E07 e E08 | estratégia de comparação e threshold open-set, no privado | `python scripts/run_open_set.py --config configs/E07_E08.yaml` |

E03 e E05 (fine-tuning completo da ResNet-50) foram cortados por prazo (D4 e D5). O E07 e o E08 só rodam na máquina do grupo, com o cofre aberto e os recortes privados gerados antes (`python scripts/preprocess_faces.py --input-dir data/private --output-dir faces/private`).

## Privacidade e LGPD

Dados biométricos são dados pessoais sensíveis (LGPD, Lei nº 13.709/2018). Descumprir as regras abaixo zera o trabalho.

Ficam **fora do GitHub** e são processados apenas localmente:

- fotografias e vídeos dos participantes;
- nomes e identificadores associados às fotos;
- embeddings faciais e bancos vetoriais;
- checkpoints treinados ou fine-tunados com dados privados;
- termos de consentimento;
- logs e resultados individualizados.

Tudo isso fica num cofre VeraCrypt no notebook da RTX 4060, ligado ao projeto por junções em `data/private/`, `faces/private/`, `embeddings/private/` e `checkpoints/private/` (decisão D10 em [docs/division.md](docs/division.md)).

O [.gitignore](.gitignore) ignora `data/private/`, `data/public/`, `faces/`, `embeddings/`, `checkpoints/`, `consent/`, bancos (`*.db`, `*.sqlite`), pesos de modelo (`*.pt`, `*.pth`, `*.onnx`, `*.safetensors`) e arrays ou classificadores salvos (`*.npy`, `*.npz`, `*.pkl`, `*.joblib`). Como segunda barreira, o hook de pre-commit [scripts/check_private_files.py](scripts/check_private_files.py) bloqueia fotos, vídeos, checkpoints e bancos em qualquer pasta, mesmo adicionados com `git add -f`, e só aceita PDF, Word e Excel em `docs/`, para que termos assinados e a planilha de controle não entrem por engano. A aplicação terá um mecanismo para excluir todos os dados de uma pessoa cadastrada, e os dados privados serão eliminados ao fim da disciplina.

Como cada dado é coletado, guardado, excluído e eliminado, e o mapa do checklist do Anexo B, estão em [docs/privacy.md](docs/privacy.md). Regras completas nas seções 12 a 25 e no Anexo B (checklist de entrega) de [docs/assignment.md](docs/assignment.md).

## Estrutura de pastas

```text
facial_recognition_system/
├── configs/                      # um YAML por experimento
├── data/
│   ├── README.md                 # origem, licença e como obter os dados
│   └── splits/                   # listas de treino/validação/teste (só dados públicos)
├── docs/
│   ├── assignment.md             # enunciado do trabalho
│   ├── decisions/                # ADRs
│   └── specs/                    # especificações com critérios de aceitação
├── experiments/
│   ├── figures/                  # gráficos agregados
│   └── results/                  # um JSON de resultado por experimento
├── scripts/
│   └── check_private_files.py    # hook de pre-commit contra dados privados
├── src/facial_recognition_system/
│   ├── data/                     # splits por identidade e pares de verificação
│   ├── evaluation/               # métricas, threshold e gráficos
│   ├── models/                   # wrappers dos modelos de embedding
│   ├── preprocessing/            # detecção, alinhamento e recorte
│   └── training/                 # loops de fine-tuning e cabeças de loss
├── tests/
├── .gitignore
├── .pre-commit-config.yaml
├── pyproject.toml                # configuração do ruff e do pytest
├── requirements.txt              # versões fixadas (exceto torch e facenet-pytorch)
└── README.md
```

## Tipos de commit

| Tipo | Descrição |
|---|---|
| `feat` | Nova funcionalidade |
| `fix` | Correção de bug |
| `docs` | Alteração de documentação |
| `test` | Adição ou ajuste de testes |
| `refactor` | Refatoração sem mudar comportamento |
| `perf` | Melhoria de desempenho |
| `chore` | Tarefas de manutenção (dependências, configuração) |
| `ci` | Alterações de integração contínua |

</details>

<details>
<summary><strong>🇺🇸 English</strong></summary>

# Facial Recognition

**Facial recognition** system built for the Artificial Intelligence course at **UFMS**, in two parts: **experimentation and fine-tuning** of face embedding models, and an enrollment and recognition **application** built on the best model found.

**Team members:**

- Paola Campos da Silva
- Valentina Campos Soares

## Table of contents

- [Overview](#overview)
  - [Planned pipeline](#planned-pipeline)
- [Technologies](#technologies)
- [Installation](#installation)
  - [Privacy hook](#privacy-hook)
  - [Tests](#tests)
- [Data](#data)
  - [Public dataset](#public-dataset)
  - [Private dataset](#private-dataset)
- [Experiments (Part 1)](#experiments-part-1)
  - [Data preparation](#data-preparation)
  - [Evaluation protocol](#evaluation-protocol)
  - [Metrics](#metrics)
  - [Threshold](#threshold-1)
  - [Reproducibility](#reproducibility)
- [Privacy and LGPD](#privacy-and-lgpd)
- [Folder structure](#folder-structure)
- [Commit types](#commit-types)

## Overview

> **Status:** in development. The environment and tests are documented in [Installation](#installation); running the experiments and the application will be documented once they exist.

The full assignment (in Portuguese) is in [docs/assignment.md](docs/assignment.md).

| Part | Goal | Deliverable |
|---|---|---|
| Part 1, Experimentation | Compare models, training/fine-tuning strategies, loss functions and thresholds to find the best configuration | Code, experiments, metrics and a video presenting the results |
| Part 2, Application | Use the best Part 1 configuration in an application for enrolling, recognizing and deleting people | Working application and a demo video |

### Planned pipeline

```text
Image / camera
      ↓
Face detection
      ↓
Alignment / preprocessing
      ↓
Neural network
      ↓
Face embedding
      ↓
Comparison with enrolled people
      ↓
Threshold
      ↓
Recognized / unknown
```

A face whose similarity falls below the threshold is classified as an **unknown person**. The threshold value will be chosen experimentally in Part 1.

## Technologies

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Face embedding model | To be defined in Part 1 |
| Deep learning framework | PyTorch 2.7.1 (CUDA 12.8) |
| Application interface | To be defined in Part 2 |

## Installation

Requirements: **Python 3.11.4 or newer in the 3.11 series** (the `tarfile.extractall(filter=...)` used in [scripts/download_datasets.py](scripts/download_datasets.py) does not exist in 3.11.0 to 3.11.3) and, for training, an NVIDIA GPU with a driver that supports CUDA 12.8. Project machines: laptop with an RTX 4060 (experiments and training) and Mac with an M5 (development, no CUDA).

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install facenet-pytorch==2.6.0 --no-deps
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
pre-commit install
```

The order matters:

- **torch comes from the cu128 index**, the CUDA 12.8 build of torch 2.7.1. On macOS, which has no CUDA, install it without `--index-url` (`python -m pip install torch==2.7.1 torchvision==0.22.1`); everything else is the same;
- **facenet-pytorch is installed with `--no-deps`**, because version 2.6.0 declares `torch<2.3`. The dependencies it actually uses are pinned in [requirements.txt](requirements.txt);
- `pip` will warn that facenet-pytorch asks for other versions of torch, numpy and Pillow. The warning is expected, and the tests below confirm the pipeline works;
- **`pip install -e .`** installs the project package (`src/facial_recognition_system`) in editable mode, so the scripts in `scripts/` can import it. `--no-deps` keeps pip from touching the versions already installed. Without this step `pytest` works, but the scripts fail with `ModuleNotFoundError: No module named 'facial_recognition_system'`.

### Privacy hook

`pre-commit install` enables, in this clone, the hooks in [.pre-commit-config.yaml](.pre-commit-config.yaml), including [scripts/check_private_files.py](scripts/check_private_files.py), which blocks photos, videos, embeddings, checkpoints and databases from any commit. It is the second barrier after `.gitignore` and must be run **once in every clone**, because Git does not version hooks. To check that it is active:

```bash
pre-commit run --all-files
```

Never use `git commit --no-verify`, which skips the hook.

### Tests

```bash
pytest              # CPU tests (default)
pytest -m gpu -s    # GPU tests; -s shows which card and CUDA version were used
```

Tests marked `gpu` are excluded from the default run and **fail** (they are not skipped) if the GPU cannot run the model. Run both commands on every new machine with an NVIDIA GPU; on the Mac, only the first one.

## Data

### Public dataset

No copy of any public dataset is versioned. `python scripts/download_datasets.py` downloads each one from the source and verifies its checksum. Details, citation and instructions in [data/README.md](data/README.md).

| Dataset | Official source | Purpose | License / terms of use | How it was used |
|---|---|---|---|---|
| LFW (Labeled Faces in the Wild) | <http://vis-www.cs.umass.edu/lfw/> (offline; downloaded from the figshare copy used by scikit-learn) | verification benchmark and fine-tuning subset | no formal license, distributed for research with a citation request; not redistributed here | aligned with MTCNN and split by identity into train, validation and test |

### Private dataset

Photos of the team members and people close to them, all over 18, collected directly by the team with a signed consent form. This dataset is **never** versioned or sent to external services; the only exception is sending the photos over WhatsApp, only for participants who authorized it in the consent form, with the chat deleted within 24 hours (see [Privacy and LGPD](#privacy-and-lgpd) and [data/README.md](data/README.md)).

## Experiments (Part 1)

Results involving the private dataset are reported only in aggregate, without identifying participants.

Each experiment has a YAML in [configs/](configs/) and writes its result to `experiments/results/<EXP>.json`. Embeddings (`embeddings/`) and checkpoints (`checkpoints/`) stay outside Git.

```bash
# no fine-tuning (E01, E02): only extracts the validation and test embeddings
python scripts/extract_embeddings.py --config configs/E01.yaml

# with fine-tuning (E04, E06): run --smoke first (2 short epochs, a few seconds)
python scripts/train.py --config configs/E04.yaml --smoke
python scripts/train.py --config configs/E04.yaml

# private (E07 and E08): only on the RTX 4060 laptop, with the vault open; writes only aggregated results
python scripts/preprocess_faces.py --input-dir data/private --output-dir faces/private
python scripts/run_open_set.py --config configs/E07_E08.yaml
```

| Experiment | Model | Strategy | Dataset | Accuracy | F1 | Notes |
|---|---|---|---|---|---|---|
| To be filled | | | | | | |

### Data preparation

Every experiment uses the same detection pipeline (D3): MTCNN from `facenet-pytorch`, 160×160 crop, minimum probability 0.9. The order to rebuild the public data from scratch:

```bash
python scripts/download_datasets.py                       # LFW from the source, SHA-256 checked
python scripts/preprocess_faces.py --input-dir data/public/lfw --output-dir data/public/lfw_crop160 --selection center
python scripts/make_splits.py --originals-dir data/public/lfw --crops-dir data/public/lfw_crop160 --output-dir data/splits
python scripts/check_leakage.py --originals-dir data/public/lfw --crops-dir data/public/lfw_crop160 --splits-dir data/splits
```

The splits are already versioned in `data/splits/`, so the last two commands are only needed to audit or regenerate them. In LFW, 1 of the 13,233 photos had no face above 0.9 and left the splits. Details of each step in [data/README.md](data/README.md).

Fine-tuning (E04 and E06) uses the train split only: identities with at least 4 photos, at most 40 per identity, which gives **433 identities and 3,961 photos** (D2).

### Evaluation protocol

**Public (E01 to E06), LFW pair verification** (D9):

- Split **by identity**, 70/15/15 with seed 42, stratified by photo count (2–4, 5–9 and 10+). No person appears in two splits, so fine-tuning never sees anyone in validation or test.
- Pairs: every combination of two photos of the same person (up to 20 per identity) and the same number of negative pairs. Every experiment uses **the same pairs**.
- The official LFW pairs (View 2) are not used, because they include identities that enter the fine-tuning train set.
- Leakage: a two-stage dHash (originals and crops) found 2 photos repeated across splits, which were removed.

| Split | Identities | Photos | Pairs (positive + negative) |
|---|---|---|---|
| train | 1,176 | 6,439 | — |
| validation | 2,286 | 3,489 | 1,447 + 1,447 |
| test | 2,285 | 3,302 | 1,445 + 1,443 |

**Private (E07 and E08), open-set identification** (D9):

- Session 1 of each participant is the gallery (enrollment). Session 2 photos go half to validation and half to test.
- 2 participants stay out of the gallery as unknowns, one in validation and one in test, with all their photos.
- Each split also gets 200 faces of single-photo LFW identities, so there are enough unknown attempts to measure a 1% FAR.
- The same leakage check runs between session 1 and session 2.

**Rules for everything:** the test set never chooses anything (section 29). The best model of E01 to E06, which goes on to E07, is chosen by **validation EER**. The E07 strategy and the E08 threshold are chosen on private validation. The test set is run once per experiment, after everything is fixed.

### Metrics

**Verification** (E01 to E06, D7):

- ROC-AUC, EER and TAR@FAR=1%, which do not depend on a threshold;
- accuracy and F1 at the threshold chosen on validation;
- similarity histograms of same-person and different-people pairs (`experiments/figures/<EXP>_hist.png`) and ROC curves (`<EXP>_roc.png`, `comparison_roc.png`);
- **95% bootstrap confidence interval**, with 1,000 resamples of the pairs, positive and negative pairs resampled separately.

**Open-set identification** (E07 and E08, D7):

- accuracy and macro F1, with "unknown" counted as a class;
- FAR (unknowns accepted as someone), FRR (known people rejected) and misidentification rate (known person assigned to the wrong identity);
- aggregated confusion matrix, with anonymous IDs only (`experiments/figures/E08_confusion.png`).

### Threshold

A face is accepted when `similarity >= threshold`, and the threshold is always chosen **on validation** and applied to test **without readjusting** (D8).

| Where | Criterion | Value |
|---|---|---|
| Public (E01 to E06) | EER threshold on the validation pairs, where FAR = FRR. The neutral criterion to compare models | _to be filled_ |
| Private and application (E08) | smallest threshold with **FAR ≤ 1%** on validation. If the validation FRR goes above 20%, the EER threshold is used | _to be filled_ |

A false accept weighs more than a false reject: recognizing a stranger as an enrolled person is worse than asking someone to try again. With about 200 unknown attempts per split, the private FAR has a 0.5% resolution (1 acceptance in 200), which is why the target is 1% and not 0.1%. In E07, each comparison strategy has its own threshold, because the scores are not on the same scale (similarity for cosine, prototype and k-NN, probability for the SVM).

### Reproducibility

- **Configuration:** one YAML per experiment in [configs/](configs/), with every hyperparameter written out (D11). The result goes to `experiments/results/<EXP>.json`, with the hypothesis, a copy of the YAML, seed, library versions, hardware, timings, threshold and metrics.
- **Seed 42** in Python, NumPy and PyTorch, with cuDNN autotuning off. The split, fine-tuning subset and DataLoader draws also use the seed. One seed per training run; variation is measured by the bootstrap.
- **Versions:** Python 3.11, PyTorch 2.7.1 with CUDA 12.8, facenet-pytorch 2.6.0, scikit-learn 1.9.1, all pinned in [requirements.txt](requirements.txt). Each experiment's JSON records the versions it ran with.
- **Hardware:** every experiment runs on the RTX 4060 (8 GB) laptop, which keeps training times comparable (D10). Results are repeatable on the same machine, but not bit-identical on another GPU or CUDA version.

Command of each experiment:

| Experiment | What it tests | Command |
|---|---|---|
| E01 | Pretrained FaceNet (VGGFace2), no fine-tuning | `python scripts/extract_embeddings.py --config configs/E01.yaml` |
| E02 | ResNet-50 (ImageNet), no fine-tuning | `python scripts/extract_embeddings.py --config configs/E02.yaml` |
| E04 | FaceNet with partial fine-tuning and ArcFace | `python scripts/train.py --config configs/E04.yaml` |
| E06 | FaceNet with partial fine-tuning and Triplet Loss | `python scripts/train.py --config configs/E06.yaml` |
| Evaluation of E01 to E06 | threshold on validation, metrics with CI and plots | `python scripts/evaluate.py --config configs/E01.yaml configs/E02.yaml configs/E04.yaml configs/E06.yaml` |
| E07 and E08 | comparison strategy and open-set threshold, on private data | `python scripts/run_open_set.py --config configs/E07_E08.yaml` |

E03 and E05 (full fine-tuning of ResNet-50) were cut for lack of time (D4 and D5). E07 and E08 only run on the group's machine, with the vault open and the private crops generated first (`python scripts/preprocess_faces.py --input-dir data/private --output-dir faces/private`).

## Privacy and LGPD

Biometric data is sensitive personal data under Brazil's LGPD (Law No. 13,709/2018). Breaking the rules below results in a zero grade.

The following stays **out of GitHub** and is processed only locally:

- photos and videos of participants;
- names and identifiers linked to the photos;
- face embeddings and vector databases;
- checkpoints trained or fine-tuned on private data;
- consent forms;
- logs and individual-level results.

All of this lives in a VeraCrypt vault on the RTX 4060 laptop, linked to the project through junctions at `data/private/`, `faces/private/`, `embeddings/private/` and `checkpoints/private/` (decision D10 in [docs/division.md](docs/division.md)).

The [.gitignore](.gitignore) ignores `data/private/`, `data/public/`, `faces/`, `embeddings/`, `checkpoints/`, `consent/`, databases (`*.db`, `*.sqlite`), model weights (`*.pt`, `*.pth`, `*.onnx`, `*.safetensors`) and saved arrays or classifiers (`*.npy`, `*.npz`, `*.pkl`, `*.joblib`). As a second barrier, the [scripts/check_private_files.py](scripts/check_private_files.py) pre-commit hook blocks photos, videos, checkpoints and databases in any folder, even when added with `git add -f`, and only accepts PDF, Word and Excel files in `docs/`, so signed consent forms and the control spreadsheet are not committed by mistake. The application will provide a way to delete all data of an enrolled person, and private data will be deleted at the end of the course.

How each piece of data is collected, stored, deleted and destroyed, and the Annex B checklist map, are in [docs/privacy.md](docs/privacy.md) (in Portuguese). Full rules in sections 12 to 25 and Annex B (delivery checklist) of [docs/assignment.md](docs/assignment.md).

## Folder structure

```text
facial_recognition_system/
├── configs/                      # one YAML per experiment
├── data/
│   ├── README.md                 # data source, license and how to obtain it
│   └── splits/                   # train/validation/test lists (public data only)
├── docs/
│   ├── assignment.md             # assignment description
│   ├── decisions/                # ADRs
│   └── specs/                    # specifications with acceptance criteria
├── experiments/
│   ├── figures/                  # aggregated plots
│   └── results/                  # one result JSON per experiment
├── scripts/
│   └── check_private_files.py    # pre-commit hook against private data
├── src/facial_recognition_system/
│   ├── data/                     # identity splits and verification pairs
│   ├── evaluation/               # metrics, threshold and plots
│   ├── models/                   # embedding model wrappers
│   ├── preprocessing/            # detection, alignment and cropping
│   └── training/                 # fine-tuning loops and loss heads
├── tests/
├── .gitignore
├── .pre-commit-config.yaml
├── pyproject.toml                # ruff and pytest configuration
├── requirements.txt              # pinned versions (except torch and facenet-pytorch)
└── README.md
```

## Commit types

| Type | Description |
|---|---|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation change |
| `test` | Adding or adjusting tests |
| `refactor` | Refactoring without behavior change |
| `perf` | Performance improvement |
| `chore` | Maintenance tasks (dependencies, configuration) |
| `ci` | Continuous integration changes |

</details>
