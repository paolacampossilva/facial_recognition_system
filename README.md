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

| Experimento | Modelo | Estratégia | Dataset | Accuracy | F1 | Observação |
|---|---|---|---|---|---|---|
| A preencher | | | | | | |

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

Regras completas nas seções 12 a 25 e no Anexo B (checklist de entrega) de [docs/assignment.md](docs/assignment.md).

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

| Experiment | Model | Strategy | Dataset | Accuracy | F1 | Notes |
|---|---|---|---|---|---|---|
| To be filled | | | | | | |

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

Full rules in sections 12 to 25 and Annex B (delivery checklist) of [docs/assignment.md](docs/assignment.md).

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
