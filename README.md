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
- [Conjuntos de dados](#conjuntos-de-dados)
  - [Dataset público](#dataset-público)
  - [Dataset privado](#dataset-privado)
- [Experimentos (Parte 1)](#experimentos-parte-1)
- [Privacidade e LGPD](#privacidade-e-lgpd)
- [Estrutura de pastas](#estrutura-de-pastas)
- [Tipos de commit](#tipos-de-commit)

## Visão geral

> **Status:** em desenvolvimento. Ainda não há código; instalação, execução, variáveis de ambiente e testes serão documentados quando existirem.

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
| Linguagem | Python |
| Modelo de embedding facial | A definir na Parte 1 |
| Framework de deep learning | A definir na Parte 1 |
| Interface da aplicação | A definir na Parte 2 |

## Conjuntos de dados

### Dataset público

Nenhuma cópia de dataset público é versionada. O repositório trará instruções para baixar cada um diretamente da fonte oficial.

| Dataset | Fonte oficial | Finalidade | Licença / termos de uso | Como foi usado |
|---|---|---|---|---|
| A definir | | | | |

### Dataset privado

Fotos dos integrantes e de pessoas próximas, maiores de 18 anos, coletadas diretamente pela dupla e com Termo de Consentimento assinado. Esse dataset **nunca** é versionado nem enviado a serviços externos (ver [Privacidade e LGPD](#privacidade-e-lgpd)).

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

O [.gitignore](.gitignore) já ignora `data/private/`, `faces/`, `embeddings/`, `checkpoints/private/`, `consent/`, `*.db` e `*.sqlite`. A aplicação terá um mecanismo para excluir todos os dados de uma pessoa cadastrada, e os dados privados serão eliminados ao fim da disciplina.

Regras completas nas seções 12 a 25 e no Anexo B (checklist de entrega) de [docs/assignment.md](docs/assignment.md).

## Estrutura de pastas

```text
facial_recognition_system/
├── docs/
│   └── assignment.md   # enunciado do trabalho
├── .gitignore
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
- [Data](#data)
  - [Public dataset](#public-dataset)
  - [Private dataset](#private-dataset)
- [Experiments (Part 1)](#experiments-part-1)
- [Privacy and LGPD](#privacy-and-lgpd)
- [Folder structure](#folder-structure)
- [Commit types](#commit-types)

## Overview

> **Status:** in development. There is no code yet; installation, running, environment variables and tests will be documented once they exist.

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
| Language | Python |
| Face embedding model | To be defined in Part 1 |
| Deep learning framework | To be defined in Part 1 |
| Application interface | To be defined in Part 2 |

## Data

### Public dataset

No copy of any public dataset is versioned. The repository will include instructions to download each one directly from its official source.

| Dataset | Official source | Purpose | License / terms of use | How it was used |
|---|---|---|---|---|
| To be defined | | | | |

### Private dataset

Photos of the team members and people close to them, all over 18, collected directly by the team with a signed consent form. This dataset is **never** versioned or sent to external services (see [Privacy and LGPD](#privacy-and-lgpd)).

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

The [.gitignore](.gitignore) already ignores `data/private/`, `faces/`, `embeddings/`, `checkpoints/private/`, `consent/`, `*.db` and `*.sqlite`. The application will provide a way to delete all data of an enrolled person, and private data will be deleted at the end of the course.

Full rules in sections 12 to 25 and Annex B (delivery checklist) of [docs/assignment.md](docs/assignment.md).

## Folder structure

```text
facial_recognition_system/
├── docs/
│   └── assignment.md   # assignment description
├── .gitignore
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
