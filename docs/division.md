# Divisão do trabalho de Reconhecimento Facial (Paola e Valentina)

Prazos do Moodle: **Parte 1 vence dom 25/10/2026 23:59** e **Parte 2 vence dom 15/11/2026 23:59**.
Cada dia de atraso custa 1 ponto (máximo de 2 dias), então a meta interna é entregar no **sábado anterior** (24/10 e 14/11).

O enunciado pede duas entregas sequenciais (Parte 1: experimentação; Parte 2: aplicação), cada uma com repositório GitHub e vídeo, seguindo Spec-Driven Development e com LGPD como critério eliminatório.

## Princípio da divisão

Cada uma é dona de uma fatia vertical (escreve a spec/ADR, implementa, testa e documenta), e a outra revisa o PR. Isso cumpre a exigência de que as duas saibam explicar tudo (seção 5) e gera revisão de código visível no histórico (seção 41).

- **Paola:** dados, pipeline de pré-processamento, avaliação, métricas e threshold, núcleo da aplicação (cadastro, reconhecimento, exclusão, armazenamento).
- **Valentina:** ambiente, modelos e treinamento, análise dos resultados, interface/câmera, vídeos.
- **Juntas:** LGPD e consentimento, coleta do dataset privado, decisões de arquitetura, revisão cruzada, checklist final.

## Situação em 18/10

- Repositório com 2 commits: enunciado (`main`) e README inicial (`dev`, que só entra na `main` na entrega).
- O `.gitignore` de privacidade já existe, mas ainda falta ignorar `*.pt`, `*.pth`, `*.onnx` e `data/public/`.
- Ainda não existem `src/`, `configs/`, specs, ADRs nem `requirements.txt`.
- Hardware: o código é escrito e testado nos notebooks (RTX 4060, 8 GB) e o treino completo roda na **RTX 5070 (8 GB)**.
- Ambiente igual nas máquinas: Python 3.11.4+ (série 3.11), `torch==2.7.1` e `torchvision==0.22.1` com CUDA 12.8 (cu128), `facenet-pytorch==2.6.0` instalado com `--no-deps`.

---

## Cronograma da Parte 1 (dom 18/10 a sáb 24/10)

| Dia | Paola | Valentina | Juntas |
|---|---|---|---|
| **dom 18** | Criar a estrutura do repo (branch `chore/repo-setup` saindo da `dev`), `pyproject`, `ruff`/`pytest`, completar o `.gitignore` e o hook de pre-commit | Enviar o termo de consentimento (Anexo A) às participantes; montar o venv, o `requirements.txt` e o teste de ambiente `tests/test_environment.py`; validar o ambiente na 4060 e na 5070 com `pytest` e `pytest -m gpu -s` | Reunião de 45 min: fechar D1 a D15 aceitando as recomendações, o protocolo de avaliação, o split do privado e o contrato entre as fatias (seções abaixo) |
| **seg 19** | Download do LFW, pré-processamento com MTCNN, splits por identidade e pares de teste comuns; `data/README.md` com origem, licença e uso do LFW | Wrappers dos modelos; **E01** (baseline FaceNet) e **E02** (ResNet-50 sem fine-tuning) salvando embeddings no formato combinado | **Sessão 1** da coleta privada, com os termos assinados |
| **ter 20** | Métricas, escolha do threshold na validação, checagem de leakage; ADRs de D1 a D3 e D6 a D9 | Loop de treino com cabeça ArcFace, congelamento parcial e modo `--smoke`; **E04** na 5070 (early stopping por val loss); ADRs de D4, D5, D10 e D11 | |
| **qua 21** | Avaliação de E01, E02 e E04, gráficos agregados; ADR de D12 | **E06** (Triplet Loss) na 5070; `docs/privacy.md`; ADR de D13 | **Sessão 2** da coleta (outro cômodo ou outro fundo, pode ser no mesmo dia da sessão 1); juntar as fotos privadas na máquina escolhida, por pendrive; ADRs de D14 e D15 |
| **qui 22** | Avaliação de E06; MTCNN nas fotos privadas; **E07 e E08** com todas as participantes, localmente | Tabela de experimentos e análise de E01 a E06 (seção 33); acompanhar E07 e E08 para saber explicar | ADR "modelo escolhido" |
| **sex 23** | README: dados, protocolo, métricas, threshold e reprodutibilidade; gravar a sua parte do vídeo à noite | README: tabela final e análise da seção 33 incluindo E07 e E08; roteiro do vídeo | Revisão cruzada do README |
| **sáb 24** | Rodar o Anexo B item por item | Gravar a sua parte do vídeo e editar o vídeo 1 | Auditoria do histórico (`git log --all --stat`), juntar `dev` na `main`, **entrega** |
| dom 25 | | | Folga para imprevistos |

## Cronograma da Parte 2 (26/10 a 14/11)

| Semana | Paola | Valentina | Juntas |
|---|---|---|---|
| 26/10 a 01/11 | specs 001 a 004 e 006 com critérios de aceitação (até qua 28/10), storage SQLite, `EnrollmentService` | specs 005 e 007 (até qua 28/10), esqueleto da UI em Streamlit com serviços fake | Revisão cruzada das specs antes de codar |
| 02/11 a 08/11 | `RecognitionService`, `DeletionService`, exceções, testes unitários e de integração | Config externa, logs, integração da UI com os serviços reais e o modelo escolhido | |
| 09/11 a 14/11 | README da Parte 2 com arquitetura; explicação da arquitetura no vídeo | Teste e2e, gravação e edição do vídeo 2 (até sex 13/11) | Checklist do Anexo B, auditoria do histórico, juntar `dev` na `main`, **entrega sáb 14/11** |
| dom 15/11 | | | Folga para imprevistos |

Nada impede adiantar as specs da Parte 2 durante a Parte 1 se sobrar tempo.

---

## Decisões (D1 a D15)

Fechadas na reunião de domingo, cada uma registrada como ADR em `docs/decisions/`. Dona = quem escreve o ADR.

| # | Decisão | Escolha | Dona |
|---|---|---|---|
| D1 | Dataset público de benchmark | **LFW** (licença de pesquisa, baixado pelo script da fonte oficial) | Paola |
| D2 | Dataset público para fine-tuning | **Subconjunto do LFW** com split por identidade, identidades com pelo menos N imagens. O CelebA ficou de fora por falta de tempo para baixar e preparar; registrar no README como limitação | Paola |
| D3 | Detector e alinhamento | **MTCNN** (`facenet-pytorch`): detecta, devolve landmarks e recorta 160×160, tudo em PyTorch | Paola |
| D4 | Backbones comparados | **A:** FaceNet InceptionResnetV1 pré-treinado em VGGFace2 (`facenet-pytorch`, fácil de fine-tunar). **B:** ResNet-50 ImageNet (genérico, sem pré-treino facial), para medir o valor do pré-treino específico | Valentina |
| D5 | Funções de perda | ArcFace Loss vs Triplet Loss | Valentina |
| D6 | Estratégia de comparação | Similaridade de cosseno vs protótipo por pessoa (média dos embeddings) vs k-NN vs SVM | Paola |
| D7 | Métricas | Verificação: ROC-AUC, EER, TAR@FAR=1%, distribuição mesma pessoa vs diferentes. Identificação open-set: accuracy, F1 macro, FAR, FRR, matriz de confusão | Paola |
| D8 | Escolha do threshold | Escolhido **só na validação**, por EER ou FAR alvo; reportado no teste sem reajuste | Paola |
| D9 | Protocolo de split | Ver "Protocolo de avaliação" abaixo. Dedupe por hash perceptual | Paola |
| D10 | Hardware | Código e `--smoke` nos notebooks (4060), treino completo na **5070**. **Dados privados só nos computadores do grupo** (seção 23) | Valentina |
| D11 | Registro de experimentos | Config YAML por experimento em `configs/` e resultado JSON em `experiments/results/` (seed, bibliotecas, hiperparâmetros, hardware). Sem MLflow, para manter simples | Valentina |
| D12 | Armazenamento de embeddings | SQLite (pessoa, metadados) com embeddings como BLOB numpy, tudo em `data/private/` (no `.gitignore`). Exclusão em cascata | Paola |
| D13 | Interface | **Streamlit** (`st.camera_input` e upload), tudo em Python, uma tela para cadastro, reconhecimento e exclusão | Valentina |
| D14 | Repositório | Público no GitHub. `dev` é a branch de integração: cada tarefa ou spec sai da `dev` numa branch própria e volta por PR revisado pela outra. A `dev` entra na `main` só nas entregas (24/10 e 14/11), então a `main` é sempre "o que foi entregue" | Juntas |
| D15 | Quem aparece no vídeo | Só Paola e Valentina (evita depender da autorização de vídeo de terceiros) | Juntas |

## Protocolo de avaliação

- **Público:** o LFW é dividido por identidade em treino, validação e teste (nenhuma pessoa aparece em mais de um split). E01, E02, E04 e E06 são avaliados **nos mesmos pares de teste**, montados só com identidades do split de teste. Os pares padrão do LFW (View 2) não são usados, porque contêm identidades que entram no treino do fine-tuning.
- **Privado:** a sessão 1 é a galeria (cadastro e treino do k-NN/SVM). As fotos da sessão 2 são divididas por pessoa, metade para validação e metade para teste. Pelo menos 2 participantes ficam fora da galeria como desconhecidas, uma na validação e outra no teste, complementadas por identidades do teste do LFW como desconhecidos extras.
- O threshold é escolhido só na validação e reportado no teste sem reajuste.
- Se o número de participantes ficar muito pequeno, registrar no README como limitação.

## Contrato entre as fatias

- **Splits:** listas de caminhos relativos em `data/splits/` (só dados públicos são versionados, nunca caminhos do privado).
- **Embeddings:** um arquivo por experimento e split, com os embeddings L2-normalizados de 512 dimensões e os rótulos de identidade, fora do Git.
- **Resultados:** um JSON por experimento em `experiments/results/`, com seed, versões das bibliotecas, hiperparâmetros, hardware e métricas.
- Com isso, E01 e E02 rodam na segunda só gerando embeddings, e a avaliação roda depois em cima deles, quando as métricas estiverem prontas.

---

## Fase 0, fundação (dom 18)

| Tarefa | Dona |
|---|---|
| Estrutura do repo (`src/`, `experiments/`, `configs/`, `tests/`, `scripts/`, `docs/specs/`, `docs/decisions/`, `data/README.md`), `pyproject.toml`, ruff, pytest com marcadores `unit`/`integration`/`e2e`/`gpu`, pre-commit | Paola |
| `.gitignore` de privacidade (`data/private/`, `data/public/`, `faces/`, `embeddings/`, `checkpoints/`, `*.db`, `*.sqlite`, `consent/`, `*.pt`, `*.pth`, `*.onnx`) e hook de pre-commit que bloqueia imagens e checkpoints fora das pastas permitidas | Paola |
| venv, `requirements.txt` com versões fixadas e `tests/test_environment.py` (CPU por padrão, GPU com `pytest -m gpu -s`), validados na 4060 e na 5070 | Valentina |
| Adaptar o Termo de Consentimento (Anexo A), enviar e colher assinaturas; guardar fora do repo (pasta local criptografada ou papel) | Valentina |
| Ler o enunciado inteiro e fechar as decisões D1 a D15 | Juntas |

## Parte 1, experimentação

### Dados (Paola)
- Script `scripts/download_datasets.py` que baixa o LFW da fonte oficial; `data/README.md` com nome, URL, finalidade, licença e como foi usado (seção 10).
- Pipeline `src/preprocessing/`: detecção MTCNN, alinhamento por landmarks, recorte, normalização. Testes unitários com imagens sintéticas (sem face, múltiplas faces).
- Splits de treino, validação e teste por identidade e pares de teste comuns, salvos como listas de arquivos versionadas.
- Checagem de leakage: hash perceptual entre splits e garantia de que as augmentations só acontecem dentro do treino. Script e relatório.

### Dataset privado (juntas, execução local)
- Protocolo de coleta: cerca de 5 a 10 adultos consentidos, 20 a 30 fotos cada, em 2 sessões em cômodos ou com fundos diferentes, podendo ser no mesmo dia (variando iluminação, pose, distância, óculos e fundo). Minimização de dados.
- Cada uma coleta as pessoas do seu lado e guarda os termos delas. Depois as fotos são juntadas numa só máquina de uma de nós (passadas por pendrive, nunca por nuvem ou mensageiro), e E07 e E08 rodam uma vez com o conjunto inteiro.
- `data/README.md` com instruções para alguém criar o próprio dataset privado localmente.

### Modelos e treinamento (Valentina)
- `src/models/`: wrappers dos backbones A e B devolvendo embedding L2-normalizado de 512 dimensões.
- `src/training/`: loop de treino com cabeça ArcFace e Triplet Loss, congelamento parcial configurável, early stopping na validação, seed fixa, modo `--smoke`.
- Checkpoints de experimentos com dados privados ficam em `checkpoints/private/`, fora do Git.

Cada experimento tem hipótese, configuração, resultado e interpretação (seção 27):

| ID | Modelo | Estratégia | Dataset | Hipótese |
|---|---|---|---|---|
| E01 | FaceNet (VGGFace2) | pré-treinado, sem fine-tuning | LFW (teste comum) | baseline forte |
| E02 | ResNet-50 ImageNet | sem fine-tuning | LFW (teste comum) | o pré-treino genérico é ruim para faces |
| E04 | FaceNet | fine-tuning parcial + ArcFace | subconjunto do LFW | o fine-tuning parcial melhora sem overfit |
| E06 | FaceNet | fine-tuning parcial + Triplet | subconjunto do LFW | ArcFace vs Triplet |
| E07 | melhor de E01 a E06 | cosseno, k-NN, SVM e protótipo | privado | qual estratégia de comparação |
| E08 | melhor configuração | estudo de threshold (open-set) | privado | qual threshold usar na aplicação |

**Cortes de escopo:** E03 (ResNet-50 com fine-tuning completo + ArcFace) e E05 (FaceNet com fine-tuning completo + ArcFace) ficam de fora e só voltam se sobrar tempo. Os experimentos mantidos ainda cobrem o impacto do pré-treino facial, do fine-tuning, da loss e do threshold.

### Avaliação (Paola)
- `src/evaluation/`: cálculo de todas as métricas de D7, curva ROC, histogramas de similaridade mesma pessoa vs diferentes, matriz de confusão, escolha do threshold na validação.
- Saída só agregada para o privado (sem nomes, sem imagens por pessoa, IDs anônimos como `P01`).
- Testes unitários das métricas com casos conhecidos (ex.: EER de distribuições sintéticas).

### Análise e entrega da Parte 1
- README com a tabela de experimentos (seção 35), gráficos agregados e a discussão da seção 33 (melhor modelo e por quê, impacto do fine-tuning e do threshold, erros, limitações, custo computacional vs desempenho). Valentina escreve a análise e a tabela; Paola escreve dados, protocolo, métricas, threshold e reprodutibilidade.
- Seção de reprodutibilidade: seed, versões, hardware e o comando para rodar cada experimento.
- ADR final "modelo escolhido" ligando o resultado à decisão.
- Vídeo da Parte 1: roteiro e edição da Valentina; cada uma apresenta a parte que fez (Paola: dados, protocolo, métricas, threshold; Valentina: arquitetura, treino, resultados, escolha).

## Parte 2, aplicação

Specs em `docs/specs/`, com critérios de aceitação antes de implementar:

| Spec | Conteúdo | Dona |
|---|---|---|
| 001-face-detection | detecção, alinhamento, erros (sem face, múltiplas faces, imagem inadequada) | Paola |
| 002-person-enrollment | cadastro, várias fotos por pessoa, cadastro duplicado (por nome e por similaridade) | Paola |
| 003-face-recognition | embedding, comparação, threshold vindo da config | Paola |
| 004-unknown-person | comportamento abaixo do threshold | Paola |
| 005-camera-interface | telas, fluxos, mensagens em português | Valentina |
| 006-person-deletion | exclusão em cascata (registro, embeddings, fotos, cache) | Paola |
| 007-logging-config | config externa (YAML/env: threshold, caminhos, modelo), logs sem dados pessoais | Valentina |

### Núcleo (Paola)
- `src/app/core/`: serviços `EnrollmentService`, `RecognitionService` e `DeletionService`, com modelos Pydantic de entrada e saída e exceções específicas (`NoFaceDetectedError`, `MultipleFacesError`, `DuplicatePersonError`, `PersonNotFoundError`).
- `src/app/storage/`: repositório SQLite, criado vazio na primeira execução ("Nenhuma pessoa cadastrada", seção 43).
- Testes unitários com detector e modelo fake injetados, cobrindo os caminhos de erro; testes de integração com o modelo real em imagens públicas.

### Interface e integração (Valentina)
- `src/app/ui/`: Streamlit com abas Cadastro, Reconhecimento (foto e câmera) e Pessoas cadastradas (listar e excluir).
- Tratamento de erro na borda (UI), com mensagens amigáveis em português.
- Carregar o modelo escolhido na Parte 1 (pré-treinado por referência; se for checkpoint fine-tunado com dado privado, carregar de caminho local não versionado e documentar).
- Teste e2e do fluxo cadastro → reconhecimento → desconhecido → exclusão.

### Entrega da Parte 2
- README com instalação, execução, arquitetura (diagrama) e como criar o próprio banco local. Paola.
- Vídeo da Parte 2 (inicialização, cadastro, reconhecimento, desconhecido, erro, arquitetura), só com as duas na câmera. Valentina grava e edita, Paola explica a arquitetura.

---

## Caminho crítico e cuidados

- **Coleta privada:** as sessões 1 e 2 podem ser no mesmo dia, desde que em cômodos ou com fundos diferentes, para o split por sessão funcionar. Os termos têm que estar assinados até segunda (19/10).
- **Fluxo entre máquinas:** código pelo Git; cada máquina tem o próprio venv e baixa o LFW sozinha; rodar o `--smoke` no notebook antes de mandar o treino para a 5070; os resultados JSON e os gráficos agregados voltam pelo Git; checkpoints ficam fora do Git.
- **LGPD:** as fotos privadas só são processadas nos computadores do grupo, e agentes de IA nunca recebem fotos reais. Se a 5070 não for de uma das integrantes, ela roda apenas E01 a E06 (dados públicos), e E07 e E08 rodam nos notebooks.

## Fechamento de cada entrega (juntas)

- Rodar `pytest` e `ruff`, reproduzir um experimento do zero a partir do README e conferir o Anexo B item por item.
- Auditoria do histórico Git (`git log --all --stat` procurando imagens, `.db` e `.pt`).
- Ensaio cruzado: Paola explica treino e losses, Valentina explica protocolo, métricas e threshold (a seção 5 exige que as duas expliquem tudo).
- Após a disciplina: excluir fotos, embeddings, banco e checkpoints privados.

## Organização

- Toda tarefa vira uma issue no GitHub com dona e link para a spec; cada PR é revisado pela outra.
- As issues estão nos milestones **Parte 1** (24/10) e **Parte 2** (14/11), com o prazo do dia no corpo e labels por área. A branch sugerida está em cada issue.
- Fluxo de cada tarefa:
  1. Criar a branch ligada à issue, saindo da `dev`: `gh issue develop <N> --base dev --name <branch> --checkout`.
  2. Commitar normalmente na branch.
  3. Abrir o PR: `gh pr create --base dev --reviewer <a outra>`, com `Closes #<N>` no corpo (o template já traz).
  4. A outra revisa e faz o merge. O workflow `close-issues-on-dev.yml` fecha a issue (o `Closes` nativo só vale para a `main`) e o GitHub apaga a branch remota. Local: `git switch dev && git pull && git branch -d <branch>`.
- Tarefas com a label `sem-codigo` (coleta, termo, vídeo, entrega) fecham à mão: `gh issue close <N> --comment "..."`.

## Resumo da carga

| | Paola | Valentina |
|---|---|---|
| Fundação | repo, tooling, `.gitignore`, hook | ambiente, termo de consentimento, `privacy.md` |
| Parte 1 | dataset, pré-processamento, splits, leakage, métricas, threshold, E07 e E08, README de dados e reprodutibilidade | modelos, treino, E01 a E06, análise, README de resultados, vídeo 1 |
| Parte 2 | specs 001 a 004 e 006, núcleo, storage, exclusão, testes, README | specs 005 e 007, interface, config e logs, integração, e2e, vídeo 2 |
| Juntas | decisões e ADRs, coleta privada, revisão cruzada de PRs, checklist LGPD, ensaio | |

## Ponto em aberto

- De quem é a 5070 (define se ela pode processar as fotos privadas).
