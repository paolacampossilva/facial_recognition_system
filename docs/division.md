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
- Hardware: todos os experimentos rodam no **notebook da Valentina (RTX 4060, 8 GB)**, que só ela acessa (D10). A Paola desenvolve no Mac (M5), sem CUDA.
- Ambiente igual nas máquinas: Python 3.11.4+ (série 3.11), `torch==2.7.1` e `torchvision==0.22.1` com CUDA 12.8 (cu128), `facenet-pytorch==2.6.0` instalado com `--no-deps`.

---

## Cronograma da Parte 1 (dom 18/10 a sáb 24/10)

| Dia | Paola | Valentina | Juntas |
|---|---|---|---|
| **dom 18** | Criar a estrutura do repo (branch `chore/repo-setup` saindo da `dev`), `pyproject`, `ruff`/`pytest`, completar o `.gitignore` e o hook de pre-commit | Enviar o termo de consentimento (Anexo A) às participantes; montar o venv, o `requirements.txt` e o teste de ambiente `tests/test_environment.py`; validar o ambiente na 4060 com `pytest` e `pytest -m gpu -s` | Reunião de 45 min: fechar D1 a D15 aceitando as recomendações, o protocolo de avaliação, o split do privado e o contrato entre as fatias (seções abaixo) |
| **seg 19** | Download do LFW, pré-processamento com MTCNN, splits por identidade e pares de teste comuns; `data/README.md` com origem, licença e uso do LFW | Wrappers dos modelos; **E01** (baseline FaceNet) e **E02** (ResNet-50 sem fine-tuning) salvando embeddings no formato combinado | **Sessão 1** da coleta privada, com os termos assinados |
| **ter 20** | Métricas, escolha do threshold na validação, checagem de leakage; ADRs de D1 a D3 e D6 a D9 | Loop de treino com cabeça ArcFace, congelamento parcial e modo `--smoke`; **E04** na 4060 (early stopping pelo EER de validação); ADRs de D4, D5, D10 e D11 | |
| **qua 21** | Avaliação de E01, E02 e E04, gráficos agregados; ADR de D12 | **E06** (Triplet Loss) na 4060; `docs/privacy.md`; ADR de D13 | **Sessão 2** da coleta (outro cômodo ou outro fundo, pode ser no mesmo dia da sessão 1); juntar as fotos privadas no notebook da 4060, por pendrive dentro de um cofre VeraCrypt; ADRs de D14 e D15 |
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

Fechadas na reunião de domingo (issue #6), cada uma registrada como ADR em `docs/decisions/`. Dona = quem escreve o ADR. Os valores numéricos são o ponto de partida; se um deles mudar durante a implementação, a mudança vai no ADR com o motivo.

| # | Decisão | Escolha | Dona |
|---|---|---|---|
| D1 | Dataset público de benchmark | **LFW**, versão original (sem alinhamento), baixada pelo `scripts/download_datasets.py` da cópia no figshare (a mesma do scikit-learn), conferida por SHA-256, porque o site oficial saiu do ar. Nada é redistribuído | Paola |
| D2 | Dataset público para fine-tuning | **Identidades do split de treino do LFW com pelo menos 4 fotos** (N = 4), no máximo 40 fotos por identidade para o George W. Bush e afins não dominarem. ArcFace e Triplet treinam exatamente no mesmo conjunto, para a comparação ser controlada. O CelebA ficou de fora por falta de tempo para baixar e preparar; registrar no README como limitação | Paola |
| D3 | Detector e alinhamento | **MTCNN** (`facenet-pytorch`), recorte 160×160 com `--alignment crop` e margem 0 (a entrada com que o FaceNet pré-treinado foi treinado), probabilidade mínima 0.9. LFW com `--selection center`; privado e aplicação com `strict`. Fotos sem rosto ficam fora dos splits e listadas no `preprocessing_report.json`. O alinhamento por similaridade fica disponível, mas não entra nos experimentos | Paola |
| D4 | Backbones comparados | **A:** FaceNet InceptionResnetV1 pré-treinado em VGGFace2 (`facenet-pytorch`), embedding de 512 dimensões, entrada com `fixed_image_standardization`. **B:** ResNet-50 do torchvision com pesos `IMAGENET1K_V2`, sem a camada `fc`, embedding de 2048 dimensões (saída do pooling global, sem projeção aleatória), entrada redimensionada para 224×224 com média e desvio do ImageNet. B mede o valor do pré-treino facial. Limitação a registrar: o VGGFace2 pode conter celebridades que também estão no LFW, o que favorece E01 | Valentina |
| D5 | Funções de perda | **ArcFace** (s = 32, m = 0.5) vs **Triplet** (margem 0.2, *batch-hard mining*, amostragem P×K com P = 32 identidades e K = 4 fotos). Configuração comum às duas: só `block8`, `last_linear` e `last_bn` do FaceNet treinam (o resto congelado), AdamW com lr 1e-4 no backbone e 1e-3 na cabeça ArcFace, weight decay 5e-4, batch 128, até 30 épocas, **early stopping pelo EER nos pares de validação** com paciência 5, flip horizontal e color jitter leve só no treino. O early stopping não usa a val loss porque as identidades da validação não existem na cabeça ArcFace (split por identidade), e a mesma métrica para as duas losses deixa a comparação justa | Valentina |
| D6 | Estratégia de comparação | Na galeria privada (E07): **cosseno com o vizinho mais próximo** (maior similaridade entre todas as fotos da galeria), **protótipo** (média dos embeddings da pessoa, renormalizada), **k-NN** (k = 3, cosseno, ponderado pela similaridade) e **SVM linear** (C escolhido na validação entre 0.1, 1 e 10, probabilidades por calibração de Platt). O escore de rejeição é a similaridade no cosseno e no protótipo, a similaridade média dos k vizinhos da classe vencedora no k-NN e a probabilidade máxima no SVM. A distância euclidiana não entra separada porque, com embeddings L2-normalizados, d² = 2 − 2·cos e as duas ordenam igual | Paola |
| D7 | Métricas | **Verificação** (pares do LFW): ROC-AUC, EER, TAR@FAR=1%, e accuracy e F1 no threshold da validação (para a tabela da seção 35), histogramas de similaridade mesma pessoa vs diferentes, intervalo de confiança de 95% por bootstrap (1000 reamostragens dos pares). **Identificação open-set** (privado): accuracy e F1 macro tratando "desconhecido" como uma classe, FAR (tentativas de desconhecidos aceitas como alguém), FRR (tentativas de conhecidos rejeitadas), taxa de identificação errada (conhecido atribuído à pessoa errada, que FAR e FRR não capturam) e matriz de confusão agregada com IDs anônimos | Paola |
| D8 | Escolha do threshold | Sempre **só na validação** e reportado no teste sem reajuste. **Público:** threshold do EER nos pares de validação. **Privado e aplicação (E08):** o menor threshold com FAR ≤ 1% na validação, reportando o FRR; se o FRR de validação passar de 20%, usar o threshold do EER e justificar no ADR. Os desconhecidos extras do LFW existem para a validação ter tentativas suficientes (cerca de 200) para medir FAR de 1% | Paola |
| D9 | Protocolo de split | Ver "Protocolo de avaliação" abaixo. Dedupe por **dHash de 64 bits** nos recortes 160×160, implementado com Pillow (sem dependência nova): distância de Hamming ≤ 4 conta como quase-duplicata. Entre splits do LFW, a cópia em validação/teste sai; no privado, a mesma checagem roda entre sessão 1 e sessão 2 (participante pode mandar a mesma foto nas duas etapas). Augmentations só no dataloader de treino | Paola |
| D10 | Hardware | **O notebook da Valentina com RTX 4060 roda tudo**: `--smoke`, E01 a E06 e, com as fotos privadas, E07 e E08. O fine-tuning da D5 é leve (só o último bloco do FaceNet treina, em algumas centenas de identidades a 160×160) e cabe nos 8 GB. Concentrar tudo numa máquina deixa um ambiente só, tempos de treino comparáveis entre os experimentos (seção 33) e os dados privados num único computador que só a Valentina acessa (seção 22). As fotos privadas e tudo que deriva delas (recortes, embeddings, resultados por pessoa) ficam num cofre VeraCrypt aberto apenas durante o uso. O Mac da Paola (M5, sem CUDA) serve para desenvolver e testar com dados sintéticos ou do LFW | Valentina |
| D11 | Registro de experimentos | Config YAML por experimento em `configs/` (`E01.yaml` ...) e resultado JSON em `experiments/results/` (formato no contrato abaixo). Seed 42 em Python, NumPy e PyTorch. Uma seed por treino, por falta de tempo; a variação é medida pelo bootstrap de D7. Sem MLflow, para manter simples | Valentina |
| D12 | Armazenamento de embeddings | SQLite em `data/private/app.db` (no `.gitignore`): tabela `persons` (id, nome, data) e tabela `embeddings` (id, `person_id` com `ON DELETE CASCADE`, vetor float32 como BLOB, `model_id`, data). `PRAGMA foreign_keys = ON` em toda conexão, senão o SQLite ignora o cascade. **A aplicação não guarda fotos**, só embeddings (minimização de dados), então a exclusão apaga a linha da pessoa e o cascade leva os embeddings | Paola |
| D13 | Interface | **Streamlit**, uma tela com abas Cadastro, Reconhecimento e Pessoas cadastradas. Entrada por `st.camera_input` (foto da webcam, não vídeo contínuo) e `st.file_uploader`, o que cobre "imagem ou vídeo" da seção 38. Vídeo ao vivo (`streamlit-webrtc`) fica fora do escopo | Valentina |
| D14 | Repositório | Público no GitHub. `dev` é a branch de integração: cada tarefa ou spec sai da `dev` numa branch própria e volta por PR revisado pela outra. A `dev` entra na `main` só nas entregas (24/10 e 14/11), então a `main` é sempre "o que foi entregue". Commits no formato `tipo(escopo): descrição` | Juntas |
| D15 | Quem aparece no vídeo | Só Paola e Valentina. Na demonstração da Parte 2, uma se cadastra e a outra aparece antes de se cadastrar como a pessoa desconhecida, sem precisar de terceiros nem de foto de celebridade | Juntas |

## Protocolo de avaliação

- **Público:** as 1.680 identidades do LFW com 2 fotos ou mais são divididas **por identidade** em treino, validação e teste (70/15/15, seed 42, estratificado por faixa de número de fotos: 2–4, 5–9, 10+), e nenhuma pessoa aparece em mais de um split. As 4.069 identidades com uma foto só vão metade para a validação e metade para o teste, como impostoras nos pares negativos e como desconhecidas extras do privado.
- **Pares:** validação e teste usam **todos os pares positivos** das suas identidades, com no máximo 20 por identidade (sorteados com seed 42), e o mesmo número de negativos, o que dá cerca de 1.450 + 1.450 pares em cada. A meta inicial era 3.000 + 3.000 (o tamanho do View 2), mas com o split por identidade e o limite de 20 a validação e o teste só têm cerca de 1.450 positivos possíveis; subir o limite deixaria poucas identidades dominarem os pares. E01, E02, E04 e E06 são avaliados **nos mesmos pares de teste**. Os pares padrão do LFW (View 2) não são usados, porque contêm identidades que entram no treino do fine-tuning.
- **Escolha do modelo:** o "melhor de E01 a E06" que segue para E07 é escolhido pelo EER **da validação**, nunca do teste (seção 29). O teste é rodado uma vez por experimento, depois que tudo está fixado.
- **Privado:** a sessão 1 é a galeria (cadastro e treino do k-NN/SVM). As fotos da sessão 2 de cada pessoa da galeria são divididas metade para validação e metade para teste. **2 participantes ficam fora da galeria** como desconhecidas, uma na validação e outra no teste (todas as fotos delas, das duas sessões), complementadas por 200 rostos de identidades com uma foto do LFW em cada split. A estratégia de comparação (E07) e o threshold (E08) são escolhidos na validação privada e reportados no teste privado.
- Com menos de 5 participantes, a galeria fica pequena demais: registrar no README como limitação e, se der tempo, repetir E08 rodando qual participante fica de fora.

## Contrato entre as fatias

- **Splits** (versionados, só dados públicos) em `data/splits/`, com caminhos relativos a `data/public/lfw_crop160/`, a identidade sendo a pasta (`George_W_Bush/George_W_Bush_0001.png`):
  - `lfw_train.txt`, `lfw_val.txt`, `lfw_test.txt`: uma foto por linha;
  - `lfw_val_pairs.csv`, `lfw_test_pairs.csv`: colunas `path_a,path_b,same` (`same` = 1 ou 0);
  - `lfw_val_unknowns.txt`, `lfw_test_unknowns.txt`: as 200 fotos de desconhecidos extras do privado.
  - O split privado não é versionado: é gerado em tempo de execução a partir de `data/private/<ID>/session{1,2}/` com seed 42.
- **Embeddings** (fora do Git, a pasta `embeddings/` e `*.npz` estão no `.gitignore`): `embeddings/<EXP>/<split>.npz` (no privado, `embeddings/private/<EXP>/<split>.npz`, dentro do cofre) com `embeddings` (float32, N×D, L2-normalizados), `paths` e `labels` (strings), e `model_id`. D = 512 no FaceNet e 2048 na ResNet-50; a avaliação não assume a dimensão.
- **Resultados:** `experiments/results/<EXP>.json` com `experiment`, `hypothesis`, `model`, `strategy`, `dataset`, `seed`, `config` (cópia do YAML), `libraries` (Python, torch, torchvision, facenet-pytorch, scikit-learn, numpy), `hardware` (GPU, CUDA), `training` (épocas rodadas, melhor época, tempo) quando houver, `threshold` (`value`, `criterion`, `chosen_on: "val"`) e `metrics` (`val` e `test`, com os intervalos de confiança). Resultados do privado só agregados, sem nada por pessoa.
- **Gráficos:** `experiments/figures/<EXP>_roc.png`, `<EXP>_hist.png` e, no privado, `<EXP>_confusion.png` com IDs anônimos.
- Com isso, E01 e E02 rodam na segunda só gerando embeddings, e a avaliação roda depois em cima deles, quando as métricas estiverem prontas.

---

## Fase 0, fundação (dom 18)

| Tarefa | Dona |
|---|---|
| Estrutura do repo (`src/`, `experiments/`, `configs/`, `tests/`, `scripts/`, `docs/specs/`, `docs/decisions/`, `data/README.md`), `pyproject.toml`, ruff, pytest com marcadores `unit`/`integration`/`e2e`/`gpu`, pre-commit | Paola |
| `.gitignore` de privacidade (`data/private/`, `data/public/`, `faces/`, `embeddings/`, `checkpoints/`, `*.db`, `*.sqlite`, `consent/`, `*.pt`, `*.pth`, `*.onnx`) e hook de pre-commit que bloqueia imagens e checkpoints fora das pastas permitidas | Paola |
| venv, `requirements.txt` com versões fixadas e `tests/test_environment.py` (CPU por padrão, GPU com `pytest -m gpu -s`), validados na 4060 | Valentina |
| Adaptar o Termo de Consentimento (Anexo A), com a cláusula de envio por mensageiro (ver "Dataset privado"), enviar e colher assinaturas; guardar no cofre do notebook da 4060 (ver "Termos e planilha de controle") | Valentina |
| Ler o enunciado inteiro e fechar as decisões D1 a D15 | Juntas |

## Parte 1, experimentação

### Dados (Paola)
- Script `scripts/download_datasets.py` que baixa o LFW da fonte oficial; `data/README.md` com nome, URL, finalidade, licença e como foi usado (seção 10).
- Pipeline `src/preprocessing/`: detecção MTCNN, alinhamento por landmarks, recorte, normalização. Testes unitários com imagens sintéticas (sem face, múltiplas faces).
- Splits de treino, validação e teste por identidade e pares de teste comuns, salvos como listas de arquivos versionadas.
- Checagem de leakage: hash perceptual entre splits e garantia de que as augmentations só acontecem dentro do treino. Script e relatório.

### Dataset privado (juntas, execução local)
- Protocolo de coleta: cerca de 5 a 10 adultos consentidos, 20 a 30 fotos cada, em 2 sessões em cômodos ou com fundos diferentes, podendo ser no mesmo dia (variando iluminação, pose, distância, óculos e fundo). Minimização de dados.
- Cada uma convida as pessoas do seu lado e colhe os termos delas. As fotos e os termos ficam só no cofre VeraCrypt do notebook da 4060 (D10), e E07 e E08 rodam uma vez com o conjunto inteiro.
- **Envio das fotos por mensageiro (WhatsApp).** A participante pode mandar as próprias fotos por mensageiro, desde que todas as regras abaixo sejam cumpridas. O mensageiro é um serviço de terceiros, então o envio só vale se estiver coberto pelo consentimento (seções 14 e 23):
  - **Consentimento:** o termo traz uma cláusula específica autorizando o envio, por exemplo: "Autorizo o envio das minhas fotografias por mensagem privada no WhatsApp à integrante [nome], que as salvará apenas no computador do grupo e apagará a conversa e as mídias do celular em até 24 horas." Quem assinou um termo sem essa cláusula assina um aditivo ou entrega as fotos sem mensageiro (pendrive ou fotos tiradas por nós).
  - **Canal:** só conversa individual com a integrante que fez o convite, com criptografia de ponta a ponta. Nunca grupos, e-mail, nuvem (iCloud, Google Fotos e afins) ou link de compartilhamento.
  - **Envio:** as fotos vão como foto, não como documento, para o aplicativo remover metadados como a localização (minimização de dados). A participante escreve só "Etapa 1" ou "Etapa 2", sem nome completo nem outros dados pessoais.
  - **Antes de receber:** a integrante desativa o download automático de mídia e a opção de salvar na galeria, e confere que o backup de conversas em nuvem está desligado, para as fotos não irem para a galeria, o Google Fotos ou o iCloud.
  - **Ao receber:** as fotos são salvas direto no cofre do notebook da 4060, em `data/private/<ID>/`, pelo WhatsApp Web aberto no notebook ou por cabo USB. Nunca passam pelo Mac da Paola nem por outro computador. Fotos recebidas no celular da Paola chegam ao notebook por cabo ou por pendrive dentro de um cofre VeraCrypt, nunca reencaminhadas por mensageiro.
  - **Exclusão em até 24 horas:** depois de conferir que as fotos estão no cofre, a integrante apaga as mensagens com "Apagar para todos", limpa a conversa, remove as mídias do armazenamento do celular e esvazia a lixeira da galeria. A participante é avisada de que pode apagar a cópia dela.
  - **Registro:** a planilha de controle, guardada junto aos termos e fora do repositório, anota para cada ID a data de recebimento e a data de exclusão da conversa. Isso serve de evidência no checklist do Anexo B.
- **Termos e planilha de controle.** Os termos têm nome e assinatura (dado pessoal, não biométrico) e ficam todos no cofre da 4060, em `V:\termos` (seção 15 e Anexo B):
  - **Planilha:** cada uma preenche a própria cópia, com as pessoas que convidou, em faixas de ID separadas (Valentina P01 a P10, Paola P11 a P20; se acabarem, combinar uma nova faixa). O nome da participante só aparece na planilha e no termo.
  - **Termos da Paola para o cofre:** a Paola manda os termos dela à Valentina como documento (PDF), na conversa individual do WhatsApp, com o backup de conversas desligado nos dois celulares. A Valentina salva direto em `V:\termos` pelo WhatsApp Web no notebook da 4060 e confere cada arquivo. Em até 24 horas, as duas apagam as mensagens com "Apagar para todos", limpam a conversa e removem os arquivos dos celulares. Só termos passam por essa conversa: fotos recebidas pela Paola nunca são reencaminhadas, vão por cabo ou pendrive com cofre.
  - **Limpeza:** depois de salvar no cofre, apagar as cópias que tenham ficado em Downloads, Área de Trabalho, anexos de e-mail ou no celular, e esvaziar as lixeiras. A cópia da planilha da Paola também vai para o cofre ao fim da coleta.
  - **Registro:** as datas de envio, recebimento e exclusão das conversas vão na planilha, e o envio dos termos pelo WhatsApp entre as duas é descrito em `docs/privacy.md`.
- **Cofre.** Um contêiner VeraCrypt no notebook da 4060 (fora de pastas sincronizadas com nuvem), montado como `V:` só durante o uso. Tudo que vem do dataset privado fica dentro dele, ligado ao projeto por junções do Windows (`New-Item -ItemType Junction`), cujas pastas de origem já estão no `.gitignore` e no hook:
  - `data/private/` → `V:\private` (fotos originais);
  - `faces/private/` → `V:\faces` (recortes do MTCNN);
  - `embeddings/private/` → `V:\embeddings` (embeddings de E07 e E08);
  - `checkpoints/private/` → `V:\checkpoints` (checkpoints e classificadores ajustados com dados privados);
  - `V:\termos` (termos e planilhas, sem junção, fora do projeto).
  Com o cofre desmontado, as junções ficam vazias e nenhum script consegue gravar dado privado fora dele. Resultados por pessoa só existem no cofre; em `experiments/` vão apenas os agregados.
- `data/README.md` com instruções para alguém criar o próprio dataset privado localmente.

### Modelos e treinamento (Valentina)
- `src/facial_recognition_system/models/`: wrappers dos backbones A e B devolvendo embedding L2-normalizado, de 512 dimensões no FaceNet e 2048 na ResNet-50 (D4).
- `src/facial_recognition_system/training/`: loop de treino com cabeça ArcFace e Triplet Loss, congelamento parcial configurável, early stopping na validação, seed fixa, modo `--smoke`.
- Checkpoints de experimentos com dados privados ficam em `checkpoints/private/`, que é uma junção para o cofre (ver "Cofre"), fora do Git.

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
- **Fluxo entre máquinas:** código pelo Git; cada máquina tem o próprio venv e baixa o LFW sozinha; rodar o `--smoke` antes do treino completo; os resultados JSON e os gráficos agregados voltam pelo Git; checkpoints ficam fora do Git.
- **LGPD:** as fotos privadas só são processadas no notebook da 4060, e agentes de IA nunca recebem fotos reais nem termos assinados. As fotos, tudo que deriva delas, os termos e as planilhas ficam num cofre VeraCrypt nesse notebook (D10 e "Cofre"). O envio por mensageiro só é aceito com a cláusula no termo, em conversa individual, e a conversa é apagada em até 24 horas (ver "Dataset privado").

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
