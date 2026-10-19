<details open>
<summary><strong>🇧🇷 Português</strong></summary>

# Dados

Nenhuma imagem é versionada. Só este README e as listas de splits em `splits/` (caminhos de dados públicos) vão para o GitHub. O hook `scripts/check_private_files.py` bloqueia qualquer outro arquivo dentro de `data/`.

| Pasta | Conteúdo | Versionada |
|---|---|---|
| `public/` | datasets públicos, baixados da fonte | não |
| `private/` | fotos dos participantes, coletadas com consentimento | **nunca** |
| `splits/` | listas de treino, validação e teste (só dados públicos) | sim |

## Dataset público

### LFW (Labeled Faces in the Wild)

| Campo | Valor |
|---|---|
| Fonte oficial | <http://vis-www.cs.umass.edu/lfw/> (University of Massachusetts Amherst). Fora do ar em 18/10/2026: o domínio não resolve mais |
| Cópia usada no download | <https://ndownloader.figshare.com/files/5976018>, a mesma usada pelo `sklearn.datasets.fetch_lfw_people`, conferida pelo SHA-256 `055f7d9c632d7370e6fb4afc7468d40f970c34a80d4c6f50ffec63f5a8d536c0` |
| Conteúdo | 13.233 fotos de 5.749 pessoas, coletadas de notícias na web. 1.680 pessoas têm 2 fotos ou mais, 423 têm 5 ou mais e 158 têm 10 ou mais |
| Finalidade | benchmark de verificação facial (E01, E02, E04 e E06) e subconjunto de identidades para o fine-tuning (decisões D1 e D2) |
| Licença e termos | o LFW não tem uma licença formal. Os autores o distribuem para pesquisa e pedem a citação abaixo, e os direitos das fotos continuam com os autores originais. Por isso o repositório não redistribui nenhuma imagem, só o script que baixa da fonte |
| Citação | Huang, G. B.; Ramesh, M.; Berg, T.; Learned-Miller, E. *Labeled Faces in the Wild: A Database for Studying Face Recognition in Unconstrained Environments*. University of Massachusetts, Amherst, Technical Report 07-49, 2007 |
| Como foi usado | versão original (sem alinhamento), detectada e alinhada pelo nosso pipeline com MTCNN e dividida por identidade em treino, validação e teste (`splits/`) |

Para baixar:

```bash
python scripts/download_datasets.py
```

O script baixa cerca de 177 MB para `data/public/lfw.tgz`, confere o SHA-256 e extrai em `data/public/lfw/` (cerca de 206 MB). Rodar de novo não baixa outra vez; `--force` baixa e extrai de novo.

> **macOS:** se aparecer `CERTIFICATE_VERIFY_FAILED` com o Python instalado pelo python.org, rode o `Install Certificates.command` que fica na pasta do Python em `/Applications`.

### Pré-processamento

O MTCNN detecta o rosto, e o pipeline recorta e salva um PNG de 160×160 por foto, com a mesma estrutura de pastas:

```bash
python scripts/preprocess_faces.py --input-dir data/public/lfw --output-dir data/public/lfw_crop160 --selection center
```

- `--selection center` fica com o rosto mais perto do centro quando a foto tem mais de um, porque as fotos do LFW são centradas na pessoa rotulada. No dataset privado e na aplicação, use o padrão `strict`, que rejeita fotos com mais de um rosto.
- `--alignment crop` (padrão) recorta a caixa do rosto como o facenet-pytorch, que é a entrada com que o FaceNet pré-treinado foi treinado. `--alignment similarity` alinha os 5 pontos faciais ao template do ArcFace, corrigindo a inclinação da cabeça.
- O `preprocessing_report.json` na pasta de saída guarda a configuração usada e cada foto em que nenhum rosto passou da probabilidade mínima (`--min-probability`, padrão 0.9).
- Rodar de novo pula os recortes que já existem; `--force` refaz tudo.

### Splits

Os splits são gerados a partir dos recortes, então as fotos em que o MTCNN não achou rosto já ficam de fora:

```bash
python scripts/make_splits.py --crops-dir data/public/lfw_crop160 --output-dir data/splits
```

- As identidades com 2 fotos ou mais são divididas **por identidade** em treino, validação e teste (70/15/15, seed 42), separadamente em cada faixa de número de fotos (2–4, 5–9 e 10+). Nenhuma pessoa aparece em mais de um split.
- As identidades com uma foto só vão metade para a validação e metade para o teste, onde servem de impostoras nos pares negativos e de desconhecidas extras do dataset privado.
- Pares positivos: todas as combinações de duas fotos da mesma pessoa, com no máximo 20 por identidade (sorteadas quando passam disso). Pares negativos: o mesmo número, sorteando primeiro duas identidades diferentes e depois uma foto de cada, para quem tem muitas fotos não dominar.
- Desconhecidos extras: 200 fotos de identidades com uma foto só, em cada split.

| Arquivo | Conteúdo | Tamanho |
|---|---|---|
| `lfw_train.txt` | fotos do treino, uma por linha | 1.176 identidades, 6.439 fotos |
| `lfw_val.txt` | fotos da validação | 2.287 identidades, 3.490 fotos |
| `lfw_test.txt` | fotos do teste | 2.286 identidades, 3.303 fotos |
| `lfw_val_pairs.csv`, `lfw_test_pairs.csv` | colunas `path_a,path_b,same` (`same` = 1 ou 0) | 1.447 + 1.447 e 1.445 + 1.445 pares |
| `lfw_val_unknowns.txt`, `lfw_test_unknowns.txt` | desconhecidos extras | 200 fotos cada |

Os caminhos são relativos a `data/public/lfw_crop160/`, com a identidade como pasta (`George_W_Bush/George_W_Bush_0001.png`). Com os mesmos recortes e a mesma seed, o script gera sempre os mesmos arquivos.
- No dataset privado, os recortes também ficam no cofre: `python scripts/preprocess_faces.py --input-dir data/private --output-dir faces/private`, com `faces/private/` ligada ao cofre (ver "Dataset privado").

## Dataset privado

Fotos de adultos que assinaram o Termo de Consentimento (Anexo A do enunciado), tiradas pelos próprios participantes na hora, a partir do [guia para participantes](../docs/participant_photo_guide.pdf), e entregues diretamente à dupla, sem scraping nem fotos antigas. Os termos ficam fora do repositório, no mesmo cofre criptografado das fotos (ver [docs/division.md](../docs/division.md), "Termos e planilha de controle").

Para montar o seu localmente:

```text
data/private/
├── P01/
│   ├── session1/   # galeria (cadastro)
│   └── session2/   # validação e teste
├── P02/
│   ├── session1/
│   └── session2/
└── ...
```

- As pastas usam IDs anônimos (`P01`, `P02`, ...), nunca nomes. A relação entre ID e pessoa fica junto dos termos, fora do repositório.
- São duas sessões em cômodos ou com fundos diferentes (podem ser no mesmo dia), com 20 a 30 fotos por pessoa no total, variando iluminação, pose, distância, expressão, óculos e fundo. Coletar só o necessário (minimização de dados).
- As fotos podem ser enviadas por mensageiro (WhatsApp) só por quem assinou o termo com a cláusula que autoriza esse envio. O mensageiro é um serviço de terceiros, então o envio precisa estar coberto pelo consentimento (seções 14 e 23 do enunciado). Quem não autorizou entrega as fotos por pendrive ou as tira com uma de nós.
- O envio é feito na conversa particular com a integrante responsável pelo convite, nunca em grupo, e-mail, nuvem ou link de compartilhamento. As fotos vão como foto, não como documento, para o aplicativo remover metadados como a localização, e a mensagem diz só "Etapa 1" ou "Etapa 2", sem outros dados pessoais.
- Antes de receber, a integrante desativa o download automático de mídia, a opção de salvar na galeria e o backup de conversas em nuvem.
- As fotos são salvas direto em `data/private/<ID>/`, dentro de um cofre VeraCrypt no notebook da RTX 4060 (decisão D10), pelo WhatsApp Web aberto nesse notebook ou por cabo USB. Fotos recebidas no celular da outra integrante chegam por cabo ou por pendrive dentro de um cofre VeraCrypt, nunca reencaminhadas por mensageiro.
- Em até 24 horas depois de salvas, a integrante apaga as mensagens com "Apagar para todos", limpa a conversa, remove as mídias do celular e esvazia a lixeira da galeria. A data de recebimento e a data de exclusão ficam registradas na planilha de controle, guardada junto aos termos e fora do repositório.
- As fotos e tudo que deriva delas (recortes, embeddings, checkpoints, resultados por pessoa) ficam só nesse cofre e nunca são enviados a serviços externos nem a agentes de IA. No projeto, `data/private/`, `faces/private/`, `embeddings/private/` e `checkpoints/private/` são junções para o cofre, então nada é gravado fora dele.
- Para excluir uma pessoa, apague `data/private/<ID>/` e os embeddings, checkpoints e resultados derivados dela.
- Ao fim da disciplina, todo o conteúdo de `data/private/` e os arquivos derivados são apagados.

</details>

<details>
<summary><strong>🇺🇸 English</strong></summary>

# Data

No image is versioned. Only this README and the split lists in `splits/` (public data paths) go to GitHub. The `scripts/check_private_files.py` hook blocks any other file inside `data/`.

| Folder | Contents | Versioned |
|---|---|---|
| `public/` | public datasets, downloaded from the source | no |
| `private/` | participant photos, collected with consent | **never** |
| `splits/` | train, validation and test lists (public data only) | yes |

## Public dataset

### LFW (Labeled Faces in the Wild)

| Field | Value |
|---|---|
| Official source | <http://vis-www.cs.umass.edu/lfw/> (University of Massachusetts Amherst). Offline as of 2026-10-18: the domain no longer resolves |
| Copy used for download | <https://ndownloader.figshare.com/files/5976018>, the same one used by `sklearn.datasets.fetch_lfw_people`, verified by SHA-256 `055f7d9c632d7370e6fb4afc7468d40f970c34a80d4c6f50ffec63f5a8d536c0` |
| Contents | 13,233 photos of 5,749 people, collected from news on the web. 1,680 people have 2 or more photos, 423 have 5 or more and 158 have 10 or more |
| Purpose | face verification benchmark (E01, E02, E04 and E06) and identity subset for fine-tuning (decisions D1 and D2) |
| License and terms | LFW has no formal license. The authors distribute it for research and ask for the citation below, and the photo rights remain with the original authors. For this reason the repository redistributes no images, only the script that downloads them from the source |
| Citation | Huang, G. B.; Ramesh, M.; Berg, T.; Learned-Miller, E. *Labeled Faces in the Wild: A Database for Studying Face Recognition in Unconstrained Environments*. University of Massachusetts, Amherst, Technical Report 07-49, 2007 |
| How it was used | original (unaligned) version, detected and aligned by our MTCNN pipeline and split by identity into train, validation and test (`splits/`) |

To download:

```bash
python scripts/download_datasets.py
```

The script downloads about 177 MB to `data/public/lfw.tgz`, verifies the SHA-256 and extracts it to `data/public/lfw/` (about 206 MB). Running it again does not download twice; `--force` downloads and extracts again.

> **macOS:** if you get `CERTIFICATE_VERIFY_FAILED` with the python.org Python installer, run the `Install Certificates.command` found in the Python folder under `/Applications`.

### Preprocessing

MTCNN detects the face, and the pipeline crops and saves one 160×160 PNG per photo, keeping the same folder structure:

```bash
python scripts/preprocess_faces.py --input-dir data/public/lfw --output-dir data/public/lfw_crop160 --selection center
```

- `--selection center` keeps the face closest to the center when a photo has more than one, because LFW photos are centered on the labeled person. For the private dataset and the application, use the default `strict`, which rejects photos with more than one face.
- `--alignment crop` (default) crops the face box like facenet-pytorch, which is the input the pretrained FaceNet was trained on. `--alignment similarity` aligns the 5 facial landmarks to the ArcFace template, correcting head tilt.
- The `preprocessing_report.json` in the output folder stores the configuration used and every photo where no face reached the minimum probability (`--min-probability`, default 0.9).
- Running it again skips crops that already exist; `--force` redoes everything.

### Splits

The splits are built from the crops, so photos where MTCNN found no face are already left out:

```bash
python scripts/make_splits.py --crops-dir data/public/lfw_crop160 --output-dir data/splits
```

- Identities with 2 or more photos are split **by identity** into train, validation and test (70/15/15, seed 42), separately within each photo-count range (2–4, 5–9 and 10+). No person appears in more than one split.
- Single-photo identities go half to validation and half to test, where they act as impostors in negative pairs and as extra unknowns for the private dataset.
- Positive pairs: every combination of two photos of the same person, at most 20 per identity (sampled when there are more). Negative pairs: the same number, drawing two different identities first and one photo of each second, so people with many photos do not dominate.
- Extra unknowns: 200 photos of single-photo identities in each split.

| File | Contents | Size |
|---|---|---|
| `lfw_train.txt` | training photos, one per line | 1,176 identities, 6,439 photos |
| `lfw_val.txt` | validation photos | 2,287 identities, 3,490 photos |
| `lfw_test.txt` | test photos | 2,286 identities, 3,303 photos |
| `lfw_val_pairs.csv`, `lfw_test_pairs.csv` | columns `path_a,path_b,same` (`same` = 1 or 0) | 1,447 + 1,447 and 1,445 + 1,445 pairs |
| `lfw_val_unknowns.txt`, `lfw_test_unknowns.txt` | extra unknowns | 200 photos each |

Paths are relative to `data/public/lfw_crop160/`, with the identity as the folder (`George_W_Bush/George_W_Bush_0001.png`). With the same crops and seed, the script always writes the same files.
- For the private dataset, the crops also stay in the vault: `python scripts/preprocess_faces.py --input-dir data/private --output-dir faces/private`, with `faces/private/` linked to the vault (see "Private dataset").

## Private dataset

Photos of adults who signed the consent form (Annex A of the assignment), taken by the participants themselves on the spot, following the [participant guide](../docs/participant_photo_guide.pdf) (in Portuguese), and handed directly to the team, with no scraping and no old photos. Consent forms are kept outside the repository, in the same encrypted vault as the photos (see [docs/division.md](../docs/division.md), "Termos e planilha de controle").

To build your own locally:

```text
data/private/
├── P01/
│   ├── session1/   # gallery (enrollment)
│   └── session2/   # validation and test
├── P02/
│   ├── session1/
│   └── session2/
└── ...
```

- Folders use anonymous IDs (`P01`, `P02`, ...), never names. The mapping between ID and person is kept with the consent forms, outside the repository.
- There are two sessions in different rooms or against different backgrounds (they may be on the same day), with 20 to 30 photos per person in total, varying lighting, pose, distance, expression, glasses and background. Collect only what is needed (data minimization).
- Photos may be sent through a messaging app (WhatsApp) only by participants who signed the consent form with the clause authorizing it. The messaging app is a third-party service, so sending must be covered by the consent (sections 14 and 23 of the assignment). Participants who did not authorize it hand the photos over on a USB drive or have them taken by one of us.
- Photos are sent in a private chat with the team member responsible for the invitation, never in a group, by email, through cloud storage or via a sharing link. They are sent as photos, not as documents, so the app strips metadata such as location, and the message says only "Etapa 1" or "Etapa 2", with no other personal data.
- Before receiving them, the team member turns off automatic media download, saving to the gallery and cloud chat backup.
- Photos are saved directly to `data/private/<ID>/`, inside a VeraCrypt vault on the RTX 4060 notebook (decision D10), through WhatsApp Web opened on that notebook or over a USB cable. Photos received on the other team member's phone arrive over a USB cable or on a USB drive inside a VeraCrypt vault, never forwarded through a messaging app.
- Within 24 hours of being saved, the team member deletes the messages with "Delete for everyone", clears the chat, removes the media from the phone and empties the gallery trash. The date received and the date deleted are recorded in the control spreadsheet, kept with the consent forms and outside the repository.
- Photos and everything derived from them (crops, embeddings, checkpoints, per-person results) stay only in that vault and are never sent to external services or AI agents. In the project, `data/private/`, `faces/private/`, `embeddings/private/` and `checkpoints/private/` are junctions to the vault, so nothing is written outside it.
- To delete a person, remove `data/private/<ID>/` and the embeddings, checkpoints and results derived from them.
- At the end of the course, all of `data/private/` and its derived files are deleted.

</details>
