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

## Dataset privado

Fotos de adultos que assinaram o Termo de Consentimento (Anexo A do enunciado), tiradas diretamente pela dupla, sem scraping. Os termos ficam fora do repositório (papel ou pasta local criptografada).

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
- São duas sessões em dias e ambientes diferentes, com 20 a 30 fotos por pessoa no total, variando iluminação, pose, distância, expressão, óculos e fundo. Coletar só o necessário (minimização de dados).
- As fotos são processadas só nos computadores da dupla e nunca são enviadas a serviços externos nem a agentes de IA.
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

## Private dataset

Photos of adults who signed the consent form (Annex A of the assignment), taken directly by the team, with no scraping. Consent forms are kept outside the repository (paper or an encrypted local folder).

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
- There are two sessions on different days and in different settings, with 20 to 30 photos per person in total, varying lighting, pose, distance, expression, glasses and background. Collect only what is needed (data minimization).
- Photos are processed only on the team's computers and are never sent to external services or AI agents.
- To delete a person, remove `data/private/<ID>/` and the embeddings, checkpoints and results derived from them.
- At the end of the course, all of `data/private/` and its derived files are deleted.

</details>
