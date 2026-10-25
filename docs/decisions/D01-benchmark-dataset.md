# D1: dataset público de benchmark

- **Status:** Aceita
- **Data:** 2026-10-18
- **Dona:** Paola

## Contexto

A Parte 1 precisa de pelo menos um dataset público de pesquisa em reconhecimento facial (seção 9 do enunciado), usado de acordo com a licença e sem redistribuir as imagens (seção 10). O benchmark precisa caber no prazo de uma semana, rodar no notebook da RTX 4060 (D10) e passar pelo mesmo pipeline de detecção que o dataset privado e a aplicação vão usar (D3).

## Decisão

**LFW (Labeled Faces in the Wild), versão original, sem alinhamento**: 13.233 fotos de 5.749 pessoas.

- O site oficial (UMass Amherst) saiu do ar, então o `scripts/download_datasets.py` baixa a cópia do figshare usada pelo `sklearn.datasets.fetch_lfw_people` e confere o SHA-256 antes de extrair.
- A detecção, o recorte e o split são feitos pelo nosso pipeline (D3 e D9), e não pelos recortes ou pares prontos do LFW.
- Nenhuma imagem é versionada: o repositório tem só o script, as listas de split e a citação ([data/README.md](../../data/README.md)).

## Alternativas consideradas

- **CelebA** (mais de 200 mil fotos, 10 mil identidades): ficou de fora pelo tempo de download e preparo e pelo espaço em disco, sem ganho claro para a pergunta dos experimentos.
- **Versões alinhadas do LFW** (deep-funneled e similares): o alinhamento delas não é o mesmo que o MTCNN faz nas fotos do celular e da câmera, e os experimentos precisam medir o pipeline que a aplicação vai usar.
- **Pares oficiais do LFW (View 2)**: não são usados porque contêm identidades que entram no treino do fine-tuning (detalhes na D9).

## Consequências

- Toda a avaliação pública depende de um dataset pequeno: 1.680 pessoas têm 2 fotos ou mais, e só 158 têm 10 ou mais. Os intervalos de confiança da D7 mostram esse efeito.
- **Limitação a registrar no README:** o FaceNet do E01 foi pré-treinado no VGGFace2, que tem celebridades em comum com o LFW. Isso favorece o E01 (D4).
- Como a fonte oficial saiu do ar, a reprodutibilidade depende da cópia do figshare. O SHA-256 no script garante que é o mesmo arquivo.
- Implementado em [scripts/download_datasets.py](../../scripts/download_datasets.py) e documentado em [data/README.md](../../data/README.md).
