# D3: detector e alinhamento

- **Status:** Aceita
- **Data:** 2026-10-18
- **Dona:** Paola

## Contexto

O pipeline do enunciado começa com detecção, alinhamento e pré-processamento da face (seção 7). O mesmo pipeline precisa servir para o LFW, para as fotos de celular do dataset privado e para a câmera da aplicação, e precisa entregar a entrada que o FaceNet pré-treinado espera. A aplicação também precisa saber quando a imagem tem nenhum ou mais de um rosto (spec 001).

## Decisão

**MTCNN do `facenet-pytorch`, recorte de 160×160, `--alignment crop`, margem 0 e probabilidade mínima de 0,9.**

- `--alignment crop` recorta a caixa do rosto do mesmo jeito que o `facenet-pytorch`, que é a entrada com que o FaceNet pré-treinado foi treinado.
- **LFW:** `--selection center`, que fica com o rosto mais perto do centro, porque as fotos do LFW são centradas na pessoa rotulada e às vezes têm outras pessoas ao fundo.
- **Dataset privado e aplicação:** `--selection strict`, que rejeita fotos com mais de um rosto.
- Fotos em que nenhum rosto passa de 0,9 ficam fora dos splits e são listadas no `preprocessing_report.json`. No LFW, só 1 de 13.233 fotos falhou.
- O alinhamento por similaridade (5 pontos faciais no template do ArcFace) está implementado, mas não entra nos experimentos.

## Alternativas consideradas

- **RetinaFace ou outros detectores:** mais precisos em poses difíceis, mas trariam uma dependência nova, e o MTCNN já vem com o `facenet-pytorch`.
- **Alinhamento por similaridade como padrão:** corrige a inclinação da cabeça, mas muda a entrada em relação à que o FaceNet pré-treinado viu, o que prejudicaria o E01 sem ser essa a pergunta do experimento.
- **Haar cascades (OpenCV):** mais simples, mas com muito mais falsos positivos e sem os pontos faciais.

## Consequências

- Experimentos e aplicação usam o mesmo código ([src/facial_recognition_system/preprocessing/](../../src/facial_recognition_system/preprocessing/)), então o threshold escolhido na Parte 1 vale para as imagens da Parte 2.
- O `strict` transforma "mais de um rosto" num erro explícito (`MultipleFacesError`). O guia para participantes pede que só a pessoa apareça na foto.
- Poses muito laterais podem não passar de 0,9. O guia pede que os dois olhos apareçam, e as falhas do dataset privado ficam no relatório.
- Implementado em [scripts/preprocess_faces.py](../../scripts/preprocess_faces.py).
