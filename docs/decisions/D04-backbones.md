# D4: backbones comparados

- **Status:** Aceita
- **Data:** 2026-10-18
- **Dona:** Valentina

## Contexto

A escolha da rede que gera o embedding precisa ser tecnicamente justificada (seção 7), e os experimentos precisam isolar o efeito de cada mudança (seção 27). A seção 28 sugere comparar o modelo pré-treinado sem fine-tuning com o fine-tuning parcial. Para isso, o grupo precisa de pelo menos um backbone treinado em rostos e de uma referência que mostre quanto esse pré-treino facial vale. Tudo roda no notebook da RTX 4060, com 8 GB (D10), e o modelo escolhido vai para a aplicação da Parte 2, rodando em tempo real na webcam.

## Decisão

**A: FaceNet (InceptionResnetV1) pré-treinado no VGGFace2**, do `facenet-pytorch`.

- Embedding de **512 dimensões**, L2-normalizado.
- Entrada: o recorte 160×160 do MTCNN (D3), com `fixed_image_standardization`, ou seja, `(x − 127,5) / 128`, que é a normalização com que ele foi treinado.
- É o backbone de E01 (sem fine-tuning), E04 e E06 (fine-tuning parcial, D5).

**B: ResNet-50 do torchvision com os pesos `IMAGENET1K_V2`**, sem a camada `fc`.

- Embedding de **2048 dimensões**: a saída do pooling global, L2-normalizada, **sem projeção** para 512. Uma camada de projeção nova teria pesos aleatórios e mediria o sorteio desses pesos, não o pré-treino.
- Entrada: o mesmo recorte 160×160, normalizado com a média e o desvio do ImageNet e redimensionado para 224×224 dentro do modelo. Os dois backbones usam exatamente os mesmos recortes.
- É o backbone de E02. Ele mede **o valor do pré-treino facial**: mesma família de arquitetura convolucional, mas aprendida em objetos genéricos.

Os dois ficam atrás da mesma interface (`FaceEmbedder`: `preprocess` e `forward`), então a avaliação compara só vetores e não precisa saber de onde eles vieram.

## Alternativas consideradas

- **ArcFace pré-treinado (InsightFace, iResNet-100):** é o estado da arte, mas os pesos oficiais têm licença não comercial com restrições de redistribuição e dependem do `onnxruntime` ou de um port não oficial. O FaceNet vem do `pip`, com licença MIT e pesos baixados por referência (seção 21). A perda ArcFace entra pelo fine-tuning (D5).
- **MobileFaceNet:** mais leve, bom para a câmera, mas não há pesos pré-treinados no `facenet-pytorch` nem no `torchvision`, e treinar do zero não cabe no prazo.
- **Vision Transformer (ViT) do ImageNet como referência B:** mede a mesma coisa que a ResNet-50 (pré-treino genérico), com mais custo e entrada de 224 com patches. A ResNet-50 já cumpre o papel.
- **ResNet-50 com projeção para 512 dimensões:** deixaria as dimensões iguais, mas a camada nova sem treino adicionaria ruído (ver acima). A avaliação não depende da dimensão (contrato entre as fatias).

## Consequências

- Verificação no teste do LFW, com o threshold do EER escolhido na validação (código do PR #70): **E01 com EER de 1,1% e ROC-AUC de 0,999; E02 com EER de 31,1% e ROC-AUC de 0,749.** O pré-treino em rostos é o que separa as pessoas. O ImageNet aprende "é um rosto", mas não "de quem é o rosto": na ResNet-50, a similaridade média nos pares de teste é 0,84 entre fotos da mesma pessoa e 0,78 entre pessoas diferentes (as distribuições do `E02_hist.png`; o comando que recalcula as médias está em "Reprodutibilidade" no README).
- **Limitação a registrar no README:** o VGGFace2 pode conter celebridades que também estão no LFW, o que favorece o E01 (D1). No dataset privado (E07), essas pessoas não existem, e o resultado vai mostrar o quanto isso pesou.
- Só o FaceNet passa por fine-tuning. Com fine-tuning, a ResNet-50 viraria os experimentos E03 e E05, que foram cortados por prazo (division.md).
- Implementado em [src/facial_recognition_system/models/embedders.py](../../src/facial_recognition_system/models/embedders.py) (PR #67), com testes de forma, norma e entrada em [tests/test_models.py](../../tests/test_models.py). Configs em [configs/E01.yaml](../../configs/E01.yaml) e [configs/E02.yaml](../../configs/E02.yaml).
