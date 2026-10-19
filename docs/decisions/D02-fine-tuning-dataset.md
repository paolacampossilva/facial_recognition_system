# D2: dataset público para fine-tuning

- **Status:** Aceita
- **Data:** 2026-10-18
- **Dona:** Paola

## Contexto

O E04 (ArcFace) e o E06 (Triplet Loss) fazem fine-tuning parcial do FaceNet (D5) e precisam de um conjunto de treino com identidades rotuladas que não apareçam na validação nem no teste (seção 29). O Triplet Loss usa amostragem P×K com K = 4 fotos por identidade, e a comparação entre as duas losses só é justa se elas treinarem nos mesmos dados.

## Decisão

**Identidades do split de treino do LFW (D9) com pelo menos 4 fotos, no máximo 40 fotos por identidade.**

- Com os splits atuais, isso dá **433 identidades e 3.961 fotos** (4.757 antes do limite de 40).
- O limite de 40 impede que poucas pessoas dominem o treino: o George W. Bush, que caiu no treino, tem 530 fotos.
- ArcFace e Triplet treinam exatamente nesse mesmo conjunto.

## Alternativas consideradas

- **Todas as identidades de treino com 2 fotos ou mais** (1.176 identidades): a maioria tem 2 ou 3 fotos, o que não permite K = 4 no Triplet, e as duas losses deixariam de treinar nos mesmos dados.
- **Sem limite de fotos por identidade:** o Bush sozinho seria mais de 10% do conjunto, e o modelo aprenderia a separar poucas pessoas muito bem.
- **CelebA como fonte do fine-tuning:** mais identidades, mas fora do prazo (D1).

## Consequências

- O conjunto é pequeno para fine-tuning, por isso só o último bloco do FaceNet treina (D5). Há risco de overfitting, que o early stopping pelo EER da validação controla.
- **Limitação a registrar no README:** o fine-tuning usa poucas centenas de identidades, então o ganho sobre o E01 pode ser pequeno ou até negativo.
- A seleção do subconjunto fica no dataloader de treino da Valentina (#17), a partir de `data/splits/lfw_train.txt`.
