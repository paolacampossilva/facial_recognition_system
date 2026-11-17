# Spec 003: reconhecimento facial

- **Status:** Em revisão
- **Data:** 2026-10-28
- **Dona:** Paola
- **Issue:** #37, #103 (vários rostos)
- **Relacionadas:** [001](001-face-detection.md), [004](004-unknown-person.md), [D08](../decisions/D08-threshold-selection.md), [D12](../decisions/D12-embedding-storage.md), [D16](../decisions/D16-chosen-model.md), [D19](../decisions/D19-multi-face-recognition.md)

## Problema

Dada uma imagem, a aplicação precisa dizer, para cada rosto que ela tem, se ele pertence a uma pessoa cadastrada e a qual (seção 38), usando o modelo, a estratégia e o threshold escolhidos na Parte 1 (seção 36). Uma imagem pode ter um rosto ou vários (issue #103).

## Requisitos

- **R1.** Pipeline: imagem → spec 001 (`process_all`, todos os rostos) → um embedding (E06) por rosto → comparação com a galeria → threshold → um resultado por rosto.
- **R2.** Estratégia de comparação: **vizinho mais próximo** (similaridade de cosseno com a foto mais parecida da galeria), conforme a D16.
- **R3.** O threshold vem da configuração (spec 007), nunca fixo no código. O valor padrão é **0,766** (E08). Uma face é aceita quando `similaridade >= threshold` (D08).
- **R4.** A galeria são os embeddings do banco com o `model_id` igual ao configurado. Embeddings de outro modelo são ignorados (D12).
- **R5.** O resultado é um modelo Pydantic com o tamanho da imagem e a lista de rostos. Cada rosto traz a caixa, o status, o nome (só se reconhecido) e a similaridade.
- **R6.** O reconhecimento não grava nada (nem a imagem, nem o embedding da consulta).
- **R7.** Galeria vazia não é erro de reconhecimento: é um estado que a interface mostra como "Nenhuma pessoa cadastrada" (seção 43).
- **R8.** Cada rosto é decidido sozinho, com o mesmo threshold (spec 004, R5). A lista sai ordenada da esquerda para a direita (spec 001, R9).
- **R9.** A mesma pessoa pode aparecer em mais de um rosto da imagem (uma foto de uma foto, um espelho, gêmeos). O serviço **não** escolhe um deles nem deduplica: devolve o resultado de cada rosto, e a interface avisa (spec 005).
- **R10.** Os rostos que a spec 001 ignora (menores que o tamanho mínimo, R7 dela) não aparecem na lista. Se não sobra nenhum, vale o erro `NoFaceDetectedError`.
- **R11.** A galeria é lida **uma vez por imagem**, e os vizinhos mais próximos de todos os rostos saem de uma única consulta em lote.

## Critérios de aceitação

| # | Dado | Quando | Então |
|---|---|---|---|
| A1 | "Ana" cadastrada e uma foto dela | reconhece | status `recognized`, nome "Ana" e similaridade ≥ threshold |
| A2 | "Ana" e "Bia" cadastradas e uma foto de Bia | reconhece | devolve "Bia", e não "Ana" |
| A3 | a similaridade com a foto mais parecida é exatamente igual ao threshold | reconhece | status `recognized` (a comparação é `>=`) |
| A4 | a similaridade com a foto mais parecida é menor que o threshold | reconhece | status `unknown` (spec 004) |
| A5 | o threshold da configuração muda de 0,766 para outro valor | reconhece a mesma imagem | o resultado segue o novo valor, sem alterar código |
| A6 | embeddings de dois `model_id` no banco | reconhece | só os do modelo configurado entram na comparação |
| A7 | banco sem nenhuma pessoa (ou sem embeddings do modelo configurado) | reconhece | lança `EmptyGalleryError` |
| A8 | uma imagem sem rosto (ou só com rostos pequenos demais) ou inválida | reconhece | propaga `NoFaceDetectedError` ou `InvalidImageError` da spec 001. Vários rostos **não** são erro |
| A9 | qualquer reconhecimento | o serviço termina | o banco e o disco não mudaram |
| A10 | uma pessoa cadastrada ou excluída logo antes | reconhece | o resultado já reflete a mudança (sem reiniciar) |
| A11 | detector e embedder fakes que devolvem vetores fixos | o serviço é testado | os testes rodam sem rede, sem GPU e sem pesos |
| A12 | "Ana" e "Bia" cadastradas e uma imagem com os rostos das duas | reconhece | a lista tem dois rostos, da esquerda para a direita, com os nomes certos |
| A13 | "Ana" cadastrada e uma imagem com ela e uma pessoa não cadastrada | reconhece | um rosto `recognized` com o nome "Ana" e outro `unknown` com `person_name` `None` (spec 004) |
| A14 | uma imagem com um único rosto | reconhece | a lista tem um rosto, com a mesma decisão que um reconhecimento de rosto único sempre teve |
| A15 | "Ana" cadastrada e uma imagem em que dois rostos passam do threshold com ela | reconhece | os dois rostos vêm `recognized` com o nome "Ana", sem deduplicar |
| A16 | uma imagem com um rosto grande e um rosto menor que o tamanho mínimo | reconhece | só o rosto grande aparece na lista |
| A17 | uma imagem em que o único rosto é menor que o tamanho mínimo | reconhece | propaga `NoFaceDetectedError` |
| A18 | uma imagem com 25 rostos | reconhece | a lista tem os 20 de maior área (spec 001, R8) |
| A19 | qualquer reconhecimento com sucesso | a resposta é inspecionada | traz a largura e a altura da imagem (depois da EXIF) e, para cada rosto, a caixa em pixels dessa imagem, com `0 <= left < right <= image_width` e `0 <= top < bottom <= image_height` |

## Interface

```python
class RecognitionStatus(StrEnum):
    RECOGNIZED = "recognized"
    UNKNOWN = "unknown"


class RecognitionResult(BaseModel):  # a decisão sobre um rosto (spec 004)
    status: RecognitionStatus
    person_name: str | None  # None quando unknown
    similarity: float  # cosseno com a foto mais parecida da galeria


class FaceBox(BaseModel):  # pixels da imagem depois da EXIF, dentro da imagem (spec 001, R9)
    left: int
    top: int
    right: int
    bottom: int


class FaceResult(RecognitionResult):  # um rosto da imagem
    box: FaceBox


class RecognitionResponse(BaseModel):
    image_width: int
    image_height: int
    faces: list[FaceResult]  # pelo menos um, da esquerda para a direita


class RecognitionService:
    def recognize(self, image: bytes) -> RecognitionResponse: ...  # arquivo como enviado
```

Exceção nova: `EmptyGalleryError`, herdando da exceção base da aplicação. A estratégia vem de `recognition/strategies.py` (`NearestNeighbor`), e **a galeria é lida do repositório a cada chamada, sem cache**. Com algumas centenas de vetores, a leitura e a busca linear levam milissegundos, e assim cadastro e exclusão valem na hora sem que os serviços dependam uns dos outros. A D12 só pede refazer em memória o modelo treinado (k-NN, SVM), e o vizinho mais próximo não treina nada.

## Fora de escopo

- Vídeo contínuo quadro a quadro: a interface reconhece uma foto por vez (D17).
- Cadastrar os vários rostos de uma foto de grupo: o cadastro continua com **um** rosto por foto (spec 001, R3).
- Agrupar ou deduplicar a mesma pessoa que aparece em vários rostos (R9).
- Trocar de estratégia em tempo de execução. A D16 fixa o vizinho mais próximo.

## Testes

- **Unitários:** A1 a A19 com detector, embedder e repositório fakes, usando vetores construídos para controlar a similaridade (inclusive o limite exato de A3). Vários rostos usam um detector roteirizado que devolve uma caixa por faixa da imagem, e um embedder que dá a cada faixa o seu vetor.
- **Integração** (#47): A1, A2 e A4 com o modelo E06 real em imagens públicas, e A12 com uma montagem de dois rostos públicos.
