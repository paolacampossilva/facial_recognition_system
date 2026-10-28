# Spec 003: reconhecimento facial

- **Status:** Em revisão
- **Data:** 2026-10-28
- **Dona:** Paola
- **Issue:** #37
- **Relacionadas:** [001](001-face-detection.md), [004](004-unknown-person.md), [D08](../decisions/D08-threshold-selection.md), [D12](../decisions/D12-embedding-storage.md), [D16](../decisions/D16-chosen-model.md)

## Problema

Dada uma imagem, a aplicação precisa dizer se o rosto pertence a uma pessoa cadastrada e a qual (seção 38), usando o modelo, a estratégia e o threshold escolhidos na Parte 1 (seção 36).

## Requisitos

- **R1.** Pipeline: imagem → spec 001 → embedding (E06) → comparação com a galeria → threshold → resultado.
- **R2.** Estratégia de comparação: **vizinho mais próximo** (similaridade de cosseno com a foto mais parecida da galeria), conforme a D16.
- **R3.** O threshold vem da configuração (spec 007), nunca fixo no código. O valor padrão é **0,766** (E08). Uma face é aceita quando `similaridade >= threshold` (D08).
- **R4.** A galeria são os embeddings do banco com o `model_id` igual ao configurado. Embeddings de outro modelo são ignorados (D12).
- **R5.** O resultado é um modelo Pydantic com o status, o nome (só se reconhecida) e a similaridade.
- **R6.** O reconhecimento não grava nada (nem a imagem, nem o embedding da consulta).
- **R7.** Galeria vazia não é erro de reconhecimento: é um estado que a interface mostra como "Nenhuma pessoa cadastrada" (seção 43).

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
| A8 | uma imagem sem rosto, com vários rostos ou inválida | reconhece | propaga `NoFaceDetectedError`, `MultipleFacesError` ou `InvalidImageError` da spec 001 |
| A9 | qualquer reconhecimento | o serviço termina | o banco e o disco não mudaram |
| A10 | uma pessoa cadastrada ou excluída logo antes | reconhece | o resultado já reflete a mudança (sem reiniciar) |
| A11 | detector e embedder fakes que devolvem vetores fixos | o serviço é testado | os testes rodam sem rede, sem GPU e sem pesos |

## Interface

```python
class RecognitionStatus(StrEnum):
    RECOGNIZED = "recognized"
    UNKNOWN = "unknown"


class RecognitionResult(BaseModel):
    status: RecognitionStatus
    person_name: str | None  # None quando unknown
    similarity: float  # cosseno com a foto mais parecida da galeria


class RecognitionService:
    def recognize(self, image: bytes) -> RecognitionResult: ...  # arquivo como enviado
```

Exceção nova: `EmptyGalleryError`, herdando da exceção base da aplicação. A estratégia vem de `recognition/strategies.py` (`NearestNeighbor`), e **a galeria é lida do repositório a cada chamada, sem cache**. Com algumas centenas de vetores, a leitura e a busca linear levam milissegundos, e assim cadastro e exclusão valem na hora sem que os serviços dependam uns dos outros. A D12 só pede refazer em memória o modelo treinado (k-NN, SVM), e o vizinho mais próximo não treina nada.

## Fora de escopo

- Vídeo contínuo quadro a quadro: a interface reconhece uma foto por vez (D13).
- Reconhecer mais de uma pessoa por imagem (a spec 001 rejeita imagens com vários rostos).
- Trocar de estratégia em tempo de execução. A D16 fixa o vizinho mais próximo.

## Testes

- **Unitários:** A1 a A11 com detector, embedder e repositório fakes, usando vetores construídos para controlar a similaridade (inclusive o limite exato de A3).
- **Integração** (#47): A1, A2 e A4 com o modelo E06 real em imagens públicas.
