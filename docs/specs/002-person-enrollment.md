# Spec 002: cadastro de pessoa

- **Status:** Em revisão
- **Data:** 2026-10-28
- **Dona:** Paola
- **Issue:** #36
- **Relacionadas:** [001](001-face-detection.md), [003](003-face-recognition.md), [D12](../decisions/D12-embedding-storage.md), [D16](../decisions/D16-chosen-model.md)

## Problema

A aplicação precisa cadastrar uma pessoa a partir de uma ou mais fotos (seção 37), gerando os embeddings que o reconhecimento vai comparar, sem guardar as fotos (minimização de dados, seção 12) e sem criar cadastros duplicados.

## Requisitos

- **R1.** Receber um nome e de 1 a 10 fotos. Cada foto passa pelo pipeline da [spec 001](001-face-detection.md), vira um embedding de 512 dimensões com o modelo da D16 (E06) e é gravada no banco.
- **R2.** O cadastro é **atômico**: se qualquer foto falhar, nada é gravado. O erro é o da spec 001, sem indicar qual foto falhou (a interface pede para conferir as fotos).
- **R3.** O nome é obrigatório, sem espaços nas pontas, com no máximo 100 caracteres, e **único sem diferenciar maiúsculas de minúsculas**. A D12 define o nome como único, mas sem diferenciar maiúsculas não está nela: a implementação (#42) usa `COLLATE NOCASE` na coluna `persons.name`, e a D12 é atualizada com esse motivo.
- **R4.** **Duplicidade por similaridade:** se o rosto do cadastro novo é parecido demais com o de uma pessoa já cadastrada (similaridade ≥ threshold da configuração), o cadastro é recusado e informa o nome da pessoa existente, para ninguém ser cadastrada duas vezes com nomes diferentes.
- **R5.** As fotos não são gravadas. Só os embeddings, com o `model_id` (D12).
- **R6.** O cadastro vale para o reconhecimento seguinte sem reiniciar a aplicação.

## Critérios de aceitação

| # | Dado | Quando | Então |
|---|---|---|---|
| A1 | banco vazio e uma foto válida | cadastra "Ana" | existe 1 pessoa com 1 embedding de dimensão 512 e o `model_id` configurado |
| A2 | três fotos válidas | cadastra "Ana" | existe 1 pessoa com 3 embeddings |
| A3 | uma das três fotos sem rosto | cadastra "Ana" | lança `NoFaceDetectedError`, e o banco continua sem "Ana" |
| A4 | uma foto com dois rostos | cadastra "Ana" | lança `MultipleFacesError`, e o banco continua sem "Ana" |
| A5 | um arquivo que não é imagem | cadastra "Ana" | lança `InvalidImageError`, e o banco continua sem "Ana" |
| A6 | "Ana" já cadastrada | cadastra "ana" com outro rosto | lança `DuplicatePersonError` (por nome), e o cadastro existente não muda |
| A7 | "Ana" já cadastrada | cadastra "Bia" com fotos em que o rosto tem similaridade ≥ threshold com o de "Ana" | lança `DuplicatePersonError` (por similaridade) com o nome "Ana" |
| A8 | "Ana" já cadastrada | cadastra "Bia" com um rosto de similaridade < threshold com o de "Ana" | cadastra normalmente |
| A9 | nome vazio, só espaços ou com mais de 100 caracteres | cadastra | falha na validação do Pydantic, sem tocar no banco |
| A10 | lista de fotos vazia ou com mais de 10 fotos | cadastra | falha na validação do Pydantic, sem tocar no banco |
| A11 | cadastro concluído | o serviço termina | nenhuma foto foi gravada em disco, e a tabela só tem nome, embeddings e `model_id` |
| A12 | cadastro concluído | reconhece o mesmo rosto (spec 003) | a pessoa é reconhecida sem reiniciar |
| A13 | detector e modelo fakes injetados | o serviço é testado | os testes rodam sem rede, sem GPU e sem pesos |

## Interface

Em `src/app/core/`, com dependências recebidas por parâmetro e tipadas com `Protocol` (detector, embedder, repositório):

```python
class EnrollmentRequest(BaseModel):
    name: str  # strip, 1 a 100 caracteres
    photos: list[bytes]  # 1 a 10, arquivos como enviados (decodificados na spec 001)


class EnrollmentResult(BaseModel):
    person_id: int
    name: str
    embedding_count: int


class EnrollmentService:
    def enroll(self, request: EnrollmentRequest) -> EnrollmentResult: ...
```

Exceção nova: `DuplicatePersonError(name: str, reason: Literal["name", "similarity"])`, herdando da exceção base da aplicação. Os erros da spec 001 propagam sem captura. A interface trata na borda (spec 005).

## Decisões que esta spec fixa

- **Limite de 1 a 10 fotos:** a Parte 1 usou cerca de 10 fotos por pessoa na galeria, e mais que isso não melhora o vizinho mais próximo de forma relevante. O limite protege contra abuso e mantém o cadastro rápido.
- **Duplicidade por similaridade usa o mesmo threshold do reconhecimento (0,766):** assim, "já cadastrada" e "reconhecida" significam a mesma coisa. Como esse threshold foi escolhido para FAR ≤ 1%, uma pessoa diferente raramente é barrada por engano.
- **Nome único sem diferenciar maiúsculas:** evita "Ana" e "ana" como pessoas diferentes.

## Fora de escopo

- Editar uma pessoa já cadastrada (para trocar de fotos, exclui e cadastra de novo).
- Cadastro em lote e importação de pastas.
- Consentimento: a aplicação é de demonstração, e o cadastro de demonstração é só das duas autoras (D15).

## Testes

- **Unitários:** A1 a A13 com detector, embedder e banco `:memory:` fakes. Inclui o caminho de erro de cada item e a atomicidade (A3 a A5).
- **Integração** (#47): A1, A7 e A8 com o MTCNN e o FaceNet reais em imagens públicas.
