# Spec 004: pessoa desconhecida

- **Status:** Em revisão
- **Data:** 2026-10-28
- **Dona:** Paola
- **Issue:** #38
- **Relacionadas:** [003](003-face-recognition.md), [D08](../decisions/D08-threshold-selection.md), [D15](../decisions/D15-video-participants.md), [D16](../decisions/D16-chosen-model.md)

## Problema

Um rosto que não está cadastrado não pode ser atribuído a ninguém. Reconhecer um estranho como uma pessoa cadastrada (falsa aceitação) é pior do que pedir para a pessoa tentar de novo (falsa rejeição), e a aplicação precisa se comportar de acordo (D08).

## Requisitos

- **R1.** Quando a similaridade com a foto mais parecida da galeria é **menor** que o threshold, o resultado é `unknown`.
- **R2.** O resultado `unknown` **não revela** o nome da pessoa mais parecida. Ele traz só a similaridade, para a interface poder mostrá-la. Um "quase Ana" abaixo do threshold induziria a tratar a pessoa como a Ana, e o threshold existe justamente para não atribuir identidade nesse caso.
- **R3.** `unknown` é um resultado normal, não um erro: não lança exceção, e o serviço devolve `RecognitionResult` com `status=UNKNOWN` e `person_name=None`.
- **R4.** O desconhecido não é cadastrado automaticamente nem guardado: nada é gravado (spec 003, R6).
- **R5.** A decisão usa o mesmo threshold da configuração do reconhecimento (padrão 0,766). Não há um segundo threshold para "desconhecido".
- **R6.** Quando a galeria está vazia, o comportamento é o da spec 003 (A7, `EmptyGalleryError`), e não `unknown`.

## Critérios de aceitação

| # | Dado | Quando | Então |
|---|---|---|---|
| A1 | "Ana" cadastrada e o rosto de uma pessoa não cadastrada | reconhece | status `unknown`, `person_name` é `None` |
| A2 | o resultado é `unknown` | o resultado é inspecionado | não contém o nome nem o id de nenhuma pessoa da galeria |
| A3 | similaridade logo abaixo do threshold (ex.: 0,765 com threshold 0,766) | reconhece | `unknown` |
| A4 | similaridade igual ao threshold | reconhece | `recognized` (spec 003, A3) |
| A5 | o resultado é `unknown` | o serviço termina | nenhuma exceção foi lançada, e o banco não mudou |
| A6 | o threshold sobe na configuração | reconhece um rosto que era `recognized` | pode passar a `unknown`, sem mudar código |
| A7 | uma de nós, antes de se cadastrar, aparece na câmera com a outra já cadastrada | reconhece | `unknown` (é o cenário do vídeo, D15) |
| A8 | o resultado é `unknown` | a UI mostra o resultado | exibe "Pessoa desconhecida" e a similaridade, sem sugerir um nome |

## Interface

Sem classe nova. Usa `RecognitionResult` e `RecognitionStatus` da [spec 003](003-face-recognition.md). A regra vive numa função pura, testável sem o modelo:

```python
def decide(similarity: float, person_name: str, threshold: float) -> RecognitionResult: ...
```

A função recebe o nome da pessoa mais parecida e só o devolve quando `similarity >= threshold`.

## Expectativa de desempenho (E08, D16)

Medidos no teste privado, com a galeria de 11 pessoas: 0 de 222 desconhecidos aceitos (FAR 0%, e provavelmente abaixo de 1,4% com 95% de confiança) e 6 de 65 conhecidos rejeitados (FRR 9,2%). Isso significa que a aplicação às vezes pede para a pessoa tentar de novo, e quase nunca aceita um estranho. O README da Parte 2 registra essa limitação.

## Fora de escopo

- Cadastrar o desconhecido direto a partir do resultado (a pessoa vai à aba de cadastro).
- Faixa de "incerto" entre dois thresholds.

## Testes

- **Unitários:** A1 a A6 sobre a função `decide` e sobre o serviço com fakes, controlando a similaridade, inclusive o limite exato de A3 e A4. O A8 é da interface e é testado na spec 005.
- **E2E** (#50): A7, cadastrando uma pessoa e reconhecendo outra.
