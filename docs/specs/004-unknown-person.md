# Spec 004: pessoa desconhecida

- **Status:** Aceita
- **Data:** 2026-10-28
- **Dona:** Paola
- **Issue:** #38, #103 (vários rostos)
- **Relacionadas:** [003](003-face-recognition.md), [D08](../decisions/D08-threshold-selection.md), [D15](../decisions/D15-video-participants.md), [D16](../decisions/D16-chosen-model.md), [D19](../decisions/D19-multi-face-recognition.md)

## Problema

Um rosto que não está cadastrado não pode ser atribuído a ninguém. Reconhecer um estranho como uma pessoa cadastrada (falsa aceitação) é pior do que pedir para a pessoa tentar de novo (falsa rejeição), e a aplicação precisa se comportar de acordo (D08).

## Requisitos

- **R1.** Quando a similaridade de um rosto com a foto mais parecida da galeria é **menor** que o threshold, o resultado **desse rosto** é `unknown`. Numa imagem com vários rostos, cada um é decidido sozinho.
- **R2.** O resultado `unknown` **não revela** o nome da pessoa mais parecida. Ele traz só a similaridade (e, numa imagem com vários rostos, a caixa), para a interface poder mostrá-las. Isso vale rosto a rosto: o desconhecido não ganha o nome do vizinho mais próximo nem quando esse nome aparece em outro rosto da mesma imagem. Um "quase Ana" abaixo do threshold induziria a tratar a pessoa como a Ana, e o threshold existe justamente para não atribuir identidade nesse caso.
- **R3.** `unknown` é um resultado normal, não um erro: não lança exceção, e o serviço devolve, para esse rosto, um resultado com `status=UNKNOWN` e `person_name=None`.
- **R4.** O desconhecido não é cadastrado automaticamente nem guardado: nada é gravado (spec 003, R6).
- **R5.** A decisão usa o mesmo threshold da configuração do reconhecimento (padrão 0,766), para todos os rostos. Não há um segundo threshold para "desconhecido" nem um para rostos pequenos.
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
| A9 | "Ana" cadastrada e uma imagem com ela e uma pessoa não cadastrada | reconhece | a lista tem um rosto `recognized` "Ana" e um `unknown` com `person_name` `None`, e a resposta do rosto `unknown` não contém o nome de ninguém |
| A10 | uma imagem com dois rostos `unknown` | reconhece | os dois vêm sem nome, cada um com a sua similaridade e a sua caixa |
| A11 | uma imagem com vários rostos | o log é inspecionado | registra a quantidade de rostos e, por rosto, o status e a similaridade, sem nenhum nome (spec 007) |

## Interface

Sem classe nova. Usa `RecognitionResult` e `RecognitionStatus` da [spec 003](003-face-recognition.md); cada rosto de uma imagem é um `FaceResult`, que herda de `RecognitionResult` e só acrescenta a caixa. A regra vive numa função pura, testável sem o modelo, e vale para cada rosto:

```python
def decide(similarity: float, person_name: str, threshold: float) -> RecognitionResult: ...
```

A função recebe o nome da pessoa mais parecida e só o devolve quando `similarity >= threshold`.

## Expectativa de desempenho (E08, D16)

Medidos no teste privado, com a galeria de 11 pessoas: 0 de 222 desconhecidos aceitos (FAR 0%, e provavelmente abaixo de 1,4% com 95% de confiança) e 6 de 65 conhecidos rejeitados (FRR 9,2%). Isso significa que a aplicação às vezes pede para a pessoa tentar de novo, e quase nunca aceita um estranho. O README da Parte 2 registra essa limitação.

**Esses números valem para uma foto com um rosto grande** (a condição do E08). Numa foto de grupo o rosto costuma ser menor e mais inclinado, e o FAR e o FRR dessa condição **não foram medidos**. O tamanho mínimo do rosto (spec 001, R7) reduz o risco sem eliminá-lo, e a D19 registra isso como limitação.

## Fora de escopo

- Cadastrar o desconhecido direto a partir do resultado (a pessoa vai à aba de cadastro).
- Faixa de "incerto" entre dois thresholds.

## Testes

- **Unitários:** A1 a A6, A9 e A10 sobre a função `decide` e sobre o serviço com fakes, controlando a similaridade, inclusive o limite exato de A3 e A4. O A11 é verificado com `caplog`. O A8 é da interface e é testado na spec 005.
- **E2E** (#50): A7, cadastrando uma pessoa e reconhecendo outra.
