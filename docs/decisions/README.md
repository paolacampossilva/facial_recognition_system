# Decisões de arquitetura (ADRs)

Cada decisão D1 a D15 do [docs/division.md](../division.md) tem um ADR curto aqui, escrito pela dona da decisão. O arquivo é numerado pela decisão (`D01-...md`), para as duas poderem escrever em paralelo sem disputar números.

## Formato

```markdown
# DNN: título

- **Status:** Aceita | Substituída por DNN | Em revisão
- **Data:** AAAA-MM-DD
- **Dona:** Paola | Valentina | Juntas

## Contexto
O problema, as restrições e o que o enunciado exige (com a seção).

## Decisão
O que foi escolhido, com os valores numéricos.

## Alternativas consideradas
Cada alternativa e por que ficou de fora.

## Consequências
O que muda no código e nos experimentos, limitações e onde está implementado.
```

Se um valor mudar durante a implementação, o ADR é atualizado com o motivo, e não apagado.

## Índice

| # | Decisão | Dona | ADR |
|---|---|---|---|
| D1 | Dataset público de benchmark | Paola | [D01-benchmark-dataset.md](D01-benchmark-dataset.md) |
| D2 | Dataset público para fine-tuning | Paola | [D02-fine-tuning-dataset.md](D02-fine-tuning-dataset.md) |
| D3 | Detector e alinhamento | Paola | [D03-face-detection-alignment.md](D03-face-detection-alignment.md) |
| D4 | Backbones comparados | Valentina | [D04-backbones.md](D04-backbones.md) |
| D5 | Funções de perda | Valentina | [D05-loss-functions.md](D05-loss-functions.md) |
| D6 | Estratégia de comparação | Paola | [D06-comparison-strategy.md](D06-comparison-strategy.md) |
| D7 | Métricas | Paola | [D07-metrics.md](D07-metrics.md) |
| D8 | Escolha do threshold | Paola | [D08-threshold-selection.md](D08-threshold-selection.md) |
| D9 | Protocolo de split | Paola | [D09-split-protocol.md](D09-split-protocol.md) |
| D10 | Hardware | Valentina | [D10-hardware.md](D10-hardware.md) |
| D11 | Registro de experimentos | Valentina | [D11-experiment-tracking.md](D11-experiment-tracking.md) |
| D12 | Armazenamento de embeddings | Paola | a escrever |
| D13 | Interface | Valentina | [D13-interface.md](D13-interface.md) |
| D14 | Repositório | Juntas | [D14-repository.md](D14-repository.md) |
| D15 | Quem aparece no vídeo | Juntas | [D15-video-participants.md](D15-video-participants.md) |
| D16 | Modelo escolhido para a aplicação | Juntas | [D16-chosen-model.md](D16-chosen-model.md) |
