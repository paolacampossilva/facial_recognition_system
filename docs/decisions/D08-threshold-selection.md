# D8: escolha do threshold

- **Status:** Aceita
- **Data:** 2026-10-18
- **Dona:** Paola

## Contexto

O sistema precisa poder responder "pessoa desconhecida", e o threshold dessa decisão precisa ser justificado experimentalmente (seção 32). O teste não pode ser usado para escolher o threshold (seção 29). Os erros também não pesam igual na aplicação: reconhecer um estranho como uma pessoa cadastrada (falsa aceitação) é pior do que pedir para a pessoa tentar de novo (falsa rejeição).

## Decisão

O threshold é escolhido **sempre na validação** e aplicado no teste **sem reajuste**. Uma face é aceita quando `similaridade >= threshold`.

- **Público (E01 a E06):** o threshold do EER nos pares de validação, onde FAR e FRR se igualam. É o critério neutro para comparar modelos.
- **Privado e aplicação (E08):** o **menor threshold com FAR ≤ 1%** na validação, reportando o FRR.
- **Fallback:** se esse threshold deixar o FRR de validação acima de 20%, usa-se o threshold do EER, e a escolha é justificada na análise do E08.
- Para a validação privada ter tentativas de desconhecidos suficientes para medir FAR de 1%, entram 200 rostos de identidades do LFW com uma foto só em cada split (D9).

## Alternativas consideradas

- **Threshold que maximiza a accuracy na validação:** depende da proporção de pares positivos e negativos e não controla o tipo de erro.
- **Threshold fixo (por exemplo, 0,5):** não se adapta ao modelo, e a seção 32 pede justificativa experimental.
- **FAR ≤ 0,1%:** com cerca de 200 tentativas de desconhecidos, não dá para medir FAR tão baixo, porque uma única aceitação já seria 0,5%.

## Consequências

- O FAR do privado tem resolução de 0,5% (1 em 200 tentativas de desconhecidos), e isso vai na análise do E08.
- O valor e o critério ficam registrados no JSON do experimento, em `threshold` (`value`, `criterion`, `chosen_on: "val"`), e a aplicação lê o threshold da configuração (spec 007).
- Implementado em [src/facial_recognition_system/evaluation/threshold.py](../../src/facial_recognition_system/evaluation/threshold.py) (PR #70): `choose_eer_threshold` e `choose_far_threshold`, com o fallback.
- Exemplo com o E01 na validação pública: o threshold do EER é 0,450 (FAR = FRR = 0,76%), e o de FAR ≤ 1% é 0,438 (FAR de 0,97% e FRR de 0,69%).
