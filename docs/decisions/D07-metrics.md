# D7: métricas

- **Status:** Aceita
- **Data:** 2026-10-18
- **Dona:** Paola

## Contexto

O enunciado pede métricas adequadas (accuracy, precision, recall, F1, ROC-AUC, FAR, FRR, EER, matriz de confusão, que no privado vira a tabela de totais) e a análise das distribuições de similaridade de mesma pessoa e de pessoas diferentes (seção 31). O projeto tem dois tipos de avaliação: **verificação** nos pares públicos (E01 a E06) e **identificação open-set** no dataset privado (E07 e E08), em que o sistema pode responder "desconhecido".

## Decisão

**Verificação** (pares do LFW):
- ROC-AUC, EER e TAR@FAR=1%, que não dependem de threshold;
- accuracy e F1 no threshold escolhido na validação (D8), para a tabela da seção 35;
- histogramas de similaridade de mesma pessoa e de pessoas diferentes;
- intervalo de confiança de 95% por bootstrap, com 1.000 reamostragens dos pares. Os pares positivos e negativos são reamostrados separadamente, para cada reamostra manter o balanceamento.

**Identificação open-set** (dataset privado):
- accuracy e F1 macro, com "desconhecido" como uma classe;
- FAR (tentativas de desconhecidos aceitas como alguém) e FRR (tentativas de conhecidos rejeitadas);
- taxa de identificação errada (conhecido atribuído à pessoa errada), que FAR e FRR não capturam;
- tabela de totais (conhecidos corretos, atribuídos a outra pessoa e rejeitados; desconhecidos aceitos e rejeitados), sem nenhuma linha por participante.

## Alternativas consideradas

- **Accuracy em 10 folds do View 2**, o protocolo clássico do LFW: não se aplica, porque os pares oficiais não são usados (D9).
- **Precision e recall separados:** o F1 já resume os dois, e FAR e FRR são as taxas de erro com sentido direto para a aplicação.
- **Só accuracy:** depende do threshold e da proporção de pares, e esconde o comportamento em FAR baixo, que é o que importa para não reconhecer um estranho.

## Consequências

- Status: a matriz por ID foi publicada na primeira versão do E08 e trocada por totais em 24/10/2026, porque é resultado individualizado (enunciado, seções 17 e 35).
- Resultados do privado são sempre agregados. A validação de entrada rejeita rótulos que não sejam IDs anônimos (`P01`) ou "unknown", sem repetir o nome rejeitado na mensagem de erro (LGPD, seção 18).
- Implementado em [src/facial_recognition_system/evaluation/](../../src/facial_recognition_system/evaluation/) (PR #70): `verification.py`, `open_set.py` e `scores.py`. As métricas e os intervalos vão para `metrics.val` e `metrics.test` no JSON de cada experimento.
- Teste de ponta a ponta com o E01: EER de 0,76% na validação e de 1,11% (IC 95% de 0,73% a 1,56%) no teste.
