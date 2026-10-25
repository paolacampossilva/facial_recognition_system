# D16: modelo escolhido para a aplicação

- **Status:** Aceita
- **Data:** 2026-10-22
- **Dona:** Juntas

## Contexto

A Parte 2 usa a melhor configuração encontrada na Parte 1, e usar outra exige justificativa (seção 36). O protocolo (D9) fixou a regra antes dos resultados: o modelo que segue para o E07 é o de **menor EER na validação** entre E01 e E06, e o teste nunca escolhe nada (seção 29). A aplicação precisa de três escolhas: o modelo que gera o embedding, a estratégia de comparação com a galeria (E07, D6) e o threshold de "pessoa desconhecida" (E08, D8).

Resultados da verificação no LFW (`experiments/results/`, avaliação do #20/#26, threshold do EER escolhido na validação):

| Experimento | EER validação | EER teste (IC 95%) | ROC-AUC teste | Accuracy teste |
|---|---|---|---|---|
| E01 FaceNet pré-treinado | 0,76% | 1,11% (0,76–1,59) | 0,9994 | 98,75% |
| E02 ResNet-50 ImageNet | 31,93% | 31,06% (29,60–32,86) | 0,7491 | 68,66% |
| E04 FaceNet + ArcFace | 0,48% | 1,18% (0,76–1,59) | 0,9992 | 98,68% |
| **E06 FaceNet + Triplet** | **0,41%** | 1,04% (0,69–1,42) | 0,9995 | 98,86% |

## Decisão

1. **Modelo: E06**, o FaceNet com fine-tuning parcial e Triplet Loss, na melhor época (época 1). É o menor EER de validação, como manda o protocolo.
2. **Estratégia de comparação: vizinho mais próximo** (cosseno com a foto mais parecida da galeria), escolhida no E07 pelo maior F1 macro na validação privada.
3. **Threshold: 0,766**, o menor com FAR ≤ 1% na validação privada (E08, D8). O FRR de validação ficou em 7,9%, abaixo do limite de 20%, então o fallback para o EER não foi necessário.

No teste privado, com esse modelo, essa estratégia e esse threshold (E08):

| Accuracy | F1 macro | FAR (desconhecidos aceitos) | FRR (conhecidos rejeitados) | Identificação errada |
|---|---|---|---|---|
| 97,9% | 0,946 | **0%** (0 de 222) | 9,2% (6 de 65) | **0%** |

O sistema não aceitou nenhum desconhecido e não confundiu nenhuma pessoa cadastrada com outra. Os erros foram só rejeições de pessoas cadastradas, o lado menos grave para a aplicação (D8).

## Alternativas consideradas

- **E01, o FaceNet pré-treinado sem fine-tuning:** no teste do LFW, o E06 não é estatisticamente melhor que o E01 (diferença de EER de −0,07 p.p., IC 95% de −0,45 a +0,21, bootstrap pareado, D5), e o E01 é mais simples (sem checkpoint próprio). Ficou de fora porque trocar a regra **depois** de ver o teste seria usar o teste para escolher (seção 29). A regra da validação foi fixada antes dos resultados e foi seguida.
- **E04 (ArcFace):** EER de validação maior que o do E06 (0,48% contra 0,41%) e equivalente ao E01 no teste.
- **E02 (ResNet-50 ImageNet):** EER de 31% em validação e teste; não reconhece rostos.
- **Protótipo ou k-NN no lugar do vizinho mais próximo:** ficaram logo atrás no E07 (F1 macro de validação de 0,934 contra 0,942). O SVM foi bem pior (até 0,656), porque cada pessoa tem só cerca de 10 fotos na galeria, poucas para treinar um classificador.
- **Threshold do EER (0,744) no lugar do de FAR ≤ 1%:** no teste, aceitaria o mesmo número de desconhecidos (0) e rejeitaria menos conhecidos (7,7%), mas é escolhido para equilibrar os erros, e a D8 prioriza não aceitar estranhos. O resultado de teste não muda a regra.

## Consequências

- **Ressalva sobre a escolha:** nos modelos com fine-tuning, a mesma validação escolheu a melhor época (early stopping) e depois o modelo, então o EER de validação do E04 e do E06 é otimista. Isso aparece no teste, onde E01, E04 e E06 empatam. A escolha segue a regra, mas o ganho do E06 sobre o E01 não está comprovado.
- **O threshold do LFW não serve para a aplicação:** 0,445 no LFW contra 0,766 no privado, provavelmente porque os desconhecidos do privado foram fotografados nas mesmas condições das pessoas cadastradas e se parecem mais com elas. Por isso o threshold da aplicação vem do E08, e não da verificação pública.
- **Limitações do E08:** a galeria tem 11 pessoas, e o FAR é medido com 222 desconhecidos (22 fotos de uma participante e 200 do LFW). Com 0 aceitações, o FAR real pode ser maior que zero; com essa amostra, só dá para dizer que ele provavelmente fica abaixo de cerca de 1,4% (regra de três, 3/222, com 95% de confiança). Uma seed só (D11).
- **Na aplicação (Parte 2):** carregar o FaceNet com o checkpoint `checkpoints/E06/best.pt`, comparar pelo vizinho mais próximo e usar o threshold 0,766, lido da configuração (spec 007). O checkpoint foi treinado só com o LFW, então não é dado privado (seção 19), mas fica fora do Git pelo tamanho. Ele é refeito com `python scripts/train.py --config configs/E06.yaml`, e o README da Parte 2 explica isso.
- Resultados em `experiments/results/E07.json` e `E08.json` (só números agregados) e gráficos em `experiments/figures/E08_confusion.png` e `E08_threshold.png`.
