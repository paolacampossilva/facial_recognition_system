# Roteiro do vídeo da Parte 1

Duração alvo: **cerca de 13 minutos** (soma das cenas abaixo). O enunciado pede problema, datasets, arquitetura, hipóteses, experimentos, treinamento, resultados, comparação, conclusão e escolha do modelo (seção 35). Cada uma apresenta o que fez (division.md e issue #32): a **Paola** apresenta dados, protocolo, métricas e threshold, e a **Valentina** apresenta o problema, a arquitetura, o treino, os resultados e a escolha. Só nós duas aparecemos (D15).

Formato sugerido: gravação de tela com a câmera num canto. Cada uma grava as próprias cenas, e a Valentina junta tudo na edição. O texto falado é um guia, não precisa ser lido palavra por palavra.

## Antes de gravar (privacidade)

- [ ] Cofre VeraCrypt **desmontado**: nada de `V:\`, `data/private/`, `faces/private/` ou termos na tela.
- [ ] WhatsApp Web, e-mail e a planilha de controle **fechados**.
- [ ] Terminal limpo, sem histórico com caminhos de fotos ou nomes.
- [ ] Na tela, só código, documentação e os gráficos de `experiments/figures/`. Os do privado têm só IDs anônimos.
- [ ] Nenhuma foto de participante no vídeo. Se precisar de uma imagem de exemplo, usar uma de nós duas.

## Cenas

| # | Cena | Quem | Tempo | Na tela |
|---|---|---|---|---|
| 1 | Abertura e problema | Valentina | 0:45 | README, "Visão geral" e "Pipeline previsto" |
| 2 | Dados | Paola | 1:30 | README, "Conjuntos de dados" e "Preparação dos dados"; `data/README.md` |
| 3 | Protocolo e leakage | Paola | 1:30 | README, "Protocolo de avaliação"; `data/splits/lfw_leakage_report.json` |
| 4 | Métricas e threshold | Paola | 1:45 | README, "Métricas" e "Threshold"; `E06_hist.png` |
| 5 | Arquitetura e modelos | Valentina | 1:30 | `models/embedders.py`; D4 |
| 6 | Hipóteses e treino | Valentina | 1:45 | `configs/E04.yaml` e `E06.yaml`; `training/`; D5 |
| 7 | Resultados e comparação | Valentina | 2:00 | Tabela do README; `comparison_roc.png`; `E02_hist.png` e `E06_hist.png` |
| 8 | Dataset privado: E07 e E08 | Valentina | 1:15 | `E08_confusion.png` e `E08_threshold.png` |
| 9 | Escolha do modelo e conclusão | Valentina | 1:00 | D16; README, "Resultados e análise" |
| 10 | Encerramento | as duas | 0:20 | README |

## 1. Abertura e problema (Valentina, 0:45)

> Oi, eu sou a Valentina, e esta é a Paola. Este é o nosso trabalho de reconhecimento facial da disciplina de Inteligência Artificial. Na Parte 1, a pergunta é: qual modelo e qual configuração reconhecem melhor um rosto, e como decidir quando a pessoa é desconhecida?
>
> O pipeline é sempre o mesmo: a foto passa pelo detector MTCNN, o rosto é recortado em 160 por 160, uma rede neural transforma esse recorte num vetor, o embedding, e dois rostos são comparados pela similaridade de cosseno entre os vetores. Acima de um threshold, é a mesma pessoa; abaixo, é desconhecida.
>
> Como rosto é dado biométrico, sensível pela LGPD, as fotos das participantes nunca saem do notebook do grupo, e no GitHub só entram código e números agregados.

## 2. Dados (Paola, 1:30)

> Usamos dois conjuntos. O público é o LFW, com 13.233 fotos de 5.749 pessoas. O site oficial saiu do ar, então o nosso script baixa a cópia do figshare, a mesma do scikit-learn, e confere o SHA-256. Nenhuma imagem vai para o repositório, só o script.
>
> Todas as fotos passam pelo MTCNN, com probabilidade mínima de 0,9. No LFW, só uma foto ficou sem rosto. Para o fine-tuning, usamos só o split de treino: pessoas com pelo menos 4 fotos e no máximo 40 cada, para ninguém dominar o treino. São 433 pessoas e 3.961 fotos.
>
> O privado tem 13 participantes adultas que assinaram o termo de consentimento, com cerca de 22 fotos cada, em duas sessões em cômodos diferentes. As pastas usam IDs anônimos, como P00, e tudo fica num cofre criptografado no notebook da 4060. Fotos em HEIC foram convertidas para JPEG sem metadados, para não guardar localização.

## 3. Protocolo e leakage (Paola, 1:30)

> O split do LFW é por identidade: 70% das pessoas no treino, 15% na validação e 15% no teste, e ninguém aparece em dois splits. Assim, o modelo é testado em pessoas que ele nunca viu. A validação e o teste têm cerca de 1.450 pares da mesma pessoa e o mesmo número de pares de pessoas diferentes, e todos os experimentos usam os mesmos pares.
>
> Para checar leakage, comparamos hashes das fotos entre os splits e achamos 2 fotos repetidas no próprio LFW: a mesma imagem com o nome de duas pessoas. Elas saíram dos splits.
>
> A regra mais importante: o teste nunca escolhe nada. O threshold, a época do treino, o modelo e a estratégia são escolhidos na validação. O teste só mede o resultado final.
>
> No privado, a sessão 1 de cada pessoa é a galeria, o cadastro, e a sessão 2 é dividida entre validação e teste. Duas participantes ficam fora da galeria como desconhecidas, uma em cada split, junto com 200 rostos do LFW.

## 4. Métricas e threshold (Paola, 1:45)

> No LFW, medimos verificação: dado um par de fotos, o modelo diz se é a mesma pessoa. A métrica principal é o EER, o ponto em que o modelo erra igual para os dois lados: aceitar quem não é e rejeitar quem é. Também reportamos ROC-AUC, a taxa de acerto com no máximo 1% de falsas aceitações, accuracy e F1, sempre com intervalo de confiança de 95% por bootstrap.
>
> No privado, medimos identificação open-set: o sistema precisa dizer quem é, ou dizer que é desconhecido. Aí olhamos o FAR, quantos desconhecidos foram aceitos, o FRR, quantas pessoas cadastradas foram rejeitadas, e a identificação errada, quando confunde uma pessoa com outra.
>
> *(mostrar `E06_hist.png`)* Este histograma mostra a similaridade dos pares: em azul, a mesma pessoa; em laranja, pessoas diferentes. O threshold fica no meio. No LFW, usamos o threshold do EER. No privado e na aplicação, usamos o menor threshold com FAR de no máximo 1%, porque aceitar um estranho é pior do que pedir para a pessoa tentar de novo.

## 5. Arquitetura e modelos (Valentina, 1:30)

> Comparamos dois backbones com o mesmo tamanho, cerca de 23,5 milhões de parâmetros. O A é o FaceNet, pré-treinado em rostos do VGGFace2, que gera um embedding de 512 dimensões. O B é a ResNet-50, pré-treinada em objetos do ImageNet, sem a última camada, com embedding de 2.048 dimensões.
>
> *(mostrar `embedders.py`)* Os dois ficam atrás da mesma interface: recebem o recorte e devolvem um vetor com norma 1. O resto do código não precisa saber qual modelo está usando.
>
> A ResNet está aqui para responder uma pergunta: quanto vale ter sido treinado em rostos?

## 6. Hipóteses e treino (Valentina, 1:45)

> As hipóteses: o E01, o FaceNet sem treino, é um baseline forte; o E02, a ResNet do ImageNet, é ruim para rostos; e o E04 e o E06 testam se um fine-tuning no LFW melhora o FaceNet.
>
> *(mostrar `E04.yaml`)* O fine-tuning é parcial: só as três últimas partes do FaceNet treinam, 2,5 milhões dos 23,5 milhões de parâmetros. O resto fica congelado, inclusive as estatísticas das camadas de normalização, para o modelo não esquecer o que já sabia.
>
> Comparamos duas funções de perda nas mesmas condições. No E04, o ArcFace: uma cabeça tenta adivinhar quem é a pessoa, com uma margem que obriga cada rosto a ficar perto das fotos da mesma pessoa. No E06, a Triplet Loss: compara trios, a foto, outra da mesma pessoa e uma de outra pessoa, usando sempre os casos mais difíceis do lote. Para isso, cada lote tem 32 pessoas com 4 fotos cada.
>
> O treino para pelo EER da validação: medimos antes de treinar, como referência, e depois de cada época, e paramos depois de 5 épocas sem melhora. Cada treino levou cerca de 2 minutos na RTX 4060.

## 7. Resultados e comparação (Valentina, 2:00)

> *(mostrar a tabela do README)* No teste do LFW: o FaceNet sem treino tem EER de 1,11%. A ResNet do ImageNet tem 31%. Com o mesmo tamanho de rede, o pré-treino em rostos é a diferença entre errar 1 par em 90 e errar quase 1 em 3.
>
> *(mostrar `E02_hist.png` e `E06_hist.png`)* Dá para ver no histograma: na ResNet, os dois grupos se sobrepõem, ela acha todo mundo parecido. No FaceNet, eles ficam separados.
>
> E o fine-tuning? Na validação, o EER caiu de 0,76% para 0,48% com o ArcFace e para 0,41% com a Triplet. Mas no teste o ArcFace ficou em 1,18% e a Triplet em 1,04%. Comparando par a par com o E01, nenhum dos dois tem melhora estatisticamente significativa. A Triplet errou 3 pares a menos em 2.888.
>
> *(mostrar `comparison_roc.png`)* Na curva ROC, as três curvas do FaceNet ficam praticamente coladas.
>
> Por quê? O FaceNet já acerta 98,75% no LFW, e 433 pessoas ensinam pouco a um modelo treinado com milhares. A melhor época foi sempre a primeira; depois, o modelo começa a decorar as pessoas do treino. É overfitting, e o early stopping cortou.

## 8. Dataset privado: E07 e E08 (Valentina, 1:15)

> No privado, o E07 comparou quatro formas de comparar um rosto com a galeria. O vizinho mais próximo, que compara com a foto mais parecida, ganhou. O SVM foi bem pior, porque cada pessoa tem só cerca de 10 fotos no cadastro, pouco para treinar um classificador.
>
> *(mostrar `E08_threshold.png`)* O E08 escolheu o threshold na validação: 0,766, o menor com no máximo 1% de desconhecidos aceitos. É bem mais alto que o do LFW, 0,445, porque as desconhecidas foram fotografadas nas mesmas condições que as cadastradas.
>
> *(mostrar `E08_confusion.png`)* No teste: 97,9% de acerto, nenhum desconhecido aceito em 222 tentativas e nenhuma pessoa confundida com outra. Os erros foram 6 fotos de pessoas cadastradas rejeitadas, o erro menos grave para a aplicação.

## 9. Escolha do modelo e conclusão (Valentina, 1:00)

> A regra para escolher o modelo foi definida antes dos resultados: o menor EER de validação. Isso dá o E06, o FaceNet com Triplet, com o vizinho mais próximo e o threshold de 0,766. É essa configuração que vai para a aplicação da Parte 2.
>
> Uma ressalva honesta: a mesma validação escolheu a época e o modelo, então o número de validação do E06 é otimista. No teste, ele empata com o FaceNet sem treino.
>
> As conclusões: o pré-treino em rostos é o que mais importa; com poucos dados, o fine-tuning não trouxe ganho comprovado; e o threshold precisa ser calibrado no domínio da aplicação. As limitações, como o LFW perto do limite e a galeria pequena, estão no README.

## 10. Encerramento (as duas, 0:20)

> **Paola:** O código, os experimentos e as instruções para reproduzir estão no repositório.
>
> **Valentina:** Na Parte 2, essa configuração vira uma aplicação de cadastro e reconhecimento pela câmera. Obrigada!

## Para o ensaio (seção 5)

O enunciado exige que as duas saibam explicar tudo, então vale cada uma conseguir responder sobre a parte da outra:

- **Para a Paola:** por que só o último bloco do FaceNet treina? Qual a diferença entre ArcFace e Triplet? Por que o early stopping usa o EER e não a loss de validação?
- **Para a Valentina:** por que o split é por identidade? O que é o EER? Por que o threshold do privado é FAR de no máximo 1%, e não 0,1%?
