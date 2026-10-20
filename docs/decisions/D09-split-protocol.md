# D9: protocolo de split

- **Status:** Aceita, com o número de pares ajustado na implementação (ver Consequências)
- **Data:** 2026-10-18
- **Dona:** Paola

## Contexto

Os experimentos precisam separar treino, validação e teste, e o teste não pode influenciar treino, hiperparâmetros, threshold nem a escolha do modelo (seção 29). Em reconhecimento facial, o grupo também precisa verificar se a mesma imagem, uma imagem quase idêntica ou uma transformação dela atravessa os conjuntos (seção 30). E01, E02, E04 e E06 precisam ser comparados nos mesmos pares de teste.

## Decisão

**Público (LFW):**
- As identidades com 2 fotos ou mais são divididas **por identidade** em treino, validação e teste (70/15/15, seed 42), separadamente em cada faixa de número de fotos (2–4, 5–9 e 10+). Nenhuma pessoa aparece em mais de um split.
- As identidades com uma foto só vão metade para a validação e metade para o teste, como impostoras nos pares negativos e como desconhecidas extras do privado (200 fotos por split).
- **Pares:** todas as combinações de duas fotos da mesma pessoa, com no máximo 20 por identidade, e o mesmo número de pares negativos. Nos negativos, sorteia-se primeiro a identidade e depois a foto, para quem tem muitas fotos não dominar.
- O "melhor de E01 a E06" que segue para o E07 é escolhido pelo EER da validação, nunca do teste.

**Privado:**
- A sessão 1 é a galeria. As fotos da sessão 2 de cada pessoa da galeria vão metade para a validação e metade para o teste.
- 2 participantes ficam fora da galeria como desconhecidas, uma na validação e outra no teste, com todas as fotos das duas sessões.
- O split privado não é versionado: é gerado em tempo de execução a partir de `data/private/<ID>/session{1,2}/`, com seed 42.

**Leakage:**
- Dedupe por **dHash de 64 bits** nos recortes de 160×160: distância de Hamming ≤ 4 conta como quase-duplicata. Entre splits do LFW, a cópia em validação ou teste sai. No privado, a mesma checagem roda entre a sessão 1 e a sessão 2.
- Augmentations só no dataloader de treino.

## Alternativas consideradas

- **Pares oficiais do LFW (View 2):** contêm identidades que entram no treino do fine-tuning, o que seria leakage no E04 e no E06.
- **Split por imagem:** a mesma pessoa estaria no treino e no teste, e o teste mediria memorização.
- **3.000 + 3.000 pares por split, a meta inicial** (o tamanho do View 2): com o split por identidade e o limite de 20, a validação e o teste só têm cerca de 1.450 pares positivos possíveis. Subir o limite deixaria poucas identidades dominarem os pares, e desbalancear tornaria a accuracy dependente da proporção.

## Consequências

- **Números reais** (`python scripts/make_splits.py`):

  | Split | Identidades | Fotos | Pares |
  |---|---|---|---|
  | treino | 1.176 | 6.439 | — |
  | validação | 2.287 | 3.490 | 1.447 + 1.447 |
  | teste | 2.286 | 3.303 | 1.445 + 1.445 |

- Com cerca de 1.450 pares negativos, o TAR@FAR=1% se apoia em uns 14 falsos positivos. O intervalo de confiança da D7 mostra essa incerteza.
- Os splits públicos são versionados em `data/splits/` e regenerados de forma determinística pelo [scripts/make_splits.py](../../scripts/make_splits.py) ([src/facial_recognition_system/data/splits.py](../../src/facial_recognition_system/data/splits.py)).
- A checagem por dHash e o relatório de leakage entram com a #15.
