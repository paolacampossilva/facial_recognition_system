# Reconhecedor Facial

## Trabalho Prático — Sistema de Reconhecimento Facial

**Disciplina de Inteligência Artificial**

**Trabalho individual ou em duplas**

> **Atraso na entrega:** perde-se um ponto por dia (serão aceitos, no máximo, dois dias de atraso; após isso, não serão aceitas entregas).

## 1. Objetivo

O objetivo deste trabalho é desenvolver um sistema de reconhecimento facial baseado em técnicas modernas de Inteligência Artificial, percorrendo o ciclo completo de construção de uma solução de IA:

```text
Problema
   ↓
Dados
   ↓
Experimentação
   ↓
Treinamento / Fine-tuning
   ↓
Avaliação
   ↓
Escolha do melhor modelo
   ↓
Engenharia de Software
   ↓
Aplicação
   ↓
Validação
```

O trabalho será desenvolvido em duas partes sequenciais:

- Parte 1 — Experimentação, treinamento e fine-tuning
- Parte 2 — Desenvolvimento da aplicação

A Parte 1 deverá determinar experimentalmente qual abordagem produz os melhores resultados.

A Parte 2 deverá utilizar a solução escolhida na Parte 1 para construir uma aplicação funcional de reconhecimento facial.

## 2. Entregas

Para cada parte do trabalho, deverão ser entregues:

- Link para o repositório GitHub do projeto;
- Link para um vídeo demonstrativo.

Na Parte 1, o vídeo deverá apresentar principalmente os experimentos e resultados.

Na Parte 2, o vídeo deverá apresentar a aplicação funcionando.

## 3. Desenvolvimento assistido por IA

É permitido e incentivado o uso de ferramentas de desenvolvimento assistido por IA, tais como:

- Claude Code;
- OpenAI Codex;
- GitHub Copilot;
- ferramentas equivalentes.

Entretanto, o uso dessas ferramentas deverá ocorrer dentro de um processo organizado de Engenharia de Software.

Não é objetivo deste trabalho simplesmente solicitar a um agente:

> "Faça um sistema de reconhecimento facial."

Espera-se que o grupo utilize práticas modernas de desenvolvimento assistido por agentes de IA.

## 4. Spec-Driven Development

O projeto deverá utilizar uma abordagem de Spec-Driven Development.

O fluxo esperado é:

```text
Problema
   ↓
Requisitos
   ↓
Especificação
   ↓
Critérios de aceitação
   ↓
Planejamento
   ↓
Implementação
   ↓
Testes
   ↓
Revisão
```

As especificações deverão ser mantidas no próprio repositório.

Sugestão:

```text
docs/
├── specs/
│   ├── 001-face-detection.md
│   ├── 002-person-enrollment.md
│   ├── 003-face-recognition.md
│   ├── 004-unknown-person.md
│   └── 005-camera-interface.md
│
└── decisions/
    ├── ADR-001-face-model.md
    ├── ADR-002-embedding-storage.md
    └── ADR-003-recognition-threshold.md
```

O objetivo não é produzir documentação excessiva, mas demonstrar que o desenvolvimento foi realizado a partir de requisitos e decisões explícitas.

## 5. Uso de agentes de IA durante o desenvolvimento

Os agentes poderão auxiliar em atividades como:

- planejamento;
- implementação;
- geração de testes;
- refatoração;
- documentação;
- análise de erros;
- revisão de código;
- revisão arquitetural;
- análise dos experimentos.

Entretanto, a responsabilidade pelo código é do grupo.

Todo integrante deverá ser capaz de explicar:

- arquitetura;
- modelos utilizados;
- funções de perda;
- treinamento;
- métricas;
- código principal;
- decisões tomadas;
- resultados obtidos.

Código produzido por IA que não seja compreendido pelos integrantes será considerado um problema de qualidade do projeto.

---

# PARTE 1 — EXPERIMENTAÇÃO

## 6. Objetivo da Parte 1

A primeira etapa deverá responder experimentalmente à seguinte pergunta:

> Qual configuração de modelo, treinamento e/ou fine-tuning apresenta o melhor desempenho para o problema de reconhecimento facial estudado?

O objetivo não é simplesmente baixar um modelo pré-treinado e utilizá-lo.

O grupo deverá realizar uma investigação experimental.

## 7. Pipeline básico

O sistema deverá considerar um pipeline semelhante a:

```text
Imagem
   ↓
Detecção da face
   ↓
Alinhamento / pré-processamento
   ↓
Rede Neural
   ↓
Embedding facial
   ↓
Comparação
   ↓
Reconhecimento
```

Os grupos poderão utilizar modelos como:

- FaceNet;
- ArcFace;
- MobileFaceNet;
- ResNet;
- Vision Transformers;
- outros modelos adequados.

A escolha deverá ser tecnicamente justificada.

## 8. Dados utilizados no projeto

O projeto deverá trabalhar com dois tipos diferentes de dados:

```text
                    DADOS DO PROJETO

            ┌────────────┴────────────┐

      DATASET PÚBLICO           DATASET PRIVADO
                                DO GRUPO

     Experimentação          Fotos coletadas pelos
     e benchmark             próprios alunos
```

Esses dois conjuntos possuem regras de utilização diferentes.

## 9. Dataset público

A Parte 1 deverá utilizar pelo menos um dataset público destinado à pesquisa em reconhecimento facial ou visão computacional.

Exemplos incluem:

### LFW — Labeled Faces in the Wild

Dataset clássico para avaliação de reconhecimento facial em condições não controladas.

Possui imagens com variações de:

- iluminação;
- pose;
- expressão;
- ambiente;
- qualidade de imagem.

### CelebA — CelebFaces Attributes Dataset

Dataset com mais de 200 mil imagens e mais de 10 mil identidades, contendo ainda informações sobre landmarks e atributos faciais.

O CelebA, por exemplo, estabelece condições específicas para utilização acadêmica e não comercial e restringe sua redistribuição.

### Outros datasets

Outros datasets poderão ser utilizados desde que o grupo verifique previamente:

- origem;
- licença;
- finalidade permitida;
- restrições de utilização;
- restrições de redistribuição.

## 10. Dataset público não significa dado sem restrições

Um dataset estar disponível para download na Internet não significa que suas imagens possam ser livremente copiadas ou republicadas.

Antes de utilizar qualquer dataset, o grupo deverá verificar:

```text
Quem disponibilizou?
        ↓
Qual a licença?
        ↓
Permite uso acadêmico?
        ↓
Permite treinamento?
        ↓
Permite redistribuição?
        ↓
Existem restrições?
```

O README deverá informar:

- nome do dataset;
- endereço da fonte oficial;
- finalidade;
- licença ou termos de utilização;
- como o dataset foi utilizado.

Não deverão ser incluídas cópias de datasets no GitHub quando seus termos não permitirem redistribuição.

O repositório deverá fornecer instruções para que o próprio usuário obtenha os dados diretamente da fonte original.

## 11. É proibido criar dataset por scraping indiscriminado

Não será considerado aceitável obter imagens por meio de:

- scraping de Instagram;
- scraping de Facebook;
- scraping de redes sociais;
- busca automática de pessoas na Internet;
- coleta de fotografias sem conhecimento dos titulares.

A disponibilidade pública de uma fotografia na Internet não representa autorização automática para sua utilização como dado biométrico em um sistema de reconhecimento facial.

## 12. Dataset privado coletado pelo grupo

Além dos datasets públicos, cada grupo deverá criar um pequeno dataset próprio, contendo imagens dos integrantes e, opcionalmente, de:

- familiares;
- amigos;
- colegas;
- conhecidos.

As fotografias deverão ser obtidas diretamente pelos alunos e com conhecimento e consentimento das pessoas fotografadas.

O objetivo é avaliar o sistema em condições próximas às encontradas em uma aplicação real.

Recomenda-se obter imagens com variações de:

- iluminação;
- posição da cabeça;
- distância da câmera;
- expressão;
- fundo;
- uso ou não de óculos;
- diferentes momentos de captura.

Deve-se respeitar o princípio da minimização de dados: coletar apenas a quantidade necessária para realização do trabalho.

## 13. LGPD — requisito fundamental do projeto

> **ATENÇÃO**

Este trabalho envolve reconhecimento facial e tratamento de dados biométricos.

Dados biométricos associados a uma pessoa são considerados dados pessoais sensíveis pela Lei Geral de Proteção de Dados Pessoais — LGPD, Lei nº 13.709/2018.

Portanto, privacidade e proteção dos dados fazem parte do trabalho e da avaliação.

Não são aspectos opcionais.

## 14. Consentimento obrigatório

Nenhuma imagem de familiar, amigo, colega ou conhecido poderá ser utilizada sem autorização.

Antes da coleta, a pessoa deverá:

- ser informada sobre a atividade;
- compreender que suas fotografias serão utilizadas em um experimento de reconhecimento facial;
- compreender que poderão ser extraídas representações biométricas, como embeddings;
- compreender a finalidade exclusivamente acadêmica;
- concordar voluntariamente com a participação;
- assinar um Termo de Consentimento.

O consentimento deverá ser:

- informado;
- específico;
- destacado;
- relacionado exclusivamente à finalidade do trabalho.

## 15. Termos de consentimento NÃO devem estar no GitHub

Os documentos de consentimento:

```text
NÃO DEVEM
SER COLOCADOS
NO GITHUB
```

Eles próprios contêm informações pessoais.

Os termos deverão ser mantidos pelos integrantes do grupo em local privado e seguro durante o desenvolvimento do trabalho.

Caso solicitado pelo professor, o grupo deverá ser capaz de comprovar que possui os consentimentos correspondentes às pessoas utilizadas no dataset.

A comprovação deverá ocorrer de forma privada.

## 16. Preferência por participantes maiores de 18 anos

Recomenda-se fortemente que o dataset privado utilize somente pessoas maiores de 18 anos.

Caso excepcionalmente sejam utilizados dados de crianças ou adolescentes, deverão ser observadas todas as exigências legais aplicáveis.

Para crianças, a LGPD estabelece requisitos específicos, incluindo consentimento específico e destacado de pelo menos um dos pais ou responsável legal.

Portanto, para simplificar o projeto e reduzir riscos, a recomendação da disciplina é:

> Utilize participantes adultos.

## 17. O que é considerado dado privado neste trabalho

Deverão ser tratados como dados privados:

- fotografias coletadas pelo grupo;
- vídeos contendo os participantes;
- nomes associados às fotografias;
- identificadores das pessoas;
- embeddings faciais;
- bancos vetoriais;
- arquivos contendo características biométricas;
- arquivos de cadastro;
- resultados individualizados;
- checkpoints treinados ou fine-tunados com esses dados;
- logs contendo informações das pessoas.

## 18. É PROIBIDO publicar dados privados

Os dados coletados dos participantes NUNCA poderão ser enviados para o GitHub público.

Por exemplo:

```text
data/familia/
data/amigos/
data/teste_local/
faces/
embeddings/
database/
```

não deverão ser versionados.

Deverá existir um `.gitignore` apropriado.

Exemplo:

```gitignore
data/private/
faces/
embeddings/
checkpoints/private/
*.db
*.sqlite
consent/
```

O grupo deverá conferir também o histórico do Git.

Adicionar posteriormente um arquivo ao `.gitignore` não remove uma versão que já tenha sido enviada anteriormente ao repositório.

## 19. MODELOS TREINADOS COM DADOS PRIVADOS TAMBÉM NÃO PODEM SER PUBLICADOS

Este é um ponto particularmente importante.

Caso um modelo tenha sido:

- treinado;
- fine-tunado;
- adaptado;
- calibrado de maneira que incorpore informações biométricas dos participantes;

o respectivo checkpoint deverá ser considerado privado.

Portanto:

```text
modelo treinado/fine-tunado
com dados dos participantes
          ↓
        PRIVADO
          ↓
NÃO PUBLICAR NO GITHUB
```

Isso inclui arquivos como:

```text
model.pt
model.pth
model.ckpt
model.onnx
weights.bin
adapter.safetensors
```

quando derivados dos dados privados utilizados no projeto.

Um modelo pré-treinado publicamente disponível poderá ser referenciado por seu endereço oficial.

## 20. Embeddings também são privados

Não será permitido contornar a regra removendo as fotografias e publicando apenas embeddings.

Embeddings utilizados para reconhecer uma pessoa deverão ser tratados neste projeto como informação biométrica privada.

Portanto:

```text
FOTOGRAFIAS → PRIVADAS

EMBEDDINGS → PRIVADOS

BANCO VETORIAL → PRIVADO

CHECKPOINT DERIVADO → PRIVADO
```

## 21. O que pode ir para o GitHub

O repositório deverá conter:

- código-fonte;
- especificações;
- testes;
- configurações;
- documentação;
- scripts de treinamento;
- scripts de avaliação;
- métricas agregadas;
- gráficos agregados;
- instruções para obtenção dos datasets públicos;
- instruções para criação local do dataset privado;
- modelos públicos apenas por referência ou quando sua licença permitir.

Exemplo:

```text
project/
│
├── src/
├── experiments/
├── configs/
├── tests/
├── docs/
│   ├── specs/
│   └── decisions/
│
├── data/
│   └── README.md
│
├── .gitignore
├── README.md
├── requirements.txt
└── pyproject.toml
```

## 22. Segurança dos dados

Os dados privados deverão permanecer:

- no computador do grupo;
- em diretório protegido;
- com acesso restrito aos integrantes responsáveis;
- fora de repositórios públicos;
- fora de serviços públicos de compartilhamento.

O grupo deverá adotar medidas razoáveis para evitar:

- vazamento;
- exposição acidental;
- publicação;
- compartilhamento não autorizado;
- perda dos dados.

## 23. Uso de serviços externos e agentes de IA

Deve-se ter cuidado também ao utilizar:

- Claude;
- ChatGPT;
- Codex;
- Copilot;
- APIs externas;
- serviços de visão computacional;
- notebooks em nuvem;
- outros serviços de terceiros.

Não deverão ser enviadas fotografias ou embeddings privados para serviços externos sem que isso esteja claramente coberto pelo consentimento, pelas regras institucionais e pelas condições legais aplicáveis.

Para este projeto, a opção preferencial é:

> processar localmente os dados privados.

Os agentes de programação deverão receber:

- código;
- especificações;
- estruturas;
- dados sintéticos;

mas não precisam receber as fotografias reais dos participantes.

## 24. Término do projeto e eliminação dos dados

Após a conclusão da disciplina e quando não houver mais necessidade legítima de manutenção dos dados para a finalidade informada, o grupo deverá eliminar:

- fotografias privadas;
- embeddings;
- bancos biométricos;
- checkpoints privados;
- arquivos temporários derivados desses dados.

O participante também poderá solicitar a retirada de seu consentimento nos termos aplicáveis.

O sistema deverá ser organizado para tornar possível excluir os dados correspondentes a uma pessoa.

## 25. Critério ELIMINATÓRIO relacionado à LGPD

> **ESTE ITEM É CRITÉRIO ELIMINATÓRIO DO TRABALHO**

O descumprimento das regras de privacidade, consentimento e proteção dos dados estabelecidas neste enunciado poderá resultar em:

> **NOTA ZERO NO TRABALHO**

São exemplos de situações consideradas graves:

- utilizar fotografias sem consentimento;
- utilizar imagens obtidas secretamente;
- realizar scraping de pessoas para criar o dataset privado;
- publicar fotografias privadas no GitHub;
- publicar embeddings das pessoas;
- publicar banco biométrico;
- publicar checkpoint treinado com dados privados;
- publicar os Termos de Consentimento;
- compartilhar dados dos participantes fora da finalidade autorizada;
- utilizar os dados para finalidade diferente da disciplina;
- manter deliberadamente os dados após solicitação válida de retirada;
- utilizar o sistema para identificar pessoas que não consentiram com a atividade.

Mesmo que o sistema apresente excelente desempenho técnico, o descumprimento dessas regras poderá resultar em nota zero.

## 26. Responsabilidade pelo uso

Ao realizar o trabalho, o aluno declara estar ciente de que:

- reconhecimento facial envolve dados pessoais e potencialmente dados pessoais sensíveis;
- existem obrigações previstas na LGPD e em outras normas aplicáveis;
- os dados coletados somente poderão ser utilizados para a finalidade acadêmica prevista neste trabalho;
- não é autorizado utilizar o software, modelos ou dados para vigilância, monitoramento ou identificação não consentida de terceiros;
- não é autorizado reutilizar os dados para outros projetos sem a correspondente base legal e autorização;
- não é autorizado comercializar, distribuir ou publicar os dados privados.

Qualquer utilização deliberada fora do escopo acadêmico estabelecido neste enunciado constitui iniciativa do próprio aluno e não é autorizada pelo professor nem pela Universidade Federal de Mato Grosso do Sul — UFMS.

O aluno reconhece sua responsabilidade por observar a legislação aplicável e por utilizar corretamente os dados aos quais tiver acesso.

Esta disposição não pretende modificar ou afastar responsabilidades que a legislação eventualmente atribua a cada pessoa ou instituição.

## 27. Experimentos

Cada grupo deverá realizar múltiplos experimentos controlados.

Cada experimento deverá possuir:

- hipótese;
- configuração;
- dataset;
- modelo;
- estratégia de treinamento;
- hiperparâmetros;
- métricas;
- resultado;
- interpretação.

Exemplo:

> Hipótese: O fine-tuning das últimas camadas de um modelo baseado em ArcFace produzirá embeddings mais discriminativos para o domínio estudado do que o uso direto do modelo pré-treinado.

## 28. Estratégias a serem investigadas

Podem ser comparadas, entre outras:

### Modelo pré-treinado sem fine-tuning

```text
Modelo pré-treinado
        ↓
Embeddings
        ↓
Reconhecimento
```

### Fine-tuning parcial

Parte da rede permanece congelada enquanto as últimas camadas são atualizadas.

### Fine-tuning completo

Os parâmetros da rede são atualizados durante o treinamento.

### Estratégias de comparação

Podem ser utilizadas:

- distância Euclidiana;
- similaridade de cosseno;
- k-NN;
- SVM;
- protótipos de classe;
- outros classificadores.

### Funções de perda

Podem ser investigadas:

- Cross-Entropy;
- Contrastive Loss;
- Triplet Loss;
- ArcFace Loss;
- outras técnicas de Metric Learning.

## 29. Separação dos conjuntos

Os experimentos deverão manter separação entre:

```text
TREINAMENTO
     ↓
VALIDAÇÃO
     ↓
TESTE
```

O conjunto de teste não poderá ser utilizado para:

- treinamento;
- escolha de hiperparâmetros;
- seleção de threshold;
- escolha do melhor modelo.

## 30. Cuidado com data leakage

Em reconhecimento facial, a separação dos dados deve ser analisada cuidadosamente.

O grupo deverá verificar se:

- a mesma imagem aparece em treino e teste;
- imagens quase idênticas aparecem em ambos;
- transformações de uma mesma imagem atravessam os conjuntos;
- informações do conjunto de teste influenciaram treinamento ou seleção de parâmetros.

Data leakage invalida a qualidade da avaliação experimental.

## 31. Métricas

Os grupos deverão escolher métricas adequadas.

Exemplos:

- Accuracy;
- Precision;
- Recall;
- F1-score;
- ROC-AUC;
- FAR — False Acceptance Rate;
- FRR — False Rejection Rate;
- Equal Error Rate;
- matriz de confusão.

Também deverão ser consideradas distribuições de similaridade:

```text
MESMA PESSOA
     vs.
PESSOAS DIFERENTES
```

## 32. Threshold de reconhecimento

O sistema não deverá obrigatoriamente classificar toda face como uma das pessoas conhecidas.

Deve existir a possibilidade:

```text
Pessoa reconhecida

ou

Pessoa desconhecida
```

Portanto, o grupo deverá estudar um limiar de decisão.

Exemplo:

```text
similaridade >= threshold
        ↓
 pessoa reconhecida

similaridade < threshold
        ↓
 pessoa desconhecida
```

A escolha desse threshold deverá ser justificada experimentalmente.

## 33. Análise dos resultados

Não será suficiente apresentar somente uma tabela.

O grupo deverá discutir:

- qual modelo foi melhor;
- por que determinado modelo foi melhor;
- quais modificações melhoraram o desempenho;
- quais modificações pioraram o desempenho;
- impacto do fine-tuning;
- impacto do threshold;
- erros encontrados;
- limitações;
- custo computacional;
- relação entre desempenho e complexidade.

## 34. Reprodutibilidade

Os experimentos deverão ser reproduzíveis.

Sempre que possível deverão ser registrados:

- random seed;
- versão das bibliotecas;
- hiperparâmetros;
- arquitetura;
- número de épocas;
- optimizer;
- learning rate;
- batch size;
- transformações;
- hardware utilizado.

## 35. Entrega da Parte 1

A entrega deverá conter:

### GitHub

- código;
- configurações;
- scripts;
- documentação;
- experimentos;
- métricas;
- gráficos;
- instruções de reprodução.

O README deverá conter uma tabela semelhante a:

| Experimento | Modelo | Estratégia | Dataset | Accuracy | F1 | Observação |
|---|---|---|---|---|---|---|
| E01 | Modelo A | baseline | público | ... | ... | ... |
| E02 | Modelo A | fine-tuning | público | ... | ... | ... |
| E03 | Modelo B | fine-tuning | público | ... | ... | ... |

Resultados envolvendo o dataset privado deverão ser apresentados somente de maneira agregada e sem identificação dos participantes.

### Vídeo

O vídeo deverá apresentar:

- problema;
- datasets;
- arquitetura;
- hipóteses;
- experimentos;
- treinamento;
- resultados;
- comparação;
- conclusão;
- escolha do modelo.

---

# PARTE 2 — APLICAÇÃO

## 36. Objetivo

Na segunda etapa, o grupo deverá transformar o melhor resultado da Parte 1 em uma aplicação funcional de reconhecimento facial.

A decisão deverá seguir:

```text
EXPERIMENTOS
     ↓
RESULTADOS
     ↓
MELHOR CONFIGURAÇÃO
     ↓
APLICAÇÃO
```

Caso o grupo utilize uma configuração diferente da melhor encontrada experimentalmente, deverá justificar a decisão.

## 37. Funcionalidade — cadastro

A aplicação deverá permitir cadastrar uma pessoa.

```text
Pessoa
   ↓
Imagem
   ↓
Detecção facial
   ↓
Alinhamento
   ↓
Embedding
   ↓
Cadastro
```

O sistema deverá tratar situações como:

- nenhuma face detectada;
- múltiplas faces;
- imagem inadequada;
- cadastro duplicado.

## 38. Funcionalidade — reconhecimento

A aplicação deverá receber uma imagem ou vídeo:

```text
Imagem / câmera
       ↓
Detecção
       ↓
Embedding
       ↓
Comparação
       ↓
Threshold
       ↓
Reconhecido / desconhecido
```

## 39. Exclusão de uma pessoa

Como exercício de Privacy by Design, a aplicação deverá possuir mecanismo para excluir uma pessoa cadastrada.

A exclusão deverá remover, quando aplicável:

- cadastro;
- embedding;
- fotografias armazenadas;
- outros dados associados à identidade.

## 40. Interface

A interface poderá ser:

- Web;
- Desktop;
- Mobile;
- aplicação de câmera;
- outra abordagem justificada.

Exemplo:

```text
┌────────────────────────────┐
│                            │
│       imagem da câmera     │
│                            │
│   [Pessoa reconhecida]     │
│                            │
└────────────────────────────┘
```

## 41. Engenharia de Software

Serão avaliadas práticas como:

- separação de responsabilidades;
- modularização;
- testes;
- tratamento de erros;
- logs;
- configuração externa;
- documentação;
- ambiente reproduzível;
- versionamento;
- commits compreensíveis;
- especificações;
- critérios de aceitação;
- revisão de código.

## 42. Vídeo da Parte 2 e privacidade

O vídeo deverá demonstrar:

- inicialização;
- cadastro;
- reconhecimento;
- pessoa desconhecida;
- situação de erro;
- breve explicação da arquitetura.

Entretanto, as regras de privacidade continuam válidas para o vídeo.

Não deverão ser publicadas imagens de terceiros participantes de maneira incompatível com o consentimento fornecido.

A alternativa mais simples para uma demonstração pública é utilizar:

- o próprio aluno; ou
- material não privado apropriado para demonstração.

Dados biométricos privados não deverão ser disponibilizados como arquivos junto ao vídeo.

## 43. O GitHub da Parte 2

O repositório deverá conter o software necessário para reproduzir a aplicação, porém sem os dados privados.

Uma pessoa que clonar o projeto deverá inicialmente encontrar algo semelhante a:

```text
Nenhuma pessoa cadastrada.
```

Ela poderá então criar seu próprio banco local.

Esse comportamento é desejável.

## 44. Critérios de avaliação — Parte 1

Serão considerados:

- qualidade da metodologia experimental;
- formulação das hipóteses;
- adequação dos datasets;
- qualidade do protocolo experimental;
- treinamento e fine-tuning;
- escolha das métricas;
- prevenção de data leakage;
- análise dos resultados;
- reprodutibilidade;
- qualidade do código;
- documentação;
- vídeo.

## 45. Critérios de avaliação — Parte 2

Serão considerados:

- funcionamento da aplicação;
- utilização coerente do melhor modelo da Parte 1;
- arquitetura;
- qualidade do código;
- testes;
- tratamento de erros;
- usabilidade;
- documentação;
- uso de Spec-Driven Development;
- utilização responsável dos agentes de IA;
- qualidade da demonstração.

## 46. Critério transversal — Segurança, privacidade e LGPD

Privacidade não será apenas mais um item da pontuação.

É um pré-requisito para validade do trabalho.

```text
Excelente modelo
      +

Excelente aplicação
      +

Violação grave das regras
de privacidade
      =
   NOTA ZERO
```

Os grupos deverão tratar proteção dos dados como parte da Engenharia de Software e da própria construção do sistema de IA.

## 47. Princípio geral

O objetivo final não é somente construir um reconhecedor facial.

O objetivo é experimentar todo o ciclo moderno de desenvolvimento de um sistema de Inteligência Artificial:

```text
Dados
   ↓
Responsabilidade e privacidade
   ↓
Experimentos
   ↓
Treinamento
   ↓
Avaliação
   ↓
Escolha baseada em evidências
   ↓
Especificação
   ↓
Desenvolvimento assistido por IA
   ↓
Testes
   ↓
Aplicação
   ↓
Validação
```

A qualidade de um sistema de Inteligência Artificial envolve não apenas sua acurácia, mas também sua:

- reprodutibilidade;
- segurança;
- confiabilidade;
- explicabilidade das decisões de projeto;
- Engenharia de Software;
- utilização responsável dos dados.

---

# ANEXO A — Modelo de Termo de Consentimento

### TERMO DE CONSENTIMENTO PARA UTILIZAÇÃO DE IMAGENS EM TRABALHO ACADÊMICO DE RECONHECIMENTO FACIAL

Eu, \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_,

declaro que fui informado(a) sobre a realização de um trabalho acadêmico da disciplina de Inteligência Artificial da Universidade Federal de Mato Grosso do Sul — UFMS, envolvendo o desenvolvimento e avaliação de um sistema experimental de reconhecimento facial.

Estou ciente de que poderão ser coletadas fotografias contendo meu rosto e que essas imagens poderão ser processadas por algoritmos de Inteligência Artificial para:

- detecção facial;
- geração de representações numéricas da face, denominadas embeddings;
- treinamento ou fine-tuning, quando aplicável;
- avaliação do sistema;
- demonstração acadêmica do funcionamento da aplicação nos limites aqui informados.

Declaro estar ciente de que:

- a participação é voluntária;
- os dados serão utilizados exclusivamente para a finalidade acadêmica descrita;
- minhas fotografias não deverão ser publicadas em repositórios públicos;
- meus embeddings e outros dados biométricos não deverão ser disponibilizados publicamente;
- modelos treinados especificamente com meus dados não deverão ser publicados;
- os dados deverão ser armazenados de maneira restrita durante a realização do trabalho;
- posso solicitar a interrupção do uso dos meus dados e sua exclusão nos termos aplicáveis;
- a participação neste trabalho não autoriza a utilização dos meus dados para outros projetos ou finalidades.

### Autorização

Declaro que compreendi as informações apresentadas e consinto voluntariamente com a utilização dos meus dados para a finalidade acadêmica descrita neste documento.

**Nome do participante:**

**Assinatura:**

**Data:**

\_\_\_\_ / \_\_\_\_ / \_\_\_\_\_\_

**Nome do aluno responsável:**

**Assinatura:**

### Demonstração em vídeo

Caso minha imagem seja utilizada durante uma demonstração gravada do funcionamento do sistema:

☐ AUTORIZO a utilização da minha imagem exclusivamente para demonstração acadêmica do trabalho, nas condições que me foram informadas.

☐ NÃO AUTORIZO a utilização da minha imagem no vídeo de demonstração.

Esta autorização para demonstração é independente da autorização para utilização das imagens no desenvolvimento e avaliação do sistema.

---

# ANEXO B — Checklist obrigatório antes da entrega

Antes de entregar o trabalho, cada grupo deverá verificar:

- [ ] Todas as pessoas participantes deram consentimento.
- [ ] Os termos de consentimento estão armazenados de forma privada.
- [ ] Nenhum termo de consentimento está no GitHub.
- [ ] Nenhuma fotografia privada está no GitHub.
- [ ] Nenhum embedding privado está no GitHub.
- [ ] Nenhum banco biométrico está no GitHub.
- [ ] Nenhum checkpoint derivado dos dados privados está no GitHub.
- [ ] O `.gitignore` protege os diretórios privados.
- [ ] O histórico do Git foi verificado.
- [ ] As licenças dos datasets públicos foram verificadas.
- [ ] O README informa a origem dos datasets públicos.
- [ ] Os resultados publicados não identificam os participantes.
- [ ] O vídeo respeita as autorizações concedidas.
- [ ] Os dados privados não foram enviados inadvertidamente para serviços externos.
- [ ] Existe uma forma de excluir uma pessoa cadastrada.
- [ ] O grupo conhece as regras de proteção de dados aplicáveis ao projeto.

A entrega do trabalho representa a declaração do grupo de que realizou essas verificações.
