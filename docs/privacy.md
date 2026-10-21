# Privacidade e LGPD

Este documento descreve como o projeto trata os dados pessoais das participantes do dataset privado: que dados existem, por que são coletados, onde ficam, quem acessa, como uma pessoa é excluída e como tudo é eliminado no fim. As regras vêm das seções 12 a 25 e 46 e do Anexo B do [enunciado](assignment.md). As decisões técnicas estão no [division.md](division.md) e nos [ADRs](decisions/README.md).

Fotos de rosto, e tudo o que se extrai delas para reconhecer uma pessoa, são **dados pessoais sensíveis** (dado biométrico, LGPD, Lei nº 13.709/2018, art. 5º, II). Descumprir estas regras é critério eliminatório do trabalho (seção 25).

## Finalidade e base legal

- **Finalidade única:** experimentos acadêmicos de reconhecimento facial da disciplina de Inteligência Artificial da UFMS (Parte 1) e a demonstração da aplicação (Parte 2). Os dados não são usados para nenhum outro fim, não são compartilhados fora da dupla e não servem para identificar quem não consentiu (seções 25 e 26).
- **Base legal:** consentimento específico e destacado da titular (LGPD, art. 11, I), colhido pelo Termo de Consentimento adaptado do Anexo A antes de qualquer foto ser usada (seção 14). Sem termo assinado, a foto não entra no dataset.
- **Só adultos** (seção 16). Nenhuma participante é criança ou adolescente.
- **Revogação:** a participante pode retirar o consentimento a qualquer momento (LGPD, art. 8º, § 5º), e os dados dela são excluídos como descrito em "Exclusão de uma participante".

## Inventário dos dados

| Dado | Onde fica | Quem acessa | No GitHub? | Quando é eliminado |
|---|---|---|---|---|
| Fotos originais | cofre, `V:\private\<ID>\session{1,2}\` (junção `data/private/`) | Valentina | **nunca** | fim da disciplina ou pedido da participante |
| Recortes do MTCNN | cofre, `V:\faces\` (junção `faces/private/`) | Valentina | **nunca** | idem |
| Embeddings do E07 e do E08 | cofre, `V:\embeddings\` (junção `embeddings/private/`) | Valentina | **nunca** | idem |
| Classificadores ajustados com dados privados (k-NN, SVM, protótipos do E07) | cofre, `V:\checkpoints\` (junção `checkpoints/private/`) | Valentina | **nunca** | idem |
| Resultados por pessoa | cofre | Valentina | **nunca** | idem |
| Termos assinados e planilha de controle | cofre, `V:\termos\` (sem junção, fora do projeto) | Valentina e Paola (cada uma os das próprias convidadas) | **nunca** | fim da disciplina |
| Banco da aplicação (Parte 2) | `data/private/app.db`, no cofre | Valentina | **nunca** | ao excluir a pessoa na aplicação, e no fim da disciplina |
| Métricas e gráficos agregados, com IDs anônimos | `experiments/` | público | sim | permanecem |

O que **não** é dado privado: o LFW (dataset público, D1), os splits e pares do LFW e os modelos treinados **só** com o LFW (E04 e E06). Mesmo assim, os checkpoints públicos ficam fora do Git pelo tamanho e por regra geral (`*.pt`). Os pesos pré-treinados do FaceNet e da ResNet-50 são baixados da fonte oficial, por referência (seção 19).

## Coleta e minimização

- Cerca de 5 a 10 participantes, 20 a 30 fotos por pessoa, em duas sessões em cômodos ou com fundos diferentes. É o mínimo para medir reconhecimento com sessões distintas (D9).
- Cada participante recebe um **ID anônimo** a partir de `P00`: pares (`P00`, `P02`, ...) para as convidadas da Valentina e ímpares (`P01`, `P03`, ...) para as da Paola, para as duas numerarem em paralelo sem repetir. O nome aparece **só** no termo e na planilha. Pastas, arquivos, logs e resultados usam o ID.
- Quem tira as fotos é a própria participante, a partir do [guia de fotos](participant_photo_guide.pdf). Nada vem de scraping, de redes sociais ou de fotos antigas (seção 11).
- A aplicação da Parte 2 **não guarda fotos**: só o embedding de cada cadastro (D12).

## Envio das fotos por WhatsApp

O mensageiro é um serviço de terceiros, então o envio só acontece com a **cláusula específica no termo** que o autoriza (seções 14 e 23). Quem não autorizou entrega as fotos por pendrive ou as tira com uma de nós.

- **Canal:** só a conversa individual com a integrante que fez o convite, com criptografia de ponta a ponta. Nunca grupos, e-mail, nuvem ou link de compartilhamento.
- **Formato:** as fotos vão como foto, não como documento, para o aplicativo remover metadados como a localização. A mensagem diz só "Etapa 1" ou "Etapa 2".
- **Antes de receber:** download automático de mídia, salvamento na galeria e backup de conversas em nuvem desligados no celular de quem recebe.
- **Ao receber:** as fotos são salvas direto no cofre do notebook da RTX 4060, pelo WhatsApp Web aberto no notebook ou por cabo USB. Nunca passam pelo Mac da Paola. Fotos recebidas no celular da Paola chegam ao notebook por cabo ou por pendrive dentro de um contêiner VeraCrypt, nunca reencaminhadas por mensageiro.
- **Em até 24 horas:** depois de conferir que as fotos estão no cofre, a integrante apaga as mensagens com "Apagar para todos", limpa a conversa, remove as mídias do celular e esvazia a lixeira da galeria. A participante é avisada de que pode apagar a cópia dela.
- **Registro:** a planilha de controle anota, por ID, a data de recebimento e a data de exclusão da conversa.

### Termos da Paola para o cofre

Todos os termos ficam no cofre da 4060, em `V:\termos`. A Paola manda os termos das convidadas dela à Valentina como documento PDF, na conversa individual do WhatsApp entre as duas, com o backup de conversas desligado nos dois celulares. A Valentina salva cada arquivo direto em `V:\termos` pelo WhatsApp Web no notebook e confere se abriu. Em até 24 horas, as duas apagam as mensagens com "Apagar para todos", limpam a conversa e removem os arquivos dos celulares, e as datas vão na planilha. **Só termos** passam por essa conversa: fotos recebidas pela Paola nunca são reencaminhadas. Ao fim da coleta, a cópia da planilha da Paola também vai para o cofre.

## Armazenamento e acesso

- **Uma máquina só** (D10): o notebook da Valentina com a RTX 4060. Todo processamento de dados privados (MTCNN, embeddings, E07, E08 e a aplicação com as participantes) acontece nele. O Mac da Paola só usa dados públicos ou sintéticos.
- **Cofre VeraCrypt:** um contêiner criptografado nesse notebook, fora de pastas sincronizadas com nuvem, montado como `V:` só durante o uso e desmontado em seguida. Só a Valentina tem a senha.
- **Junções do Windows:** `data/private/`, `faces/private/`, `embeddings/private/` e `checkpoints/private/` apontam para pastas do cofre. Com o cofre desmontado, elas ficam vazias e nenhum script consegue gravar dado privado fora dele.
- **Limpeza de cópias:** depois de salvar algo no cofre, apagar as cópias em Downloads, Área de Trabalho e anexos, e esvaziar as lixeiras.

## Proteções no repositório

1. **`.gitignore`** ignora `data/private/`, `faces/`, `embeddings/`, `checkpoints/`, `consent/`, bancos (`*.db`, `*.sqlite`), pesos (`*.pt`, `*.pth`, `*.onnx`, `*.safetensors`) e arrays ou classificadores salvos (`*.npy`, `*.npz`, `*.pkl`, `*.joblib`).
2. **Hook de pre-commit** ([scripts/check_private_files.py](../scripts/check_private_files.py)), instalado em cada clone com `pre-commit install`: bloqueia fotos, vídeos, pesos, embeddings e bancos em qualquer pasta, mesmo com `git add -f`, e só aceita PDF, Word e Excel em `docs/`, para termos e planilhas não entrarem por engano. `git commit --no-verify` é proibido.
3. **Auditoria do histórico** antes de cada entrega: `git log --all --stat` procurando imagens, `.db` e `.pt` (seção 18, Anexo B). Adicionar algo ao `.gitignore` depois não apaga o que já foi enviado.

## Agentes de IA e serviços externos

Seguindo a seção 23, **fotos, recortes, embeddings privados, termos e a planilha nunca são enviados** a agentes de IA (Claude, ChatGPT, Copilot e afins), APIs externas, notebooks em nuvem ou serviços de visão computacional. Os agentes recebem só código, especificações, estruturas e dados públicos ou sintéticos. Quando um agente precisa verificar o dataset privado, ele roda comandos que devolvem apenas contagens agregadas (por exemplo, quantos arquivos há em cada `P??/session?`), sem nomes de arquivo, conteúdo ou imagens.

## Resultados, vídeos e aplicação

- **Resultados publicados só agregados:** accuracy, F1, FAR, FRR e matriz de confusão com IDs anônimos, sem foto, nome ou resultado que permita identificar alguém (seção 35 e D7). O código de avaliação rejeita rótulos que não sejam IDs anônimos.
- **Threshold do E08:** é um único número escolhido na validação privada e publicado como parte da configuração. Por ser um agregado de todas as tentativas, ele não carrega a biometria de nenhuma participante, ao contrário de um classificador ajustado, que fica no cofre.
- **Vídeos:** só Paola e Valentina aparecem (D15). Nenhuma participante aparece em vídeo, e nenhum resultado individual é mostrado.
- **Aplicação (Parte 2):** guarda só embeddings, em SQLite, com exclusão em cascata (D12). Os logs não registram nomes, fotos ou embeddings (spec 007).

## Exclusão de uma participante

Se uma participante pedir para sair (LGPD, art. 18, VI e IX), em até 7 dias:

1. Apagar `V:\private\<ID>\` e `V:\faces\<ID>\`.
2. Apagar os embeddings e os classificadores privados que a incluem (`V:\embeddings\` e `V:\checkpoints\`) e, se ainda houver experimento a rodar, gerá-los de novo sem ela.
3. Na aplicação, excluir a pessoa pela tela de pessoas cadastradas, o que apaga o registro e os embeddings em cascata (spec 006).
4. Anotar na planilha a data do pedido e a da exclusão. O termo e a planilha seguem no cofre, como registro do consentimento e da retirada, até a eliminação final.
5. Avisar a participante de que os dados foram excluídos.

Resultados agregados já publicados (por exemplo, a accuracy do E07) não identificam ninguém e permanecem. Se a exclusão acontecer antes da entrega, os experimentos privados são refeitos sem ela.

## Eliminação no fim da disciplina

Depois da nota final (issue #54), quando não houver mais necessidade para a finalidade informada (seção 24, LGPD art. 16):

- apagar todo o conteúdo do cofre: fotos, recortes, embeddings, classificadores, banco da aplicação, termos e planilhas;
- apagar o arquivo do contêiner VeraCrypt;
- conferir que não sobrou cópia em Downloads, Área de Trabalho, celulares, pendrives ou lixeiras;
- avisar as participantes de que os dados foram eliminados.

## Se algo vazar

Se um dado privado for parar onde não deveria (um commit, uma conversa de grupo, um serviço externo):

1. Remover na hora do lugar exposto e, se foi para o Git, reescrever o histórico e forçar o push, conferindo depois com `git log --all --stat`.
2. Avisar a outra integrante e registrar o que aconteceu, quando e o que foi exposto.
3. Avisar as participantes afetadas.

## Checklist do Anexo B

| Item | Onde é garantido |
|---|---|
| Todas deram consentimento | termos em `V:\termos`, conferidos contra a planilha antes de cada sessão |
| Termos armazenados de forma privada e fora do GitHub | cofre, hook de pre-commit (PDF, Word e Excel só em `docs/`) |
| Nenhuma foto, embedding, banco ou checkpoint privado no GitHub | `.gitignore`, hook, junções para o cofre |
| `.gitignore` protege os diretórios privados | [.gitignore](../.gitignore) |
| Histórico do Git verificado | auditoria antes de cada entrega (#34, #53) |
| Licenças e origem dos datasets públicos | [data/README.md](../data/README.md) e D1 |
| Resultados não identificam participantes | só agregados, IDs anônimos (D7) |
| Vídeo respeita as autorizações | só as duas integrantes aparecem (D15) |
| Nada enviado a serviços externos | "Agentes de IA e serviços externos" acima |
| É possível excluir uma pessoa | "Exclusão de uma participante" acima e a spec 006 |
| O grupo conhece as regras | este documento e as seções 12 a 25 do enunciado |
