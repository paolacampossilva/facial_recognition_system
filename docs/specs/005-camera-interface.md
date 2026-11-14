# Spec 005: interface web

- **Status:** Em revisão
- **Data:** 2026-11-14
- **Dona:** Paola
- **Issue:** #39
- **Relacionadas:** [001](001-face-detection.md), [002](002-person-enrollment.md), [003](003-face-recognition.md), [004](004-unknown-person.md), [006](006-person-deletion.md), [007](007-logging-config.md), [D15](../decisions/D15-video-participants.md), [D17](../decisions/D17-interface-web-lia.md)

## Problema

A aplicação precisa de uma interface para cadastrar uma pessoa (seção 37), reconhecer uma pessoa a partir de uma imagem da câmera ou de um arquivo (seção 38) e excluir uma pessoa cadastrada (seção 39). A interface só mostra e coleta: a regra de negócio vive nos serviços das specs 001 a 004 e 006, e os erros deles são traduzidos para mensagens em português só na borda.

## Requisitos

- **R1.** A interface é uma página web (HTML, CSS e JS puro, módulos ES) servida pela mesma API FastAPI, na mesma origem (D17). Não depende de CDN, fonte externa nem biblioteca de front-end.
- **R2.** Três telas no menu lateral: **Cadastro**, **Reconhecimento** e **Pessoas cadastradas**.
- **R3.** Entrada de fotos por **webcam** (`getUserMedia`, uma captura por clique) e por **arquivo**. Antes de enviar, a interface mostra a prévia da foto.
- **R4.** A interface só chama a API, e a API só chama os serviços do núcleo. Nenhuma lógica de reconhecimento, threshold ou duplicidade fica no front-end.
- **R5.** Os erros dos serviços são capturados só na borda da API e viram um modelo único `{code, message}`, com código estável e mensagem em inglês. A interface traduz o `code` para a mensagem em português da tabela abaixo.
- **R6.** Nenhuma foto é guardada no servidor (D12): a imagem vive em memória durante a requisição. Os logs não trazem nome nem conteúdo da imagem (spec 007).
- **R7.** Toda tela tem quatro estados desenhados: carregando, vazio, erro e sucesso. O botão de envio fica desabilitado, com indicador de carregamento, durante a chamada e volta ao normal ao terminar, com sucesso ou erro.
- **R8.** Cada tela tem um único botão primário. Vermelho só aparece em erro e em ação destrutiva.
- **R9.** Resultado `unknown`: mostra "Pessoa desconhecida" e a similaridade, sem sugerir nenhum nome (spec 004, R2 e A8). Resultado `recognized`: mostra o nome e a similaridade. Nos dois casos a similaridade aparece em porcentagem, em destaque (ex.: "91,0%"), com o valor original (cosseno) em tamanho menor.
- **R10.** Galeria vazia: Reconhecimento e Pessoas cadastradas mostram "Nenhuma pessoa cadastrada." (seção 43), e não um erro.
- **R11.** Excluir uma pessoa pede confirmação num diálogo próprio (nunca `window.confirm`), com o botão nomeado pela ação ("Excluir <nome>") e a opção de cancelar (spec 006, R8). Descartar uma foto escolhida também pede confirmação, com o botão "Descartar foto" e a opção "Manter foto".
- **R12.** O formulário de cadastro avisa antes de perder o que foi preenchido ("Sair sem salvar?" / "Continuar editando").
- **R13.** Se a câmera estiver indisponível ou a permissão for negada, a interface mostra a mensagem correspondente e mantém o envio de arquivo funcionando.
- **R14.** A aplicação roda só na máquina local: o servidor escuta em `127.0.0.1` por padrão e não tem autenticação. O tamanho máximo do upload vem da configuração (spec 007).

### Aparência

- **R15.** Tema claro e escuro, pelo botão de tema (rótulo mostra o destino, "Modo escuro" ou "Modo claro"). A escolha fica na chave `theme` do `localStorage`, sem piscar no carregamento, e segue `prefers-color-scheme` quando não há escolha.
- **R16.** Além do fundo branco e preto dos dois temas, o acento é **roxo**, definido só pelo token `--accent` do `tokens.css`, com um valor para cada tema e contraste legível. Os valores ficam só no `tokens.css`. Roxo avermelhado não é usado, porque vermelho é reservado para erro e ação destrutiva.
- **R17.** Toda cor, sombra e transição vem de variável do `tokens.css`, nunca de valor solto no CSS ou no JS.

### Segurança e acessibilidade

- **R18.** A página é montada só com nós DOM e `textContent` (helper `h(tag, attrs, ...filhos)`), nunca com `innerHTML`. Sem script nem estilo embutidos no HTML.
- **R19.** O servidor envia Content-Security-Policy restritiva: `default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' blob: data:; media-src 'self' blob:; connect-src 'self'` (`blob:` e `data:` são para a prévia e a captura da câmera).
- **R20.** `<html lang="pt-br">`, `Esc` fecha diálogos, foco visível, `aria-label` em botão só com ícone e mensagens em português.

### Casca e conteúdo

- **R21.** Casca no padrão do LIA: sidebar de 272px à esquerda (marca, menu com ícones, botão de tema e nota de privacidade no rodapé) e conteúdo à direita. Abaixo de 860px a sidebar vira off-canvas: um botão de menu com `aria-expanded` abre, e `Esc`, o clique no fundo e a navegação fecham, devolvendo o foco ao botão.
- **R22.** Cada tela começa com um título (`h1`) e uma linha de descrição, e o `document.title` é "<Tela> · Reconhecimento facial".
- **R23.** O item "Pessoas cadastradas" do menu mostra quantas pessoas há e atualiza depois de um cadastro e de uma exclusão.
- **R24.** O conteúdo fica em cards, em duas colunas que passam a uma só quando a área de conteúdo fica estreita.
- **R25.** Em Reconhecimento, o resultado aparece num painel ao lado da foto, com um texto de espera antes de reconhecer. Sem ninguém cadastrado, Reconhecimento e Pessoas cadastradas mostram um estado vazio com um botão que leva ao Cadastro. O Cadastro avisa que as fotos não são guardadas, só os dados faciais (D12).
- **R26.** Os erros de preenchimento do cadastro aparecem junto do campo que os tem, cada um com a sua mensagem (nome e fotos), com o campo marcado por `aria-invalid` e ligado à mensagem por `aria-describedby`. O foco vai para o primeiro campo com erro, e o erro some quando o campo é corrigido. Erros que vêm da API continuam numa mensagem acima do botão.
- **R27.** Com a câmera aberta, "Capturar foto" (estilo neutro) e "Fechar a câmera" (vermelho) ficam logo abaixo da imagem. A prévia da câmera é espelhada, como num espelho, e a foto capturada não é. A câmera é desligada ao sair da tela e quando o cadastro é concluído, e o aviso de limite de fotos some junto com as fotos.
- **R28.** A página tem favicon, para o navegador não registrar um erro 404 no console.
- **R29.** O Cadastro mostra dicas para boas fotos no cartão de fotos (uma só pessoa, de frente, boa luz sem contraluz, sem óculos escuros, boné ou máscara, variar ângulo e expressão, foto nítida), em uma caixa com destaque roxo e título roxo, e o Reconhecimento lembra que a foto deve ter um só rosto, de frente e bem iluminada.
- **R30.** O resultado do reconhecimento marca na barra de similaridade o threshold lido de `GET /api/config`: a barra fica verde quando a pessoa é reconhecida e neutra quando é desconhecida (sem vermelho nem amarelo), e o texto "Acima do limite (0,766)" ou "Abaixo do limite (0,766)" repete a informação, para a cor não ser o único sinal. Quem decide reconhecido ou desconhecido é a API (R4): o front-end só mostra o limite. Se `/api/config` falhar, a barra aparece sem a marca e sem o texto.

## Contrato HTTP

JSON, mesma origem. Modelos de entrada e saída são Pydantic, os mesmos dos serviços.

| Método e rota | Entrada | Saída (sucesso) |
|---|---|---|
| `GET /api/health` | | `{status: "ok"}` |
| `GET /api/config` | | `{threshold}`, o limite de similaridade da aplicação (D08) |
| `GET /api/persons` | | lista de `{id, name, embedding_count}` |
| `POST /api/persons` | multipart: `name` e 1 a 10 `photos` | 201, `EnrollmentResult` (spec 002) |
| `POST /api/recognitions` | multipart: 1 `photo` | `RecognitionResult` (spec 003) |
| `DELETE /api/persons/{id}` | | `DeletionResult` (spec 006) |

Erro: `{code, message}`, com o status HTTP da tabela. Só o `duplicate_person` traz também `reason` (`name` ou `similarity`) e `person_name` (a pessoa que já está cadastrada), para a interface escolher a mensagem.

| Erro do serviço | `code` | HTTP | Mensagem para a pessoa |
|---|---|---|---|
| `NoFaceDetectedError` | `no_face_detected` | 422 | "Nenhum rosto encontrado na imagem" |
| `MultipleFacesError` | `multiple_faces` | 422 | "Mais de um rosto na imagem" |
| `InvalidImageError` | `invalid_image` | 422 | "Arquivo de imagem inválido" |
| `DuplicatePersonError` (`reason="name"`) | `duplicate_person` | 409 | "Já existe uma pessoa cadastrada com esse nome" |
| `DuplicatePersonError` (`reason="similarity"`) | `duplicate_person` | 409 | "Esse rosto já está cadastrado como <nome>" |
| `EmptyGalleryError` | `empty_gallery` | 409 | "Nenhuma pessoa cadastrada." |
| `PersonNotFoundError` | `person_not_found` | 404 | "Pessoa não encontrada. A lista foi atualizada" |
| validação Pydantic (nome, número de fotos, tamanho) | `invalid_request` | 422 | "Confira o nome e as fotos (de 1 a 10)" |
| qualquer outro | `internal_error` | 500 | "Algo deu errado. Tente de novo" |
| câmera indisponível ou negada (só no navegador) | | | "Não foi possível usar a câmera. Envie um arquivo" |
| nome vazio (só no navegador, junto ao campo do nome) | | | "Digite o nome da pessoa." |
| nenhuma foto (só no navegador, junto às fotos) | | | "Adicione de 1 a 10 fotos." |

Quando o cadastro tem várias fotos, o erro não diz qual delas falhou (spec 002, R2): a mensagem pede para conferir as fotos.

## Critérios de aceitação

| # | Dado | Quando | Então |
|---|---|---|---|
| A1 | nome e 1 a 10 fotos válidas | cadastra | 201 com `EnrollmentResult`, e a interface mostra o sucesso e a pessoa na lista |
| A2 | uma foto sem rosto, com vários rostos ou que não é imagem | cadastra | a API devolve 422 com `no_face_detected`, `multiple_faces` ou `invalid_image`, e a interface mostra a mensagem em português da tabela |
| A3 | nome já usado, com maiúsculas diferentes | cadastra | 409 `duplicate_person`, e a interface mostra a mensagem de nome repetido |
| A4 | rosto parecido com o de uma pessoa cadastrada | cadastra com outro nome | 409 `duplicate_person`, e a mensagem inclui o nome da pessoa existente |
| A5 | nome vazio, só espaços ou fotos fora de 1 a 10 | cadastra | 422 `invalid_request`, sem tocar no banco |
| A6 | uma pessoa cadastrada e uma foto dela | reconhece | a interface mostra o nome e a similaridade |
| A7 | uma pessoa cadastrada e o rosto de outra pessoa | reconhece | a interface mostra "Pessoa desconhecida" e a similaridade, sem nome (spec 004, A8) |
| A8 | banco sem ninguém | abre Reconhecimento e Pessoas cadastradas | mostram "Nenhuma pessoa cadastrada." como estado vazio, sem erro |
| A9 | uma pessoa cadastrada | clica em excluir | abre o diálogo com o botão "Excluir <nome>" e a opção de cancelar |
| A10 | o diálogo de exclusão aberto | cancela ou aperta `Esc` | nada é excluído |
| A11 | o diálogo de exclusão aberto | confirma | `DELETE` é chamado, a pessoa some da lista e o reconhecimento seguinte não a encontra |
| A12 | um `id` que não existe mais | exclui | 404 `person_not_found`, e a interface atualiza a lista |
| A13 | a permissão da câmera negada | abre a captura | mostra a mensagem de câmera indisponível, e o envio de arquivo continua funcionando |
| A14 | qualquer chamada em andamento | a requisição está pendente | o botão de envio está desabilitado com indicador, e volta ao normal no sucesso e no erro |
| A15 | formulário de cadastro preenchido | tenta trocar de aba ou fechar | aparece "Sair sem salvar?" com "Continuar editando" |
| A16 | o tema escolhido | recarrega a página | o tema se mantém, sem piscar, e a chave `theme` é a única gravada no `localStorage` |
| A17 | os dois temas | o `--accent` é inspecionado | é roxo em ambos, e o texto sobre ele tem contraste legível |
| A18 | qualquer resposta do servidor estático | os cabeçalhos são inspecionados | trazem a CSP da R19 |
| A19 | o código do front-end | é varrido nos testes | não contém `innerHTML`, `window.confirm`, estilo ou script embutido nem URL externa |
| A20 | um erro inesperado nos serviços | qualquer rota | 500 `internal_error`, sem vazar detalhe interno nem dado pessoal |
| A21 | serviços fakes injetados | a API é testada | os testes rodam sem rede, sem GPU e sem pesos |
| A22 | a janela com 860px ou mais | a página abre | a sidebar está visível com os três itens e o contador de pessoas |
| A23 | a janela com menos de 860px | abre o menu, e depois aperta `Esc` | a sidebar abre com `aria-expanded="true"`, fecha com `Esc`, e o foco volta ao botão de menu |
| A24 | o menu | troca de tela | o `document.title` muda para "<Tela> · Reconhecimento facial" |
| A25 | banco sem ninguém | abre Reconhecimento e Pessoas cadastradas, e depois cadastra e exclui uma pessoa | as duas telas mostram o estado vazio com o botão para o Cadastro, e o contador sobe ao cadastrar e desce ao excluir |
| A26 | nome vazio e nenhuma foto | clica em Cadastrar | "Digite o nome da pessoa." aparece junto ao nome e "Adicione de 1 a 10 fotos." junto às fotos, cada campo com `aria-invalid`, o foco vai ao nome, e cada mensagem some quando o campo é corrigido |
| A27 | uma foto escolhida | clica em descartar | abre o diálogo com "Descartar foto" e "Manter foto"; `Esc` ou "Manter foto" mantém a foto, e "Descartar foto" a remove da lista |
| A28 | a câmera aberta | olha os botões e troca de tela | "Capturar foto" é neutro, "Fechar a câmera" é vermelho, e a câmera é desligada ao sair |
| A29 | a tela de Cadastro aberta | olha o cartão de fotos | vê a lista "Dicas para boas fotos" aberta, com as cinco dicas |
| A30 | a câmera aberta, 10 fotos tentadas e um nome válido | conclui o cadastro | a câmera é desligada, a lista de fotos fica vazia e o aviso "O limite é de 10 fotos." some |
| A31 | uma pessoa cadastrada | reconhece uma foto dela e depois a de outra pessoa | a barra mostra a marca do limite, verde com "Acima do limite" no primeiro caso e neutra com "Abaixo do limite" no segundo |

## Fora de escopo

- Autenticação, vários usuários e acesso pela rede.
- Vídeo contínuo quadro a quadro (a captura é uma foto por clique, D17).
- Editar uma pessoa cadastrada (exclui e cadastra de novo, spec 002).
- Outros idiomas além do português.

## Testes

- **Unitários da API** (`TestClient` com serviços fakes, A1 a A5, A8, A12, A18, A20 e A21, mais o `GET /api/config` e a ausência de logs nas rotas): contrato, mapeamento de cada erro para `code` e status, e cabeçalhos de segurança.
- **Teste do código do front-end** (A19, e A22 e A24 no que dá para checar nos arquivos, e a R30 pelo `meter.js`, que roda no `node` sem DOM, inclusive sem o threshold): varredura de `src/app/web/` no estilo de um teste de contrato, conferindo também a sidebar, o ponto de corte de 860px e o título por tela.
- **E2E** (#50, da Valentina): Chrome sem interface cobrindo A1, A6, A7, A9 a A11, A13 a A17, A22 a A31, e falhando se a página registrar erro de console.
