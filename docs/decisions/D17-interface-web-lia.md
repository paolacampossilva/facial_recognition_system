# D17: interface web (FastAPI e JS puro)

- **Status:** Aceita
- **Data:** 2026-11-14
- **Dona:** Paola

## Contexto

A Parte 2 precisa de cadastro (seção 37), reconhecimento com "reconhecido" ou "desconhecido" (seção 38) e exclusão de uma pessoa (seção 39). A interface é livre (seção 40) e a arquitetura precisa separar interface e lógica (seção 41). A [D13](D13-interface.md) escolheu Streamlit, mas a Paola passou a fazer a interface inteira e já trabalha com um padrão próprio de front-end (SPA em HTML, CSS e JS puro servida pelo FastAPI, com tema claro e escuro, diálogos e estados de tela padronizados). Usar o mesmo padrão reduz o risco e a curva de aprendizado em uma semana apertada.

## Decisão

**FastAPI serve uma SPA em HTML, CSS e JS puro (módulos ES), com três telas no menu lateral (casca do LIA):**

- **Cadastro:** nome e fotos, chamando o `EnrollmentService`. Mostra os erros da seção 37: nenhum rosto, vários rostos, imagem inadequada e cadastro duplicado.
- **Reconhecimento:** uma foto, chamando o `RecognitionService`. Mostra a pessoa e a similaridade, ou "pessoa desconhecida" abaixo do threshold.
- **Pessoas cadastradas:** lista e exclui, com confirmação, chamando o `DeletionService`.

Entrada das fotos: webcam pelo navegador (`getUserMedia`, uma captura por clique) e envio de arquivo.

Padrões da interface, em `src/app/web/`:

- `tokens.css` único, com tema claro e escuro (`[data-theme]` e `prefers-color-scheme`), acento roxo (`--accent`) e botão de tema.
- Páginas montadas só com nós DOM e `textContent` (helper `h(tag, attrs, ...filhos)`), nunca `innerHTML`.
- Confirmação por diálogo próprio, nunca `window.confirm`.
- Toda tela com quatro estados: carregando, vazio, erro e sucesso. Botão de envio desabilitado com indicador durante a chamada. Um só botão primário por tela.
- Mensagens em português, `Esc` fecha diálogos, `aria-label` em botão só com ícone.
- Cabeçalhos de segurança no servidor estático, com CSP (`default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'`).
- Configuração pelo `Settings` (pydantic-settings), erros tratados só na borda da API com código estável.

A interface só chama a API, e a API só chama os serviços do núcleo. Não há lógica de reconhecimento no front-end.

## Alternativas consideradas

- **Streamlit (D13):** mais rápido para as abas, mas não segue o padrão de interface que a Paola já domina e o tema e os estados teriam de ser reescritos à mão.
- **Next.js:** mais um projeto Node para manter, sem ganho para o que é avaliado.
- **Gradio:** voltado a demonstrar um modelo com uma entrada e uma saída, e as abas com estado ficam piores.
- **Desktop (Tkinter, PyQt):** mais código de interface para o mesmo resultado.

## Consequências

- `requirements.txt` ganha `fastapi`, `uvicorn` e `python-multipart`, com versões fixadas. O teste e2e precisa de Chrome e Node.
- Tudo roda local: a câmera e os uploads são processados no notebook, e só os embeddings são guardados (D12).
- O reconhecimento é foto a foto, não quadro a quadro. Vídeo contínuo fica fora do escopo e entra como limitação no README.
- O modelo é carregado uma vez na subida da API, não a cada requisição.
- A spec 005 (`camera-interface`) passa a ser da Paola e descreve esta interface.
