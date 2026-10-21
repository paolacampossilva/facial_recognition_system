# D13: interface

- **Status:** Aceita
- **Data:** 2026-10-18
- **Dona:** Valentina

## Contexto

A Parte 2 precisa de uma aplicação que cadastre uma pessoa (seção 37), reconheça a partir de "uma imagem ou vídeo" dizendo "reconhecido" ou "desconhecido" (seção 38) e exclua uma pessoa cadastrada (seção 39). A interface pode ser web, desktop, mobile, de câmera ou outra justificada (seção 40). A arquitetura precisa separar a interface da lógica, com serviços testáveis (seção 41). O vídeo da Parte 2 mostra inicialização, cadastro, reconhecimento, uma pessoa desconhecida e um erro (seção 42). A aplicação roda no notebook da RTX 4060, onde estão o modelo e o banco (D10 e D12), e a dupla tem cerca de três semanas para fazê-la, junto com o resto do trabalho.

## Decisão

**Streamlit, uma tela com três abas:**

- **Cadastro:** nome mais uma ou mais fotos, chamando o `EnrollmentService`. Mostra as mensagens dos erros da seção 37: nenhum rosto, vários rostos, imagem inadequada e cadastro duplicado.
- **Reconhecimento:** uma foto, chamando o `RecognitionService`. Mostra a pessoa reconhecida e a similaridade, ou "pessoa desconhecida" abaixo do threshold.
- **Pessoas cadastradas:** lista os cadastros e exclui uma pessoa com confirmação, chamando o `DeletionService`.

A entrada das fotos é por **`st.camera_input`** (uma foto da webcam por clique) e por **`st.file_uploader`** (arquivo do computador). As duas juntas cobrem o "imagem ou vídeo" da seção 38: a câmera é a entrada ao vivo, e o arquivo permite testar com imagens prontas e reproduzir os casos de erro no vídeo.

A interface só chama os serviços do núcleo da aplicação (da Paola) e trata os erros na borda, com mensagens amigáveis em português (spec 005). Ela não contém lógica de reconhecimento, então os serviços são testados sem a interface, com detector e modelo falsos.

## Alternativas consideradas

- **Vídeo contínuo da webcam com `streamlit-webrtc` ou OpenCV:** mais próximo de uma "aplicação de câmera", mas exige reconhecer quadro a quadro (o MTCNN e o FaceNet a cada quadro), controlar a taxa de quadros e tratar o WebRTC no navegador. Não muda o que é avaliado (cadastro, reconhecimento, desconhecido e exclusão) e custaria dias que não temos. Fica fora do escopo.
- **Interface desktop (Tkinter, PyQt):** mais código de interface para o mesmo resultado, e a câmera e os uploads dão mais trabalho que no Streamlit.
- **API (FastAPI) com front-end web separado:** é a arquitetura mais "de produção", mas são dois projetos para manter e testar. A separação em camadas que a seção 41 pede já existe entre a interface e os serviços.
- **Gradio:** parecido com o Streamlit em simplicidade, mas é voltado a demonstrar um modelo de uma entrada e uma saída. As três abas com estado (lista de pessoas, exclusão) se organizam melhor no Streamlit.

## Consequências

- Executar é `streamlit run` num comando só, e o navegador abre na própria máquina. Nada sai do notebook: a câmera e os uploads são processados localmente, e as fotos não são guardadas, só os embeddings (D12).
- O `st.camera_input` tira uma foto por vez. O vídeo da Parte 2 mostra o reconhecimento foto a foto, não em tempo real, e isso fica registrado como limitação no README da Parte 2.
- O Streamlit roda o script de novo a cada interação, então o modelo é carregado uma vez e reaproveitado (`st.cache_resource`), em vez de a cada clique.
- A spec 005 (`camera-interface`, #39) detalha os fluxos e as mensagens. O esqueleto da interface com serviços falsos (#44) começa antes do núcleo estar pronto.
