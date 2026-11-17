# D18: a API só atende a própria página

- **Status:** Aceita
- **Data:** 2026-11-17
- **Dona:** Paola

## Contexto

A aplicação guarda dados biométricos (embeddings e nomes) e não tem login (spec 005, R14). Ela escuta só em `127.0.0.1`, mas o navegador de quem a usa também abre outros sites, e esses sites conseguem mandar requisições para `127.0.0.1`: um formulário de outra página pode enviar um POST de cadastro (um "POST simples" não passa por pré-verificação do navegador), e uma página que usa DNS rebinding passa a ser tratada como a mesma origem da aplicação e consegue ler a lista de pessoas. Escutar só no endereço local não protege contra isso.

## Decisão

Quatro regras na borda da API (`src/app/api/guard.py`), todas sem configuração, porque não há motivo para o usuário desligá-las:

- **Host.** Só `127.0.0.1` e `localhost` (com qualquer porta, já que `--port` escolhe a porta). Outro valor devolve 400. Isso fecha o DNS rebinding.
- **Origin.** Um POST ou DELETE que traz `Origin` só passa se ele for a própria página (`http://<Host>`). Outro valor, inclusive `null`, devolve 403 com o código `forbidden_origin`. O navegador sempre manda `Origin` num POST entre sites, então uma página de fora não consegue omiti-lo. Uma requisição sem `Origin` (um script, o `curl`, os testes) passa, porque não é um navegador agindo em nome de uma página.
- **Cache.** Toda resposta de `/api` leva `Cache-Control: no-store` e `X-Content-Type-Options: nosniff`, para nomes e resultados não ficarem no cache do navegador.
- **Tamanho do corpo.** Um corpo declarado maior que as fotos que a rota aceita (10 no cadastro e 1 no reconhecimento, no tamanho máximo, mais uma margem de 64 KiB) é recusado com o mesmo 422 `invalid_request` do upload grande demais, antes de o Starlette ler e guardar tudo. Mais de 10 fotos no cadastro também é recusado antes de ler qualquer uma.

## Alternativas consideradas

- **Token anti-CSRF (cookie ou cabeçalho):** exigiria estado ou segredo na página, mais código no front-end e nos testes, para um problema que a checagem de `Origin` resolve sem estado.
- **CORS restritivo:** o CORS só impede que a página de fora leia a resposta; o POST simples chega ao servidor e cadastra mesmo assim, então não resolve o cadastro indevido.
- **Autenticação:** é o que protegeria de verdade uma aplicação exposta, mas está fora de escopo (uso local, um computador, spec 005, R14) e a aplicação continua sem opção de abrir para a rede.
- **Configuração dos hosts permitidos:** serviria para trás de um proxy, que não é um uso suportado. Fica fixo, como o endereço `127.0.0.1` (spec 007, R8).

## Consequências

- Acessar a aplicação por outro nome (o IP da máquina na rede, um proxy reverso, um túnel) passa a dar 400. É o comportamento desejado.
- A checagem de `Origin` protege contra navegadores, não contra um programa local malicioso, que pode forjar o cabeçalho. Quem já executa código na máquina está fora do modelo de ameaça.
- O limite de tamanho usa o `Content-Length` declarado: um corpo em partes (sem `Content-Length`) não é barrado nessa camada e cai no limite de cada foto, que continua valendo. O navegador sempre envia o tamanho num formulário.
- Os testes de API usam `base_url="http://127.0.0.1"`, porque o host padrão do cliente de teste (`testserver`) é recusado.
