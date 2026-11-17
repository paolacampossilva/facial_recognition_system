# D19: reconhecimento de vários rostos numa imagem

- **Status:** Aceita
- **Data:** 2026-11-17
- **Dona:** Paola

## Contexto

Até aqui a aplicação trata uma imagem como se ela tivesse um rosto só: o pipeline usa `FaceSelection.STRICT` (D03), que rejeita fotos com mais de um rosto, e o reconhecimento devolve um único resultado. A issue #103 (extra, aprovada pelo professor em 17/11) pede que o **reconhecimento** mostre todos os rostos de uma foto, conhecidos ou desconhecidos, cada um com a sua caixa. O **cadastro** continua com um rosto por foto, porque cadastrar a pessoa errada de uma foto de grupo seria pior do que pedir outra foto.

Duas coisas pesam. O threshold de 0,766 foi medido no E08 com fotos de um rosto grande (selfies), e um rosto numa foto de grupo costuma ser menor e mais inclinado. E reconhecer todo mundo numa foto aumenta o risco de tratar quem não consentiu (LGPD, seção 25 do enunciado).

## Decisão

- **Dois caminhos no mesmo `FacePreprocessor`.** `process` (estrito, para o cadastro) não muda. `process_all` (novo, para o reconhecimento) devolve o recorte e a caixa de cada rosto, com o mesmo detector, a mesma probabilidade mínima (0,9), o mesmo recorte e a mesma orientação EXIF. A D03 continua valendo para o cadastro.
- **Filtros do reconhecimento:** só entram rostos cujo menor lado da caixa tem pelo menos `FACE_APP_MIN_FACE_PX` pixels (padrão 80), e no máximo 20, os de maior área. Sem nenhum rosto, o erro continua sendo `no_face_detected`.
- **Cada rosto é decidido sozinho**, com o mesmo modelo (E06), o mesmo vizinho mais próximo e o mesmo threshold (0,766). Um rosto desconhecido continua sem nome (spec 004, R2), mesmo que o nome do vizinho mais próximo apareça em outro rosto da foto.
- **Contrato:** `POST /api/recognitions` devolve `{image_width, image_height, faces: [{box, status, person_name, similarity}]}`, da esquerda para a direita, com a caixa em pixels da imagem depois da orientação EXIF.
- **Caixas na interface:** uma camada de elementos sobre a `<img>`, posicionada em porcentagem, e não um `<canvas>`. Cada caixa leva um rótulo acessível e uma cor de token, e o navegador já aplica a orientação EXIF à `<img>`, então as coordenadas da API caem no lugar certo.
- **A mesma pessoa em mais de um rosto** (foto de uma foto, espelho, gêmeos): a lista mostra todos os rostos e a interface avisa ("Ana aparece em 2 rostos desta foto"). Nada é escondido nem deduplicado.
- **Privacidade:** nada é guardado (D12). A interface lembra que todo rosto da foto é comparado com as pessoas cadastradas, e o README e o `docs/privacy.md` registram a limitação.

## Alternativas consideradas

- **Reaproveitar `FaceSelection.LARGEST` ou `CENTER`:** descartam os outros rostos, e a issue pede todos.
- **Trocar o `STRICT` por outro padrão no pipeline inteiro:** quebraria o cadastro e a premissa de que o threshold da Parte 1 vale na aplicação (D03).
- **`<canvas>` para as caixas** (como a issue sugeria): precisaria ler as cores dos tokens por `getComputedStyle`, não dá rótulo acessível por rosto e exige tratar a orientação EXIF à mão.
- **Deixar só o melhor rosto quando a pessoa se repete e marcar os outros como desconhecidos:** rotularia como desconhecido um rosto que passou do threshold, o que contradiz a spec 004.
- **Um threshold próprio para rostos pequenos:** exigiria uma medição que não existe. Fica o tamanho mínimo, que corta o caso mais arriscado.
- **Reconhecer no navegador depois de receber as caixas:** a regra de reconhecimento não pode ficar no front-end (spec 005, R4).

## Consequências

- **O threshold fica fora da faixa medida para rostos pequenos.** O FAR e o FRR de uma foto de grupo **não foram medidos**. O tamanho mínimo de 80 px é um valor provisório, escolhido para cortar rostos pequenos o bastante para perder detalhe ao serem ampliados para 160×160, e não vem de uma medição. A spec 004 e o README registram isso como limitação. Medir o desempenho em foto de grupo (um E09) ficaria para depois da entrega.
- **LGPD:** reconhecer todos os rostos de uma foto trata dados de quem pode não ter consentido. Nada é gravado, o tratamento é em memória e a demonstração e os testes usam só as duas autoras (D15) ou imagens sintéticas e públicas. O risco continua em uso real, e por isso o aviso fica na tela do Reconhecimento.
- **Custo:** um embedding por rosto, no máximo 20 por foto, numa única consulta em lote à galeria.
- **Testes e contrato mudam:** o corpo da resposta de `/api/recognitions` deixa de ser um objeto único, e o teste que exigia `MultipleFacesError` no reconhecimento passa a exigir vários resultados. O cadastro mantém os seus testes.
- **Vídeo (D15):** a situação de erro "foto com nós duas juntas" passa a existir só no Cadastro. No Reconhecimento, a mesma foto vira uma demonstração (as duas reconhecidas juntas).
