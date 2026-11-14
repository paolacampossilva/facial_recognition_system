# Divisão do trabalho da Parte 2, aplicação (Paola ~75%, Valentina ~25%)

> Substitui o "Cronograma da Parte 2" do [division.md](division.md), que segue como plano original de 18/10. Decisões e valores que mudarem na implementação ficam nos [ADRs](decisions/README.md) e no [README](../README.md), que prevalecem sobre este documento.

Prazo do Moodle: **Parte 2 vence dom 22/11/2026 23:59** ("Trabalho 2: Aplicativo").
Cada dia de atraso custa 1 ponto (máximo de 2 dias), então a meta interna é entregar no **sábado anterior (21/11)**. O trabalho começa na **seg 16/11**, o que deixa 6 dias (seg 16 a sáb 21).

## Princípio da divisão

Cada uma é dona de uma fatia vertical (escreve a spec, implementa, testa e documenta) e a outra revisa o PR, para as duas saberem explicar tudo (seção 5 do enunciado). Por a Valentina estar com menos disponibilidade, a carga é de cerca de 75% para a Paola e 25% para a Valentina, e a Valentina segue revisando todos os PRs. O vídeo 2 é a exceção: é dividido meio a meio.

- **Paola:** núcleo da aplicação (cadastro, reconhecimento, exclusão), armazenamento, integração com o modelo escolhido, a interface inteira (API FastAPI e front-end web, [D17](decisions/D17-interface-web-lia.md)), testes (unitários, integração e da API), README com a arquitetura, roteiro do vídeo e metade do vídeo.
- **Valentina:** spec 007, configuração e logs, teste e2e da interface, Anexo B, auditoria do histórico e metade do vídeo (inclui a edição).
- **Juntas:** revisão cruzada, ensaio cruzado, `dev` na `main`, tag e release `parte-2`.

### Vídeo 2 (50/50)

- **Paola:** escreve o roteiro e apresenta inicialização, cadastro e arquitetura.
- **Valentina:** apresenta reconhecimento, pessoa desconhecida, situação de erro e conclusão, e faz a edição e a exportação.
- Cada uma grava a própria parte, seguindo a D15. As duas assistem à versão final inteira antes de publicar (item do Anexo B "o vídeo respeita as autorizações").

## Situação inicial (16/11)

- Specs 001 a 004 e 006 escritas pela Paola e já na `dev` (PR #98), falta a revisão cruzada.
- Faltam a spec 005 (camera-interface, agora da Paola, que faz a interface) e a 007 (logging-config), da Valentina.
- Decisões herdadas da Parte 1: D12 (SQLite, só embeddings, sem fotos), D15 (só as duas aparecem no vídeo) e o modelo escolhido na D16. A D13 (Streamlit) foi substituída pela [D17](decisions/D17-interface-web-lia.md) (FastAPI e front-end web, três abas).
- A `main` está na tag `parte-1` (`5430219`).

## Repartição

| | Paola | Valentina |
|---|---|---|
| Specs | 001 a 004 e 006 (na `dev`, falta a revisão cruzada) e 005 | 007 |
| Código | `src/app/core/` (`EnrollmentService`, `RecognitionService`, `DeletionService`, modelos Pydantic, exceções), `src/app/storage/` (SQLite), carga do modelo da D16, interface inteira: API FastAPI e `src/app/web/` (abas Cadastro, Reconhecimento e Pessoas cadastradas) | `Settings` e `.env.example`, logs sem dados pessoais |
| Testes | unitários com detector e modelo fake, integração com o modelo real em imagens públicas, contrato da API e CSP | e2e da interface (Chrome sem interface: cadastro, reconhecimento, desconhecido, exclusão; falha se houver erro de console), testes do `Settings` |
| Entrega | README com arquitetura (diagrama) e como criar o próprio banco, roteiro do vídeo 2, metade do vídeo 2 (grava e apresenta) | Anexo B item por item, auditoria do histórico Git (`git log --all --stat`), metade do vídeo 2 (grava e apresenta), edição e exportação |
| Juntas | revisão cruzada de PRs, ensaio cruzado, `dev` na `main`, tag e release `parte-2`, entrega no Moodle | |

## Cronograma (seg 16/11 a sáb 21/11)

| Dia | Paola | Valentina | Juntas |
|---|---|---|---|
| **seg 16** | Storage SQLite (`PRAGMA foreign_keys = ON`) e modelos Pydantic; `EnrollmentService` | Spec 007 | Reunião de 30 min: contrato entre as fatias (assinaturas dos serviços e modelos Pydantic), issues e milestone da Parte 2 |
| **ter 17** | Spec 005; `RecognitionService`, comportamento de desconhecido (specs 003 e 004) e carga do modelo da D16 | `Settings`, `.env.example` e logs | Revisão cruzada das 7 specs, de manhã, antes de seguir |
| **qua 18** | `DeletionService` em cascata, exceções, testes unitários com fakes; API FastAPI e esqueleto do front-end (tokens, tema, diálogos) com serviços fake | Testes do `Settings`; começa o roteiro do e2e | Revisão cruzada dos PRs do núcleo |
| **qui 19** | Três abas ligadas aos serviços reais; testes de integração, da API e caminhos de erro; README e diagrama | Teste e2e da interface, Anexo B (itens técnicos), revisão do README | Congelar o código à noite (só correção); `pytest`, `ruff` e `pre-commit` |
| **sex 20** | Roteiro do vídeo 2; grava inicialização, cadastro e arquitetura | Auditoria do histórico (`git log --all --stat`); Anexo B (itens finais); grava reconhecimento, desconhecido, erro e conclusão; edita e exporta | Revisão cruzada do README, ensaio cruzado e as duas assistem ao vídeo final |
| **sáb 21** | | | `dev` na `main`, tag e release `parte-2`, **entrega** no Moodle |
| dom 22 | | | Folga para imprevistos (prazo 23:59) |

## Caminho crítico e cortes

- Núcleo e UI andam em paralelo pelo contrato de segunda e pelos serviços fake. A integração com os serviços reais é na quarta. Como a Paola concentra o caminho crítico, a Valentina assume qualquer item pontual que sobrar de tempo (ex.: mensagens em português, testes extras).
- Se atrasar, cortar nesta ordem: testes de integração extras, polimento das mensagens, detalhe do diagrama.
- Não cortar: exclusão em cascata, Anexo B, vídeo com as duas (D15, meio a meio) e `.env.example`.
- Reconhecimento contínuo quadro a quadro da webcam segue fora do escopo (D17): a interface captura uma foto por vez.

## Fechamento da entrega (juntas)

- `pytest` e `ruff` passando, o fluxo completo rodado do zero a partir do README e o Anexo B conferido item por item.
- Auditoria do histórico Git (imagens, `.db` e `.pt`).
- Ensaio cruzado: a Valentina explica o núcleo, o armazenamento e a interface (mesmo sem ter escrito, pela revisão dos PRs) e a Paola explica config, logs e o e2e.
- Depois da disciplina: excluir fotos, embeddings, banco e checkpoints privados.

## Organização

- Toda tarefa vira uma issue no milestone **Parte 2** (prazo interno 21/11), com dona e link para a spec. Cada PR sai da `dev` e é revisado pela outra, como na Parte 1 (seção "Organização" do [division.md](division.md)).
- A `dev` entra na `main` só na entrega, então a `main` continua sendo "o que foi entregue". Ao final, tag `parte-2` e release no GitHub.

## Resumo da carga

| | Paola | Valentina |
|---|---|---|
| Parte 2 (~75% / ~25%) | specs 001 a 006, núcleo, storage, modelo, interface inteira, testes do núcleo e da API, README, roteiro do vídeo, tag e release | spec 007, config e logs, e2e da interface, Anexo B, auditoria do histórico |
| Vídeo 2 (50/50) | roteiro; apresenta inicialização, cadastro e arquitetura | apresenta reconhecimento, desconhecido, erro e conclusão; edição e exportação |
| Juntas | revisão cruzada de PRs, ensaio cruzado, `dev` na `main`, entrega | |
