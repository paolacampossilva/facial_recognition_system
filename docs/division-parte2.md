# Divisão do trabalho da Parte 2, aplicação (Paola e Valentina)

> Substitui o "Cronograma da Parte 2" do [division.md](division.md), que segue como plano original de 18/10. Decisões e valores que mudarem na implementação ficam nos [ADRs](decisions/README.md) e no [README](../README.md), que prevalecem sobre este documento.

Prazo do Moodle: **Parte 2 vence dom 22/11/2026 23:59** ("Trabalho 2: Aplicativo").
Cada dia de atraso custa 1 ponto (máximo de 2 dias), então a meta interna é entregar no **sábado anterior (21/11)**. O trabalho começa na **seg 16/11**, o que deixa 6 dias (seg 16 a sáb 21).

## Princípio da divisão

Igual à Parte 1: cada uma é dona de uma fatia vertical (escreve a spec, implementa, testa e documenta) e a outra revisa o PR, para as duas saberem explicar tudo (seção 5 do enunciado).

- **Paola:** núcleo da aplicação (cadastro, reconhecimento, exclusão), armazenamento, testes do núcleo e README com a arquitetura.
- **Valentina:** interface, configuração e logs, integração com o modelo escolhido, teste e2e e vídeo.
- **Juntas:** revisão cruzada, checklist do Anexo B, auditoria do histórico, `dev` na `main`, tag e release `parte-2`.

## Situação inicial (16/11)

- Specs 001 a 004 e 006 escritas pela Paola (branch `docs/specs-parte2`, falta revisão e merge na `dev`).
- Faltam as specs 005 (camera-interface) e 007 (logging-config), da Valentina.
- Decisões herdadas da Parte 1: D12 (SQLite, só embeddings, sem fotos), D13 (Streamlit, uma tela com abas), D15 (só as duas aparecem no vídeo) e o modelo escolhido na D16.
- A `main` está na tag `parte-1` (`5430219`).

## Repartição

| | Paola | Valentina |
|---|---|---|
| Specs | 001 a 004 e 006 (prontas, falta revisão e merge) | 005 e 007 |
| Código | `src/app/core/` (`EnrollmentService`, `RecognitionService`, `DeletionService`, modelos Pydantic, exceções), `src/app/storage/` (SQLite) | `src/app/ui/` (Streamlit, 3 abas), `Settings` e `.env.example`, logs sem dados pessoais, carga do modelo da D16 |
| Testes | unitários com detector e modelo fake, integração com o modelo real em imagens públicas | e2e: cadastro, reconhecimento, desconhecido, exclusão |
| Entrega | README com arquitetura (diagrama) e como criar o próprio banco | gravação e edição do vídeo 2 |
| Juntas | revisão cruzada de PRs, Anexo B, auditoria do histórico, `dev` na `main`, tag e release `parte-2` | |

## Cronograma (seg 16/11 a sáb 21/11)

| Dia | Paola | Valentina | Juntas |
|---|---|---|---|
| **seg 16** | Abrir o PR das specs 001 a 004 e 006; storage SQLite (`PRAGMA foreign_keys = ON`) e modelos Pydantic; `EnrollmentService` | Specs 005 e 007; `Settings`, `.env.example` e logs | Reunião de 30 min: contrato entre as fatias (assinaturas dos serviços e modelos Pydantic), issues e milestone da Parte 2 |
| **ter 17** | `RecognitionService` e comportamento de desconhecido (specs 003 e 004) | Esqueleto da UI em Streamlit com serviços fake; carga do modelo da D16 | Revisão cruzada das 7 specs, de manhã, antes de seguir |
| **qua 18** | `DeletionService` em cascata, exceções e testes unitários com fakes | Abas Cadastro e Reconhecimento ligadas aos serviços reais | Revisão cruzada dos PRs do núcleo |
| **qui 19** | Testes de integração e caminhos de erro; README e diagrama de arquitetura | Aba Pessoas cadastradas (listar e excluir), mensagens em português e teste e2e | Congelar o código à noite (só correção); `pytest`, `ruff` e `pre-commit` |
| **sex 20** | Anexo B (LGPD); roteiro e explicação da arquitetura no vídeo | Roteiro, gravação e edição do vídeo 2 | Revisão cruzada do README, ensaio cruzado e auditoria do histórico (`git log --all --stat`) |
| **sáb 21** | | | `dev` na `main`, tag e release `parte-2`, **entrega** no Moodle |
| dom 22 | | | Folga para imprevistos (prazo 23:59) |

## Caminho crítico e cortes

- Núcleo e UI andam em paralelo pelo contrato de segunda e pelos serviços fake. A integração com os serviços reais é na quarta.
- Se atrasar, cortar nesta ordem: testes de integração extras, polimento das mensagens, detalhe do diagrama.
- Não cortar: exclusão em cascata, Anexo B, vídeo com as duas (D15) e `.env.example`.
- Vídeo ao vivo (`streamlit-webrtc`) segue fora do escopo (D13).

## Fechamento da entrega (juntas)

- `pytest` e `ruff` passando, o fluxo completo rodado do zero a partir do README e o Anexo B conferido item por item.
- Auditoria do histórico Git (imagens, `.db` e `.pt`).
- Ensaio cruzado: Paola explica a interface e o e2e, Valentina explica o núcleo e o armazenamento.
- Depois da disciplina: excluir fotos, embeddings, banco e checkpoints privados.

## Organização

- Toda tarefa vira uma issue no milestone **Parte 2** (prazo interno 21/11), com dona e link para a spec. Cada PR sai da `dev` e é revisado pela outra, como na Parte 1 (seção "Organização" do [division.md](division.md)).
- A `dev` entra na `main` só na entrega, então a `main` continua sendo "o que foi entregue". Ao final, tag `parte-2` e release no GitHub.

## Resumo da carga

| | Paola | Valentina |
|---|---|---|
| Parte 2 | specs 001 a 004 e 006, núcleo, storage, exclusão, testes, README e arquitetura | specs 005 e 007, interface, config e logs, integração, e2e, vídeo 2 |
| Juntas | revisão cruzada de PRs, Anexo B, auditoria, entrega, tag e release | |
