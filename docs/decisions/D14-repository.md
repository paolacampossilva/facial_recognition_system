# D14: repositório e fluxo de trabalho

- **Status:** Aceita
- **Data:** 2026-10-18
- **Dona:** Juntas

## Contexto

Cada entrega é o link de um repositório GitHub (seção 2), e o professor avalia versionamento, commits compreensíveis, especificações, critérios de aceitação e revisão de código (seção 41). As duas integrantes precisam saber explicar todo o trabalho (seção 5), mesmo dividindo as tarefas em fatias (division.md). As duas entregas são sequenciais (24/10 e 14/11), e o repositório não pode conter nenhum dado privado em nenhum ponto do histórico (seções 18 e 21).

## Decisão

**Repositório público no GitHub, com a `dev` como branch de integração e a `main` guardando só o que foi entregue.**

- **`main`:** recebe a `dev` só nas entregas, depois da auditoria do histórico. Assim, a `main` é sempre "o que foi entregue" na Parte 1 e na Parte 2.
- **`dev`:** integra o trabalho do dia a dia. Ninguém commita direto nela.
- **Uma branch por tarefa**, saindo da `dev` e ligada à issue: `gh issue develop <N> --base dev --name <tipo>/<assunto> --checkout` (por exemplo, `feat/e04-arcface`, `docs/privacy`).
- **Todo PR é revisado pela outra integrante**, que faz o merge. A revisão é o momento em que a outra aprende a parte que não fez, além de achar erros.
- **Issues** para todas as tarefas, com dona, prazo, critérios de aceite e branch sugerida, nos milestones **Parte 1** e **Parte 2**. O corpo do PR traz `Closes #N`, e o workflow `close-issues-on-dev.yml` fecha a issue no merge na `dev`, já que o GitHub só faz isso sozinho para a branch padrão. Tarefas sem código (coleta, termos, vídeo, entrega) fecham à mão com `gh issue close`.
- **Commits** no formato `tipo(escopo): descrição`, em português (por exemplo, `feat(training): ...`, `docs(decisoes): ...`), com um assunto por commit.
- **Proteção de dados no próprio fluxo:** `.gitignore` e o hook de pre-commit instalados em cada clone, `--no-verify` proibido e auditoria do histórico com `git log --all --stat` antes de cada entrega ([privacy.md](../privacy.md)).

## Alternativas consideradas

- **Commits direto na `main`:** mais rápido, mas sem revisão visível e com a entrega misturada ao trabalho em andamento. Um erro (por exemplo, um arquivo privado) iria direto para o que o professor avalia.
- **GitFlow completo (branches de release e hotfix):** cerimônia demais para duas pessoas e duas entregas. A `dev` mais a `main` cobrem o que importa.
- **Uma branch por pessoa em vez de uma por tarefa:** PRs grandes e misturados, difíceis de revisar, sem ligação clara com as issues.
- **Repositório privado:** o enunciado pede o link do repositório, e um repositório público obriga a disciplina de nunca versionar dado privado, que é exatamente o que a seção 25 cobra.

## Consequências

- O histórico mostra a divisão do trabalho e a revisão cruzada: cada PR tem autora, revisora e a issue que fecha.
- Tarefas que dependem uma da outra (por exemplo, o E04 depende do loop de treino) usam uma branch saindo da outra e indicam no PR a ordem de merge.
- O merge da `dev` na `main` é um passo da entrega (#34 e #53), feito juntas, depois do checklist do Anexo B.
- Implementado em [.github/workflows/close-issues-on-dev.yml](../../.github/workflows/close-issues-on-dev.yml) e no template de PR em [.github/pull_request_template.md](../../.github/pull_request_template.md). O fluxo passo a passo está no [division.md](../division.md), em "Organização".
