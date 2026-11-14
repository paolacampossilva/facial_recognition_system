# Spec 006: exclusão de pessoa

- **Status:** Em revisão
- **Data:** 2026-10-28
- **Dona:** Paola
- **Issue:** #40
- **Relacionadas:** [002](002-person-enrollment.md), [003](003-face-recognition.md), [D12](../decisions/D12-embedding-storage.md), [privacy.md](../privacy.md)

## Problema

Como exercício de Privacy by Design, a aplicação precisa excluir uma pessoa cadastrada com tudo que está associado a ela: cadastro, embeddings, fotografias armazenadas e outros dados da identidade (seção 39).

## Requisitos

- **R1.** Excluir uma pessoa pelo `id` remove a linha em `persons` e, em cascata, todos os seus embeddings (D12).
- **R2.** **Fotografias:** a aplicação não guarda fotos (D12), então não há arquivos de foto a apagar. A spec deixa isso verificável: um teste confere que nenhuma foto é gravada no cadastro (spec 002, A11).
- **R3.** **Cache:** não há cache a invalidar. O reconhecimento lê a galeria do repositório a cada chamada (spec 003), então a pessoa some do reconhecimento assim que a exclusão é confirmada. Se um cache for introduzido no futuro, esta spec passa a exigir a invalidação dele.
- **R4.** **Banco:** toda conexão liga `PRAGMA foreign_keys = ON` (senão o cascade é ignorado) e `PRAGMA secure_delete = ON` (para o conteúdo apagado ser sobrescrito com zeros e não ficar em páginas livres).
- **R5.** Excluir um `id` que não existe lança `PersonNotFoundError`. Nada é alterado.
- **R6.** A exclusão é **atômica**: ou remove a pessoa e os embeddings, ou não remove nada.
- **R7.** Os logs da exclusão registram o `id` e a quantidade de embeddings removidos, **nunca o nome** (spec 007).
- **R8.** A confirmação ("tem certeza?") é responsabilidade da interface (spec 005). O serviço apaga quando é chamado.

## Critérios de aceitação

| # | Dado | Quando | Então |
|---|---|---|---|
| A1 | "Ana" com 3 embeddings e "Bia" com 2 | exclui "Ana" | `persons` não tem "Ana", `embeddings` não tem nenhuma linha dela, e "Bia" continua com 2 embeddings |
| A2 | "Ana" excluída | consulta `embeddings` por `person_id` da "Ana" | 0 linhas |
| A3 | "Ana" excluída | reconhece o rosto dela (spec 003) | o resultado é `unknown`, ou `EmptyGalleryError` se não sobrou ninguém, sem reiniciar |
| A4 | "Ana" excluída | cadastra "Ana" de novo | o cadastro funciona (o nome fica livre) |
| A5 | um `id` inexistente | exclui | lança `PersonNotFoundError`, e o banco não muda |
| A6 | uma falha no meio da exclusão (repositório fake que lança erro) | exclui | nada foi removido (atomicidade) |
| A7 | uma conexão aberta pelo repositório | a conexão é inspecionada | `foreign_keys` e `secure_delete` estão ligados |
| A8 | a exclusão concluída | os logs são inspecionados | contêm o `id` e a contagem, e não o nome |
| A9 | a exclusão concluída | o resultado é inspecionado | devolve o `id` e quantos embeddings foram removidos |
| A10 | "Ana" cadastrada e depois excluída, em disco | os bytes de `app.db` são inspecionados | o vetor da "Ana" não aparece no arquivo (efeito do `secure_delete`) |

## Interface

```python
class DeletionResult(BaseModel):
    person_id: int
    deleted_embeddings: int


class DeletionService:
    def delete(self, person_id: int) -> DeletionResult: ...
```

Exceção nova: `PersonNotFoundError(person_id: int)`, herdando da exceção base da aplicação. Propaga sem captura até a interface.

## Fora de escopo

- Excluir todas as pessoas de uma vez (basta apagar `data/private/app.db`).
- Exclusão agendada ou com prazo de retenção.
- Apagar os dados do vídeo e os do cofre da Parte 1: isso segue o [privacy.md](../privacy.md), ao fim da disciplina.

## Testes

- **Unitários:** A1 a A9 com banco `:memory:` e repositório fake, e A5 e A6 como caminhos de erro.
- **Integração:** A10 com um banco em arquivo temporário, procurando os bytes do vetor no arquivo depois da exclusão.
- **E2E** (#50): cadastro → reconhecimento → exclusão → reconhecimento como desconhecido.
