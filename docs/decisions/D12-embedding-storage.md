# D12: armazenamento de embeddings

- **Status:** Aceita
- **Data:** 2026-10-18
- **Dona:** Paola

## Contexto

A aplicação da Parte 2 cadastra uma pessoa a partir de uma ou mais fotos (seção 37), compara um rosto novo com os cadastros (seção 38) e precisa excluir uma pessoa com tudo que está associado a ela: cadastro, embedding, fotografias armazenadas e outros dados (seção 39). A coleta segue a minimização de dados (seção 12), e os dados privados ficam só no notebook da RTX 4060, dentro do cofre VeraCrypt (D10 e [docs/privacy.md](../privacy.md)). O volume é pequeno: algumas dezenas de pessoas, com 10 a 30 embeddings de 512 dimensões cada.

## Decisão

**SQLite em `data/private/app.db`**, um arquivo só, dentro do cofre (a pasta é uma junção para ele) e ignorado pelo Git (`*.db` no `.gitignore` e no hook de pre-commit).

| Tabela | Colunas |
|---|---|
| `persons` | `id` (chave), `name` (único), `created_at` |
| `embeddings` | `id` (chave), `person_id` (chave estrangeira para `persons.id`, `ON DELETE CASCADE`), `vector` (float32 como BLOB), `dimension`, `model_id`, `created_at` |

- **A aplicação não guarda fotos**, só embeddings. A foto do cadastro ou do reconhecimento é processada em memória e descartada.
- **Exclusão:** apagar a linha da pessoa em `persons` leva os embeddings dela pelo cascade.
- **Toda conexão** liga `PRAGMA foreign_keys = ON`, senão o SQLite ignora o cascade, e `PRAGMA secure_delete = ON`, para o conteúdo apagado ser sobrescrito com zeros no arquivo em vez de ficar em páginas livres.
- O vetor é gravado com `numpy.ndarray.astype(float32).tobytes()` e lido com `numpy.frombuffer`, com a dimensão conferida na leitura. Sem `pickle`.
- Cada embedding guarda o `model_id` do modelo que o gerou. A aplicação só compara com embeddings do modelo configurado, porque vetores de modelos diferentes não estão no mesmo espaço.

## Alternativas consideradas

- **Arquivos `.npz` por pessoa:** simples, mas a exclusão vira apagar arquivos à mão, sem transação, e o cadastro duplicado e a listagem precisariam de um índice próprio.
- **Banco vetorial (FAISS, Chroma) ou PostgreSQL com pgvector:** feitos para milhões de vetores. Com algumas centenas, a busca linear com NumPy leva menos de um milissegundo, e eles trariam mais uma dependência ou um servidor para instalar e proteger.
- **Guardar as fotos do cadastro** para poder recalcular os embeddings com outro modelo: contraria a minimização de dados, e trocar de modelo é raro. Se acontecer, a pessoa é cadastrada de novo.
- **JSON com os vetores:** sem chave estrangeira nem transação, e cada exclusão reescreveria o arquivo inteiro.

## Consequências

- A exclusão de uma pessoa (seção 39 e spec 006) é um único `DELETE` em `persons`. Um teste confere que, depois dele, a tabela `embeddings` não tem nenhuma linha da pessoa.
- Se a estratégia escolhida no E07 treina um modelo (k-NN ou SVM, D6), ele é refeito em memória a partir do banco ao iniciar a aplicação e depois de cada cadastro ou exclusão, e nunca é salvo em disco. Assim, a exclusão no banco basta para a pessoa sumir do reconhecimento, sem classificador antigo guardando dados dela.
- Os embeddings do banco são dados biométricos derivados: ficam só no cofre e entram na eliminação ao fim da disciplina ([docs/privacy.md](../privacy.md)).
- O nome é único na tabela, o que cobre o cadastro duplicado pelo nome. A checagem de que o mesmo rosto já está cadastrado com outro nome é definida na spec do cadastro.
- `sqlite3` é da biblioteca padrão do Python, então não entra dependência nova. Os testes usam um banco em memória (`:memory:`), sem tocar no cofre.
