# Spec 007: configuração externa e logs

- **Status:** Em revisão
- **Data:** 2026-11-16
- **Dona:** Valentina
- **Issue:** #41
- **Relacionadas:** [003](003-face-recognition.md), [004](004-unknown-person.md), [005](005-camera-interface.md), [006](006-person-deletion.md), [D08](../decisions/D08-threshold-selection.md), [D12](../decisions/D12-embedding-storage.md), [D16](../decisions/D16-chosen-model.md), [D17](../decisions/D17-interface-web-lia.md), [privacy.md](../privacy.md)

## Problema

O que muda de uma máquina para outra (o caminho do banco e do checkpoint, a GPU, a porta) e o threshold da aplicação não podem ficar fixos no código: as specs 003 e 004 exigem trocar o threshold "sem alterar código", e a seção 41 do enunciado avalia configuração externa e logs. Os logs, por sua vez, são dado privado quando trazem informação das pessoas (seção 17), então precisam registrar o que aconteceu sem registrar quem.

## Requisitos

### Configuração

- **R1.** Toda a configuração da aplicação fica numa classe `Settings` (pydantic-settings, D17), em `src/app/config.py`. Nenhum outro módulo lê variável de ambiente.
- **R2.** Cada campo é lido de uma variável de ambiente com o prefixo `FACE_APP_`:

| Variável | Padrão | Valores aceitos | Para quê |
|---|---|---|---|
| `FACE_APP_THRESHOLD` | `0.766` | maior que 0 e até 1 | similaridade mínima para "mesma pessoa", no reconhecimento e na duplicidade do cadastro (E08, D08) |
| `FACE_APP_DATABASE_PATH` | `data/private/app.db` | caminho de arquivo | banco SQLite de pessoas e embeddings (D12) |
| `FACE_APP_CHECKPOINT_PATH` | `checkpoints/E06/best.pt` | caminho de arquivo, ou vazio | checkpoint do modelo escolhido (D16). Vazio usa o FaceNet pré-treinado, sem fine-tuning (E01) |
| `FACE_APP_DEVICE` | `auto` | `auto`, `cpu`, `cuda` | onde o modelo roda. `auto` usa a GPU quando existe |
| `FACE_APP_PORT` | `8000` | 1024 a 65535 | porta do servidor local |
| `FACE_APP_MAX_UPLOAD_MB` | `10` | 1 a 50 | tamanho máximo de cada foto enviada (spec 005, R14) |
| `FACE_APP_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` | nível mínimo dos logs |

- **R3.** Ordem de precedência: variável de ambiente, depois o arquivo `.env` na raiz do repositório, depois o padrão da tabela. Na linha de comando, `--port` vale mais que todos.
- **R4.** O `.env` é pessoal e fica fora do Git (já está no `.gitignore`). O repositório traz um `.env.example` com todas as variáveis da R2, os padrões e um comentário curto de cada uma.
- **R5.** O `.env` só aceita as variáveis da R2. Uma linha desconhecida (por exemplo, `FACE_APP_TRESHOLD`, com erro de digitação) é um erro de configuração, para o valor não ser ignorado em silêncio.
- **R6.** Configuração inválida (valor fora da faixa, tipo errado ou variável desconhecida no `.env`) lança `ConfigError`. Em `python -m app`, a aplicação **não sobe**: mostra uma mensagem com o nome da variável e o que ela aceita e termina com código 1, sem traceback.
- **R7.** Os caminhos relativos valem a partir da pasta onde a aplicação é iniciada, que o README pede que seja a raiz do repositório.
- **R8.** **O que não é configurável, de propósito:** o endereço `127.0.0.1` (a aplicação não tem autenticação e guarda biometria, spec 005, R14), a pasta da interface (faz parte do código), os PRAGMAs do SQLite (spec 006, R4) e a arquitetura do modelo (FaceNet, D16; só o checkpoint muda).

### Logs

- **R9.** Os logs vão só para o terminal (stderr). A aplicação não grava arquivo de log, para não criar mais um arquivo com dados da aplicação a proteger e apagar ao fim da disciplina (privacy.md).
- **R10.** Formato de cada linha: `hora nível módulo: mensagem campo=valor ...`, por exemplo `14:02:31 INFO app.core.enrollment: person enrolled person_id=3 embedding_count=4`.
- **R11.** **Nunca vão para o log:** nome de pessoa, foto (bytes ou conteúdo), embedding, nome de arquivo enviado (o nome do arquivo pode ser o nome da pessoa) e o conteúdo do banco.
- **R12.** Para a R11 não depender de cada chamada, o formatador só escreve os campos extras de uma **lista permitida**: `person_id`, `embedding_count`, `deleted_embeddings`, `status` e `similarity`. Qualquer outro campo passado em `extra` é descartado. Um campo novo só entra no log depois de entrar na lista, com revisão.
- **R13.** O que cada serviço registra, em `INFO`:

| Evento | Campos |
|---|---|
| cadastro concluído (spec 002) | `person_id`, `embedding_count` |
| reconhecimento (specs 003 e 004) | `status`, `similarity` |
| exclusão (spec 006) | `person_id`, `deleted_embeddings` |

- **R14.** As rotas da API não registram nada além do log de acesso do servidor (método, caminho e status, em que o caminho só tem o `id`, nunca um nome). Um erro inesperado (500) é registrado em `ERROR` com o traceback, para depuração, e a resposta não vaza detalhe interno (spec 005, A20).

## Critérios de aceitação

| # | Dado | Quando | Então |
|---|---|---|---|
| A1 | nenhum `.env` e nenhuma variável `FACE_APP_` | a configuração é carregada | os valores são os padrões da R2 |
| A2 | `FACE_APP_THRESHOLD=0.8` no ambiente | a configuração é carregada | o threshold é 0,8, e o `GET /api/config` devolve 0,8, sem mudar código (spec 003, A5) |
| A3 | `FACE_APP_THRESHOLD=0.5` no `.env` e `0.9` no ambiente | a configuração é carregada | vale 0,9 (o ambiente vence o `.env`) |
| A4 | `FACE_APP_THRESHOLD=2`, `FACE_APP_PORT=80` ou `FACE_APP_DEVICE=tpu` | a configuração é carregada | lança `ConfigError` com o nome da variável |
| A5 | `FACE_APP_TRESHOLD=0.5` no `.env` | a configuração é carregada | lança `ConfigError` apontando a variável desconhecida |
| A6 | `FACE_APP_CHECKPOINT_PATH=` (vazio) | a configuração é carregada | o checkpoint é `None` (FaceNet pré-treinado) |
| A7 | uma configuração inválida | roda `python -m app` | mostra `error:` com a variável, não sobe o servidor e termina com código 1 |
| A8 | `FACE_APP_PORT=9000` e `--port 9100` | roda `python -m app` | o servidor sobe na 9100 |
| A9 | `FACE_APP_MAX_UPLOAD_MB=1` | envia uma foto de 2 MB | 422 `invalid_request` (spec 005), sem chamar o serviço |
| A10 | o `.env.example` do repositório | é comparado com a `Settings` | tem exatamente as variáveis da R2 |
| A11 | um registro de log com `extra={"name": "Ana", "person_id": 3}` | é formatado | a linha tem `person_id=3` e não tem "Ana" |
| A12 | cadastro, reconhecimento e exclusão de "Ana Teste" com os serviços reais e um modelo fake | os logs formatados são inspecionados | contêm os campos da R13 e não contêm "Ana" nem números do embedding |
| A13 | `FACE_APP_LOG_LEVEL=WARNING` | um cadastro é feito | nenhuma linha `INFO` aparece |

## Interface

```python
class Settings(BaseSettings):  # src/app/config.py, prefixo FACE_APP_, lê o .env
    threshold: float = 0.766
    database_path: Path = Path("data/private/app.db")
    checkpoint_path: Path | None = Path("checkpoints/E06/best.pt")
    device: Literal["auto", "cpu", "cuda"] = "auto"
    port: int = 8000
    max_upload_mb: int = 10
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @property
    def max_upload_bytes(self) -> int: ...


def load_settings() -> Settings: ...  # lança ConfigError
def configure_logging(level: str) -> None: ...  # src/app/logs.py
```

Exceção nova: `ConfigError`, herdando de `AppError`. `create_app` passa a receber a `Settings` (o threshold e o limite de upload vêm dela), e o `python -m app` carrega a configuração antes de subir o servidor, para a R6.

## Alternativas consideradas

- **YAML, como os experimentos (D11):** os YAMLs de `configs/` descrevem um experimento e são versionados. A configuração da aplicação muda por máquina e não é versionada, e o pydantic-settings já lê ambiente e `.env` com validação, sem mais um formato para manter.
- **Arquivo de log:** facilitaria olhar o histórico, mas seria mais um arquivo com dados da aplicação a proteger e excluir (R9).

## Fora de escopo

- Configuração pela interface web (a configuração é do computador, não de quem usa a tela).
- Rotação de logs, envio de logs para serviços externos e métricas.
- Recarregar a configuração sem reiniciar a aplicação.

## Testes

- **Unitários** (`tests/test_config.py` e `tests/test_logs.py`): A1 a A6, A10, A11 e A13 com `monkeypatch` no ambiente e um `.env` numa pasta temporária; A7 e A8 com o servidor fake do `test_launcher.py`; A9 com o `TestClient`; A12 com os serviços reais, o repositório em memória e o modelo fake de `tests/app_support.py`.
- **E2E** (#50): roda a aplicação com uma `Settings` de teste (banco temporário, porta livre), sem ler o `.env` de quem roda.
