# Spec 001: detecção facial

- **Status:** Em revisão
- **Data:** 2026-10-28
- **Dona:** Paola
- **Issue:** #35
- **Relacionadas:** [D03](../decisions/D03-face-detection-alignment.md), [002](002-person-enrollment.md), [003](003-face-recognition.md)

## Problema

Todo fluxo da aplicação (cadastro e reconhecimento) começa com uma imagem que pode não ter rosto, ter vários ou nem ser uma imagem. A aplicação precisa transformar a imagem em um recorte facial único e padronizado, ou recusá-la com um erro claro (seções 37 e 38 do enunciado).

## Requisitos

- **R1.** Receber uma imagem em memória (`PIL.Image`) ou um arquivo enviado pela interface e devolver o recorte 160×160 do único rosto encontrado.
- **R2.** Usar o mesmo pipeline dos experimentos (`FacePreprocessor`, MTCNN, `--alignment crop`, margem 0, probabilidade mínima 0,9, seleção `strict`), para o threshold da Parte 1 valer na aplicação (D03).
- **R3.** Rejeitar imagem sem rosto, com mais de um rosto ou que não possa ser decodificada, com uma exceção específica para cada caso.
- **R4.** Respeitar a orientação EXIF das fotos de celular.
- **R5.** Não gravar a imagem nem o recorte em disco. Tudo é processado em memória e descartado (D12).

## Critérios de aceitação

| # | Dado | Quando | Então |
|---|---|---|---|
| A1 | uma imagem com um único rosto com probabilidade ≥ 0,9 | a imagem é processada | devolve uma imagem RGB 160×160 |
| A2 | uma imagem sem rosto (paisagem, parede) | a imagem é processada | lança `NoFaceDetectedError` |
| A3 | uma imagem em que o melhor rosto tem probabilidade < 0,9 | a imagem é processada | lança `NoFaceDetectedError` |
| A4 | uma imagem com duas ou mais faces acima de 0,9 | a imagem é processada | lança `MultipleFacesError` com `face_count` igual ao número de faces, sem escolher nenhuma |
| A5 | um arquivo corrompido, vazio ou que não é imagem | a imagem é carregada | lança `InvalidImageError` |
| A6 | uma foto de celular com orientação EXIF rotacionada | a imagem é carregada | o rosto chega na orientação correta ao detector |
| A7 | qualquer um dos casos acima | o processamento termina, com sucesso ou erro | nenhum arquivo é criado em disco |
| A8 | o detector é um fake que devolve caixas pré-definidas | o serviço é testado | os testes rodam sem rede, sem GPU e sem baixar pesos |

## Interface

Reaproveita o que já existe em `src/facial_recognition_system/preprocessing/`: `FaceDetector` (`Protocol`), `FacePreprocessor`, `load_image` e as exceções de `errors.py`. Não cria camada nova. A aplicação só instancia o `FacePreprocessor` com `PreprocessingConfig(selection=FaceSelection.STRICT)`.

**Lacuna a fechar na implementação (#43):** hoje `load_image` só aceita um `Path`, e o `InvalidImageError` guarda o caminho. A aplicação recebe os **bytes** do upload ou da câmera, então `load_image` passa a aceitar também bytes (via `io.BytesIO`), e o `InvalidImageError` aceita um nome de origem opcional. Os serviços das specs 002 e 003 recebem bytes e decodificam com ele, o que mantém A5 e A6 na camada certa. O comportamento para `Path` não muda, e os experimentos seguem iguais.

| Erro | Quando | Mensagem para a pessoa (tratada na UI, spec 005) |
|---|---|---|
| `NoFaceDetectedError` | nenhum rosto ≥ 0,9 | "Nenhum rosto encontrado na imagem" |
| `MultipleFacesError` | 2 ou mais rostos | "Mais de um rosto na imagem" |
| `InvalidImageError` | arquivo não decodifica | "Arquivo de imagem inválido" |

Os erros propagam sem captura até a borda (a interface).

## Fora de escopo

- Escolher qual rosto usar quando há vários (a aplicação rejeita, D03).
- Alinhamento por similaridade (implementado, mas não usado, D03).
- Qualidade da imagem além da probabilidade do detector (borrão, iluminação, tamanho mínimo do rosto).

## Testes

- **Unitários** (`@pytest.mark.unit`): A1 a A7 com detector fake e imagens sintéticas.
- **Integração** (`@pytest.mark.integration`, issue #47): A2 e A4 com o MTCNN real em imagens públicas do LFW (uma paisagem e uma montagem com dois rostos).
