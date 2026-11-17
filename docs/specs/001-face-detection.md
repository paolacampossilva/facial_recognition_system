# Spec 001: detecção facial

- **Status:** Em revisão
- **Data:** 2026-10-28
- **Dona:** Paola
- **Issue:** #35, #103 (vários rostos no reconhecimento)
- **Relacionadas:** [D03](../decisions/D03-face-detection-alignment.md), [D19](../decisions/D19-multi-face-recognition.md), [002](002-person-enrollment.md), [003](003-face-recognition.md)

## Problema

Todo fluxo da aplicação (cadastro e reconhecimento) começa com uma imagem que pode não ter rosto, ter vários ou nem ser uma imagem. O **cadastro** precisa transformar a imagem em um recorte facial único e padronizado, ou recusá-la com um erro claro (seções 37 e 38 do enunciado). O **reconhecimento** (issue #103) aceita imagens com vários rostos e precisa de um recorte e de uma caixa para cada um.

## Requisitos

- **R1.** Receber uma imagem em memória (`PIL.Image`) ou um arquivo enviado pela interface e devolver o recorte 160×160 do único rosto encontrado (caminho estrito, usado pelo cadastro).
- **R2.** Usar o mesmo pipeline dos experimentos (`FacePreprocessor`, MTCNN, `--alignment crop`, margem 0, probabilidade mínima 0,9, seleção `strict` no cadastro), para o threshold da Parte 1 valer na aplicação (D03).
- **R3.** No caminho estrito, rejeitar imagem sem rosto, com mais de um rosto ou que não possa ser decodificada, com uma exceção específica para cada caso.
- **R4.** Respeitar a orientação EXIF das fotos de celular.
- **R5.** Não gravar a imagem nem o recorte em disco. Tudo é processado em memória e descartado (D12).
- **R6.** Para o reconhecimento, existe um segundo caminho, `process_all`, que **não rejeita** vários rostos: devolve o recorte 160×160 e a caixa de cada rosto da imagem. Ele usa a mesma detecção, a mesma probabilidade mínima (0,9), o mesmo recorte (`crop`, margem 0) e a mesma orientação EXIF do caminho estrito, para o threshold da Parte 1 continuar valendo (D19). A seleção (`strict`, `center`, `largest`) não afeta o `process_all`, e o caminho estrito do cadastro não muda.
- **R7.** O `process_all` ignora rostos pequenos demais: o **menor lado da caixa** precisa ser de pelo menos `min_face_px` pixels (configuração `FACE_APP_MIN_FACE_PX`, spec 007, padrão 80). O threshold foi medido em fotos com um rosto grande (E08), e um rosto pequeno ampliado para 160×160 perde detalhe. O valor é provisório (D19).
- **R8.** O `process_all` devolve no máximo `max_faces` rostos (20), os de maior área, para uma foto de grupo muito grande não gerar centenas de embeddings.
- **R9.** As caixas saem em **pixels da imagem depois da orientação EXIF** (as mesmas coordenadas do detector) e a lista sai ordenada da esquerda para a direita (pelo lado esquerdo da caixa, e pelo topo em caso de empate).
- **R10.** Se não sobra nenhum rosto (nenhum acima de 0,9, ou todos pequenos demais), o `process_all` lança `NoFaceDetectedError`, o mesmo erro do caminho estrito.

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
| A9 | uma imagem com três rostos ≥ 0,9 e de tamanho suficiente | `process_all` | devolve três recortes 160×160 com as suas caixas, ordenados da esquerda para a direita |
| A10 | uma imagem com um único rosto | `process_all` | devolve um rosto, com o mesmo recorte do caminho estrito |
| A11 | uma imagem sem rosto, ou com rostos só abaixo de 0,9 | `process_all` | lança `NoFaceDetectedError` |
| A12 | uma imagem com um rosto cujo menor lado é menor que `min_face_px` | `process_all` | o rosto é ignorado, e se era o único, lança `NoFaceDetectedError` |
| A13 | uma imagem com 25 rostos acima de 0,9 | `process_all` com `max_faces=20` | devolve os 20 de maior área, ordenados da esquerda para a direita |
| A14 | uma imagem com dois rostos | o caminho estrito (`process`) | continua lançando `MultipleFacesError` (A4 não muda: o cadastro segue estrito) |
| A15 | uma foto de celular com orientação EXIF rotacionada e vários rostos | `process_all` | as caixas estão na orientação correta, a mesma em que o navegador mostra a foto |

## Interface

Reaproveita o que já existe em `src/facial_recognition_system/preprocessing/`: `FaceDetector` (`Protocol`), `FacePreprocessor`, `load_image` e as exceções de `errors.py`. Não cria camada nova. A aplicação instancia um só `FacePreprocessor`, com `PreprocessingConfig(selection=FaceSelection.STRICT)`: o cadastro chama `process`, e o reconhecimento chama `process_all`.

```python
@dataclass(frozen=True)
class DetectedFace:
    box: BoundingBox  # pixels da imagem depois da EXIF
    probability: float
    crop: Image.Image  # 160×160 RGB


class FacePreprocessor:
    def process(self, image: Image.Image) -> Image.Image: ...  # estrito, não muda
    def process_all(
        self, image: Image.Image, min_face_px: int, max_faces: int
    ) -> list[DetectedFace]: ...
```

**Lacuna fechada na implementação (#43):** `load_image` só aceitava um `Path`, e o `InvalidImageError` guardava o caminho. A aplicação recebe os **bytes** do upload ou da câmera, então `load_image` aceita também bytes (via `io.BytesIO`), e o `InvalidImageError` guarda um nome de origem (`source`, que é `"upload"` quando a imagem veio em bytes). Uma imagem grande demais para ser segura (`DecompressionBombError` do Pillow) também vira `InvalidImageError`. Os serviços das specs 002 e 003 recebem bytes e decodificam com ele, o que mantém A5 e A6 na camada certa. O comportamento para `Path` não muda, e os experimentos seguem iguais.

| Erro | Quando | Mensagem para a pessoa (tratada na UI, spec 005) |
|---|---|---|
| `NoFaceDetectedError` | nenhum rosto ≥ 0,9 | "Nenhum rosto encontrado na imagem" |
| `MultipleFacesError` | 2 ou mais rostos, **só no cadastro** (o reconhecimento aceita vários) | "Mais de um rosto na imagem" |
| `InvalidImageError` | arquivo não decodifica | "Arquivo de imagem inválido" |

Os erros propagam sem captura até a borda (a interface).

## Fora de escopo

- Escolher qual rosto usar quando há vários **no cadastro** (o cadastro rejeita, D03). No reconhecimento todos os rostos são usados (R6).
- Alinhamento por similaridade (implementado, mas não usado, D03).
- Qualidade da imagem além da probabilidade do detector e do tamanho mínimo da R7 (borrão, iluminação, pose).

## Testes

- **Unitários** (`@pytest.mark.unit`): A1 a A7 e A9 a A15 com detector fake e imagens sintéticas (0, 1 e vários rostos, rosto pequeno, mais de 20 rostos, EXIF).
- **Integração** (`@pytest.mark.integration`, issue #47): A2 e A4 com o MTCNN real em imagens públicas do LFW (uma paisagem e uma montagem com dois rostos), e A9 com a mesma montagem.
