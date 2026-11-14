// Turns the error body of the API ({code, message}) into the message the person reads.

const MESSAGES = {
  no_face_detected: "Nenhum rosto encontrado na imagem",
  multiple_faces: "Mais de um rosto na imagem",
  invalid_image: "Arquivo de imagem inválido",
  empty_gallery: "Nenhuma pessoa cadastrada.",
  person_not_found: "Pessoa não encontrada. A lista foi atualizada",
  invalid_request: "Confira o nome e as fotos (de 1 a 10)",
  internal_error: "Algo deu errado. Tente de novo",
};

// Problems found in the form before anything is sent, shown next to the field that has them.
export const NAME_REQUIRED = "Digite o nome da pessoa.";
export const PHOTOS_REQUIRED = "Adicione de 1 a 10 fotos.";

export const CAMERA_UNAVAILABLE = "Não foi possível usar a câmera. Envie um arquivo";
export const NO_CONNECTION = "Sem conexão com o servidor. Tente de novo";

export function describeError({ status, data }) {
  if (status === 0) return NO_CONNECTION;
  const code = data?.code;
  if (code === "duplicate_person") {
    return data.reason === "similarity"
      ? `Esse rosto já está cadastrado como ${data.person_name}`
      : "Já existe uma pessoa cadastrada com esse nome";
  }
  return MESSAGES[code] ?? MESSAGES.internal_error;
}
