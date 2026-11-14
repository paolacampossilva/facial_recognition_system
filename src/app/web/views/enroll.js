// Enrollment screen: a name and 1 to 10 photos, sent to POST /api/persons.

import { api } from "../lib/api.js";
import { refreshCount } from "../lib/counter.js";
import { clearDraft, keepDraft, setDraft } from "../lib/drafts.js";
import { card, h, icon, loading, notice, pageHead, setBusy } from "../lib/dom.js";
import { describeError } from "../lib/errors.js";
import { photoPicker } from "../lib/photos.js";

const NAME_KEY = "enroll-name";
const PHOTOS_KEY = "enroll-photos";
const MAX_PHOTOS = 10;
const MAX_NAME_LENGTH = 100;

const STEPS = [
  "Digite o nome da pessoa.",
  `Adicione de 1 a ${MAX_PHOTOS} fotos do rosto, com ângulos e luz diferentes.`,
  "Clique em Cadastrar.",
];

export function enrollView(root) {
  const feedback = h("div", { "aria-live": "polite" });
  const counter = h("p", { class: "muted", "aria-live": "polite" }, `0 de ${MAX_PHOTOS} fotos`);
  const nameInput = keepDraft(NAME_KEY, h("input", {
    id: "person-name", type: "text", maxlength: MAX_NAME_LENGTH, autocomplete: "off",
  }));
  const picker = photoPicker({
    max: MAX_PHOTOS,
    onChange: (count) => {
      counter.textContent = `${count} de ${MAX_PHOTOS} fotos`;
      setDraft(PHOTOS_KEY, count ? String(count) : "");
    },
  });
  const submit = h("button", { type: "submit" }, "Cadastrar");

  const form = h("form", { class: "page", novalidate: true },
    pageHead("Cadastro", "Cadastre uma pessoa para que ela possa ser reconhecida."),
    h("div", { class: "grid" },
      card("Dados da pessoa", null,
        h("div", { class: "field" },
          h("label", { for: "person-name" }, "Nome"),
          nameInput,
          h("p", { class: "muted" }, "É o nome que aparece quando a pessoa é reconhecida.")),
        h("ol", { class: "steps" }, STEPS.map((step) => h("li", {}, step))),
        h("p", { class: "privacy" }, icon("lock"),
          "As fotos não são guardadas, só os dados faciais usados no reconhecimento.")),
      card("Fotos", "Pela câmera ou por arquivo.", picker.element, counter)),
    feedback,
    h("div", { class: "form-actions" }, submit));

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const name = nameInput.value.trim();
    if (!name || picker.count() === 0) {
      feedback.replaceChildren(notice("error", describeError({ status: 422, data: { code: "invalid_request" } })));
      return;
    }
    setBusy(submit, true);
    feedback.replaceChildren(loading("Cadastrando…"));
    const result = await api.enroll(name, picker.files());
    setBusy(submit, false);
    if (!result.ok) {
      feedback.replaceChildren(notice("error", describeError(result)));
      return;
    }
    clearDraft(NAME_KEY, PHOTOS_KEY);
    picker.clear();
    nameInput.value = "";
    feedback.replaceChildren(notice("success", `Cadastro de ${result.data.name} concluído.`));
    refreshCount();
  });

  root.replaceChildren(form);
  return () => picker.dispose();
}
