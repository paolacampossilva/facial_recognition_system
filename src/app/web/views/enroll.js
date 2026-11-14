// Enrollment tab: a name and 1 to 10 photos, sent to POST /api/persons.

import { api } from "../lib/api.js";
import { clearDraft, keepDraft, setDraft } from "../lib/drafts.js";
import { h, loading, notice, setBusy } from "../lib/dom.js";
import { describeError } from "../lib/errors.js";
import { photoPicker } from "../lib/photos.js";

const NAME_KEY = "enroll-name";
const PHOTOS_KEY = "enroll-photos";
const MAX_PHOTOS = 10;
const MAX_NAME_LENGTH = 100;

export function enrollView(root) {
  const feedback = h("div", { "aria-live": "polite" });
  const nameInput = keepDraft(NAME_KEY, h("input", {
    id: "person-name", type: "text", maxlength: MAX_NAME_LENGTH, autocomplete: "off",
  }));
  const picker = photoPicker({
    max: MAX_PHOTOS,
    onChange: (count) => setDraft(PHOTOS_KEY, count ? String(count) : ""),
  });
  const submit = h("button", { type: "submit" }, "Cadastrar");

  const form = h("form", { class: "stack", novalidate: true },
    h("h2", {}, "Cadastrar pessoa"),
    h("div", { class: "field" },
      h("label", { for: "person-name" }, "Nome"),
      nameInput),
    h("div", { class: "field" },
      h("span", { class: "label" }, `Fotos (de 1 a ${MAX_PHOTOS})`),
      picker.element),
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
  });

  root.replaceChildren(form);
  return () => picker.dispose();
}
