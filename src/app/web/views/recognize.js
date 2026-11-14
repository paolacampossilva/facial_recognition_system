// Recognition tab: one photo sent to POST /api/recognitions, shown as recognized or unknown.

import { api } from "../lib/api.js";
import { emptyState, h, loading, notice, setBusy } from "../lib/dom.js";
import { describeError } from "../lib/errors.js";
import { photoPicker } from "../lib/photos.js";

const formatSimilarity = (value) => value.toFixed(3).replace(".", ",");

// An unknown face shows only the similarity, never a name (spec 004).
function resultCard(result) {
  if (result.status === "recognized") {
    return h("div", { class: "result result--ok", role: "status" },
      h("strong", {}, `Pessoa reconhecida: ${result.person_name}`),
      h("p", { class: "muted" }, `Similaridade: ${formatSimilarity(result.similarity)}`));
  }
  return h("div", { class: "result", role: "status" },
    h("strong", {}, "Pessoa desconhecida"),
    h("p", { class: "muted" }, `Similaridade: ${formatSimilarity(result.similarity)}`));
}

export function recognizeView(root) {
  const picker = photoPicker({ max: 1, onChange: (count) => { submit.disabled = count === 0; } });
  const feedback = h("div", { "aria-live": "polite" });
  const submit = h("button", { type: "button", disabled: true }, "Reconhecer");
  const heading = h("h2", {}, "Reconhecer pessoa");

  async function recognize() {
    setBusy(submit, true);
    feedback.replaceChildren(loading("Reconhecendo…"));
    const result = await api.recognize(picker.files()[0]);
    setBusy(submit, false);
    submit.disabled = picker.count() === 0;
    feedback.replaceChildren(result.ok ? resultCard(result.data) : notice("error", describeError(result)));
  }

  async function render() {
    root.replaceChildren(heading, loading());
    const people = await api.listPersons();
    if (!people.ok) {
      const retry = h("button", { type: "button", class: "secondary" }, "Tentar de novo");
      retry.addEventListener("click", render);
      root.replaceChildren(heading, notice("error", describeError(people)), retry);
      return;
    }
    if (people.data.length === 0) {
      root.replaceChildren(heading, emptyState(
        "Nenhuma pessoa cadastrada.", "Cadastre uma pessoa na aba Cadastro para poder reconhecê-la."));
      return;
    }
    root.replaceChildren(h("div", { class: "stack" }, heading, picker.element, feedback,
      h("div", { class: "form-actions" }, submit)));
  }

  submit.addEventListener("click", recognize);
  render();
  return () => picker.dispose();
}
