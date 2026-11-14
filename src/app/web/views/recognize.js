// Recognition screen: one photo sent to POST /api/recognitions, with the result in a panel beside it.

import { api } from "../lib/api.js";
import { setCount } from "../lib/counter.js";
import { avatar, card, emptyState, h, icon, linkButton, loading, notice, pageHead, setBusy } from "../lib/dom.js";
import { describeError } from "../lib/errors.js";
import { photoPicker } from "../lib/photos.js";

const formatSimilarity = (value) => value.toFixed(3).replace(".", ",");

const waiting = () => h("div", { class: "placeholder" },
  h("span", { class: "icon icon--xl icon--scan-face", "aria-hidden": "true" }),
  h("strong", {}, "O resultado aparece aqui"),
  h("p", { class: "muted" }, "Escolha uma foto e clique em Reconhecer."));

const formatPercent = (value) => `${(Math.max(0, value) * 100).toFixed(1).replace(".", ",")}%`;

// The similarity as a big percentage, with the bar and the original cosine value in small print.
const similarityMeter = (value) => h("div", { class: "meter" },
  h("div", { class: "meter__head" },
    h("strong", { class: "meter__percent" }, formatPercent(value)),
    h("span", { class: "muted" }, "de similaridade")),
  h("progress", { max: 1, value: Math.max(0, value), "aria-label": "Similaridade" }),
  h("p", { class: "muted meter__raw" }, `Valor original: ${formatSimilarity(value)}`));

// An unknown face shows only the similarity, never a name (spec 004).
function resultPanel(result) {
  if (result.status === "recognized") {
    return h("div", { class: "result result--ok", role: "status" },
      h("div", { class: "result__who" },
        avatar(result.person_name, true),
        h("div", {}, h("p", { class: "muted" }, "Pessoa reconhecida"),
          h("strong", { class: "result__name" }, result.person_name))),
      similarityMeter(result.similarity));
  }
  return h("div", { class: "result", role: "status" },
    h("div", { class: "result__who" },
      h("span", { class: "avatar avatar--lg avatar--unknown", "aria-hidden": "true" }, "?"),
      h("div", {}, h("p", { class: "muted" }, "Nenhuma pessoa cadastrada combina com este rosto"),
        h("strong", { class: "result__name" }, "Pessoa desconhecida"))),
    similarityMeter(result.similarity));
}

export function recognizeView(root) {
  const heading = pageHead("Reconhecimento", "Envie uma foto para saber quem é a pessoa.");
  const output = h("div", { "aria-live": "polite" }, waiting());
  const submit = h("button", { type: "button", disabled: true }, "Reconhecer");
  // A new or removed photo makes the old result stale.
  const picker = photoPicker({
    max: 1,
    onChange: (count) => {
      submit.disabled = count === 0;
      output.replaceChildren(waiting());
    },
  });

  async function recognize() {
    setBusy(submit, true);
    output.replaceChildren(loading("Reconhecendo…"));
    const result = await api.recognize(picker.files()[0]);
    setBusy(submit, false);
    submit.disabled = picker.count() === 0;
    output.replaceChildren(result.ok ? resultPanel(result.data) : notice("error", describeError(result)));
  }

  async function render() {
    root.replaceChildren(h("div", { class: "page" }, heading, loading()));
    const people = await api.listPersons();
    if (!people.ok) {
      const retry = h("button", { type: "button", class: "secondary" }, "Tentar de novo");
      retry.addEventListener("click", render);
      root.replaceChildren(h("div", { class: "page" }, heading, notice("error", describeError(people)), retry));
      return;
    }
    setCount(people.data.length);
    if (people.data.length === 0) {
      root.replaceChildren(h("div", { class: "page" }, heading, card(null, null, emptyState(
        "Nenhuma pessoa cadastrada.",
        "Cadastre pelo menos uma pessoa para poder reconhecê-la.",
        linkButton("#/cadastro", "Ir para o Cadastro")))));
      return;
    }
    root.replaceChildren(h("div", { class: "page" }, heading,
      h("div", { class: "grid" },
        card("Foto", "Pela câmera ou por arquivo. Use uma foto de frente, com um só rosto e boa luz.", picker.element,
          h("div", { class: "form-actions" }, submit)),
        card("Resultado", null, output))));
  }

  submit.addEventListener("click", recognize);
  render();
  return () => picker.dispose();
}
