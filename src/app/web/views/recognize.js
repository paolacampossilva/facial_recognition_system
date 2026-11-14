// Recognition screen: one photo sent to POST /api/recognitions, with the result in a panel beside it.

import { api } from "../lib/api.js";
import { setCount } from "../lib/counter.js";
import { avatar, card, emptyState, h, icon, linkButton, loading, notice, pageHead, setBusy } from "../lib/dom.js";
import { describeError } from "../lib/errors.js";
import { formatSimilarity, meterState } from "../lib/meter.js";
import { photoPicker } from "../lib/photos.js";


const waiting = () => h("div", { class: "placeholder" },
  h("span", { class: "icon icon--xl icon--scan-face", "aria-hidden": "true" }),
  h("strong", {}, "O resultado aparece aqui"),
  h("p", { class: "muted" }, "Escolha uma foto e clique em Reconhecer."));

const formatPercent = (value) => `${(Math.max(0, value) * 100).toFixed(1).replace(".", ",")}%`;

// The CSP forbids style attributes in the markup, so the bar is sized through CSS variables set here.
function bar(state) {
  const track = h("div", {
    class: `meter__track meter__track--${state.tone}`,
    role: "img",
    "aria-label": "Similaridade e limite de reconhecimento",
  }, h("div", { class: "meter__fill" }));
  track.style.setProperty("--value", String(state.fill));
  if (state.mark !== null) {
    const mark = h("div", { class: "meter__mark" });
    mark.style.setProperty("--limit", String(state.mark));
    track.append(mark);
  }
  return track;
}

// The similarity as a big percentage, with the bar, the limit and the original cosine value in small print.
// Above or below the limit is also written out, so the color is not the only signal.
function similarityMeter(value, threshold, aboveLimit) {
  const state = meterState(value, threshold, aboveLimit);
  return h("div", { class: "meter" },
    h("div", { class: "meter__head" },
      h("strong", { class: "meter__percent" }, formatPercent(value)),
      h("span", { class: "muted" }, "de similaridade")),
    bar(state),
    state.limitText === null ? null : h("p", { class: "meter__limit" }, state.limitText),
    h("p", { class: "muted meter__raw" }, `Valor original: ${formatSimilarity(value)}`));
}

// An unknown face shows only the similarity, never a name (spec 004).
function resultPanel(result, threshold) {
  if (result.status === "recognized") {
    return h("div", { class: "result result--ok", role: "status" },
      h("div", { class: "result__who" },
        avatar(result.person_name, true),
        h("div", {}, h("p", { class: "muted" }, "Pessoa reconhecida"),
          h("strong", { class: "result__name" }, result.person_name))),
      similarityMeter(result.similarity, threshold, true));
  }
  return h("div", { class: "result", role: "status" },
    h("div", { class: "result__who" },
      h("span", { class: "avatar avatar--lg avatar--unknown", "aria-hidden": "true" }, "?"),
      h("div", {}, h("p", { class: "muted" }, "Nenhuma pessoa cadastrada combina com este rosto"),
        h("strong", { class: "result__name" }, "Pessoa desconhecida"))),
    similarityMeter(result.similarity, threshold, false));
}

export function recognizeView(root) {
  // The limit marks the bar; without it (the call failed) the bar is shown with no mark.
  let threshold = null;
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
    output.replaceChildren(result.ok ? resultPanel(result.data, threshold) : notice("error", describeError(result)));
  }

  async function render() {
    root.replaceChildren(h("div", { class: "page" }, heading, loading()));
    const [people, config] = await Promise.all([api.listPersons(), api.config()]);
    threshold = config.ok ? config.data.threshold : null;
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
