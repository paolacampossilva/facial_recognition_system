// Recognition screen: one photo sent to POST /api/recognitions, with the photo and one result per
// face in a panel beside it.

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

// One face of the photo. An unknown face shows only the similarity, never a name (spec 004).
function faceResult(face, number, threshold) {
  const known = face.status === "recognized";
  const who = known
    ? [avatar(face.person_name, true),
      h("div", {}, h("p", { class: "muted" }, "Pessoa reconhecida"),
        h("strong", { class: "result__name" }, face.person_name))]
    : [h("span", { class: "avatar avatar--lg avatar--unknown", "aria-hidden": "true" }, "?"),
      h("div", {}, h("p", { class: "muted" }, "Nenhuma pessoa cadastrada combina com este rosto"),
        h("strong", { class: "result__name" }, "Pessoa desconhecida"))];
  return h("li", { class: known ? "result result--ok" : "result" },
    h("p", { class: "result__label" }, `Rosto ${number}`),
    h("div", { class: "result__who" }, ...who),
    similarityMeter(face.similarity, threshold, known));
}

// A box over the photo, placed as a fraction of the image (the CSP forbids style attributes in the
// markup, so the fractions go in CSS variables). The API keeps every box inside the image, so the
// fractions stay between 0 and 1. The number is written out as well as colored.
function faceBox(face, number, imageWidth, imageHeight) {
  const label = face.status === "recognized"
    ? `Rosto ${number}: ${face.person_name}`
    : `Rosto ${number}: pessoa desconhecida`;
  const box = h("div", { class: `face-box face-box--${face.status}`, role: "img", "aria-label": label },
    h("span", { class: "face-box__tag", "aria-hidden": "true" }, String(number)));
  const { left, top, right, bottom } = face.box;
  box.style.setProperty("--left", String(left / imageWidth));
  box.style.setProperty("--top", String(top / imageHeight));
  box.style.setProperty("--width", String((right - left) / imageWidth));
  box.style.setProperty("--height", String((bottom - top) / imageHeight));
  return box;
}

// "Ana aparece em 2 rostos desta foto": only a count of what the API returned, the API decides each face.
function repeatedPeople(faces) {
  const counts = new Map();
  for (const face of faces) {
    if (face.person_name !== null) counts.set(face.person_name, (counts.get(face.person_name) ?? 0) + 1);
  }
  return [...counts]
    .filter(([, count]) => count > 1)
    .map(([name, count]) => notice("info", `${name} aparece em ${count} rostos desta foto`));
}

// The photo with a box on every face, then one result per face in the same order, left to right.
function resultPanel(data, photoUrl, threshold) {
  const image = h("img", { src: photoUrl, alt: "Foto enviada para o reconhecimento" });
  const layer = h("div", { class: "faces__layer" },
    data.faces.map((face, index) => faceBox(face, index + 1, data.image_width, data.image_height)));
  // The boxes are fractions of the image the server read. If the browser shows the photo at
  // another size (a different orientation, or it cannot show it at all), a box would sit on the
  // wrong face, so the layer is taken away and the list below still says everything.
  const dropLayer = () => layer.remove();
  image.addEventListener("error", dropLayer);
  image.addEventListener("load", () => {
    const sameSize = Math.abs(image.naturalWidth - data.image_width) <= 1
      && Math.abs(image.naturalHeight - data.image_height) <= 1;
    if (!sameSize) dropLayer();
  });
  const photo = h("figure", { class: "faces__photo" }, image, layer);
  return h("div", { class: "faces", role: "status" },
    photo,
    h("ul", { class: "faces__list" },
      data.faces.map((face, index) => faceResult(face, index + 1, threshold))),
    ...repeatedPeople(data.faces));
}

export function recognizeView(root) {
  // The limit marks the bar; without it (the call failed) the bar is shown with no mark.
  let threshold = null;
  // `alive` goes false when the person leaves the screen; `latest` numbers the recognitions so that
  // an answer that arrives after the photo changed (or after leaving) is dropped.
  let alive = true;
  let latest = 0;
  // The address of the photo shown with its boxes; it is released when the result goes away.
  let photoUrl = null;
  function releasePhoto() {
    if (photoUrl !== null) URL.revokeObjectURL(photoUrl);
    photoUrl = null;
  }
  const heading = pageHead("Reconhecimento", "Envie uma foto para saber quem é a pessoa.");
  // The page is built once and only what follows the title is replaced, so the title keeps the
  // focus that the menu gives it when the screen opens.
  const page = h("div", { class: "page" }, heading);
  function paint(...nodes) {
    while (heading.nextSibling) heading.nextSibling.remove();
    heading.after(...nodes);
  }
  const output = h("div", { "aria-live": "polite" }, waiting());
  const submit = h("button", { type: "button", disabled: true }, "Reconhecer");
  // A new or removed photo makes the old result stale.
  const picker = photoPicker({
    max: 1,
    onChange: (count) => {
      latest += 1;
      releasePhoto();
      submit.disabled = count === 0;
      output.replaceChildren(waiting());
    },
  });

  async function recognize() {
    const mine = ++latest;
    releasePhoto();
    setBusy(submit, true);
    output.replaceChildren(loading("Reconhecendo…"));
    const photo = picker.files()[0];
    const result = await api.recognize(photo);
    if (!alive) return;
    setBusy(submit, false);
    submit.disabled = picker.count() === 0;
    if (mine !== latest) return; // the photo was replaced or removed meanwhile: the result is stale
    if (!result.ok) {
      output.replaceChildren(notice("error", describeError(result)));
      return;
    }
    photoUrl = URL.createObjectURL(photo.blob);
    output.replaceChildren(resultPanel(result.data, photoUrl, threshold));
  }

  async function render() {
    paint(loading());
    const [people, config] = await Promise.all([api.listPersons(), api.config()]);
    if (!alive) return; // the person went to another screen, which owns the page now
    threshold = config.ok ? config.data.threshold : null;
    if (!people.ok) {
      const retry = h("button", { type: "button", class: "secondary" }, "Tentar de novo");
      retry.addEventListener("click", render);
      paint(notice("error", describeError(people)), retry);
      return;
    }
    setCount(people.data.length);
    if (people.data.length === 0) {
      paint(card(null, null, emptyState(
        "Nenhuma pessoa cadastrada.",
        "Cadastre pelo menos uma pessoa para poder reconhecê-la.",
        linkButton("#/cadastro", "Ir para o Cadastro"))));
      return;
    }
    paint(h("div", { class: "grid" },
      card("Foto", "Pela câmera ou por arquivo. A foto pode ter vários rostos, de frente, grandes e bem iluminados.",
        picker.element,
        h("p", { class: "privacy" }, icon("lock"),
          "Todo rosto da foto é comparado com as pessoas cadastradas. Use só fotos de quem concordou."),
        h("div", { class: "form-actions" }, submit)),
      card("Resultado", null, output)));
  }

  submit.addEventListener("click", recognize);
  root.replaceChildren(page);
  render();
  return () => {
    alive = false;
    releasePhoto();
    picker.dispose();
  };
}
