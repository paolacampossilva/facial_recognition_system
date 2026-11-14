// People screen: the enrolled people as cards, with deletion after a confirmation.

import { api } from "../lib/api.js";
import { refreshCount, setCount } from "../lib/counter.js";
import { confirmDialog } from "../lib/dialogs.js";
import { avatar, card, emptyState, h, icon, linkButton, loading, notice } from "../lib/dom.js";
import { describeError } from "../lib/errors.js";

const photoCount = (count) => `${count} ${count === 1 ? "foto usada" : "fotos usadas"}`;
const peopleCount = (count) => `${count} ${count === 1 ? "pessoa cadastrada" : "pessoas cadastradas"}`;

export function personsView(root) {
  const summary = h("p", { class: "muted" }, "Quem pode ser reconhecido neste computador.");
  const feedback = h("div", { "aria-live": "polite" });
  const content = h("div", {});

  async function remove(person) {
    const { confirmed } = await confirmDialog({
      title: `Excluir ${person.name}?`,
      message: "O cadastro e os dados faciais desta pessoa serão apagados. Isso não pode ser desfeito.",
      confirmLabel: `Excluir ${person.name}`,
      danger: true,
    });
    if (!confirmed) return;
    const result = await api.deletePerson(person.id);
    feedback.replaceChildren(result.ok
      ? notice("success", `${person.name} foi excluída(o).`)
      : notice("error", describeError(result)));
    await load();
    refreshCount();
  }

  const item = (person) => h("li", { class: "person" },
    avatar(person.name),
    h("div", { class: "person__text" },
      h("strong", {}, person.name), h("p", { class: "muted" }, photoCount(person.embedding_count))),
    h("button", {
      type: "button", class: "danger", "aria-label": `Excluir ${person.name}`, onclick: () => remove(person),
    }, icon("trash"), "Excluir"));

  async function load() {
    content.replaceChildren(loading());
    const result = await api.listPersons();
    if (!result.ok) {
      const retry = h("button", { type: "button", class: "secondary" }, "Tentar de novo");
      retry.addEventListener("click", load);
      content.replaceChildren(notice("error", describeError(result)), retry);
      return;
    }
    setCount(result.data.length);
    if (result.data.length === 0) {
      summary.textContent = "Ninguém foi cadastrado ainda.";
      content.replaceChildren(card(null, null, emptyState(
        "Nenhuma pessoa cadastrada.",
        "Cadastre a primeira pessoa para começar.",
        linkButton("#/cadastro", "Cadastrar a primeira pessoa"))));
      return;
    }
    summary.textContent = peopleCount(result.data.length);
    content.replaceChildren(h("ul", { class: "people" }, result.data.map(item)));
  }

  root.replaceChildren(h("div", { class: "page" },
    h("header", { class: "page-head" }, h("h1", {}, "Pessoas cadastradas"), summary),
    feedback, content));
  load();
  return () => {};
}
