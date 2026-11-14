// People tab: the enrolled people, with deletion after a confirmation.

import { api } from "../lib/api.js";
import { confirmDialog } from "../lib/dialogs.js";
import { emptyState, h, icon, loading, notice } from "../lib/dom.js";
import { describeError } from "../lib/errors.js";

const photoCount = (count) => `${count} ${count === 1 ? "foto usada" : "fotos usadas"} no cadastro`;

export function personsView(root) {
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
  }

  const row = (person) => h("li", { class: "person" },
    h("div", {}, h("strong", {}, person.name), h("p", { class: "muted" }, photoCount(person.embedding_count))),
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
    } else if (result.data.length === 0) {
      content.replaceChildren(emptyState("Nenhuma pessoa cadastrada.", "Use a aba Cadastro para cadastrar a primeira."));
    } else {
      content.replaceChildren(h("ul", { class: "people" }, result.data.map(row)));
    }
  }

  root.replaceChildren(h("div", { class: "stack" }, h("h2", {}, "Pessoas cadastradas"), feedback, content));
  load();
  return () => {};
}
