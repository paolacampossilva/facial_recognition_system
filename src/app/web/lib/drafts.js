// What the person has filled in and not sent yet, kept in memory (the content policy bans browser
// storage). Any way out of the screen asks before throwing it away.

import { confirmDialog } from "./dialogs.js";

const drafts = new Map();

// Binds a text field to a draft: restores its text if one exists and records every change.
export function keepDraft(key, control) {
  control.value = drafts.get(key) ?? control.value;
  control.addEventListener("input", () => setDraft(key, control.value.trim()));
  return control;
}

// Marks a piece of work as unsaved (a non-empty value) or saved (an empty one).
export function setDraft(key, value) {
  if (value) drafts.set(key, value);
  else drafts.delete(key);
}

export const clearDraft = (...keys) => keys.forEach((key) => drafts.delete(key));

export const hasUnsavedWork = () => drafts.size > 0;

// Asks before leaving with unsaved work. On "Sair sem salvar" the drafts are dropped, so the same
// content is not asked about twice.
export async function confirmLeave() {
  if (!hasUnsavedWork()) return true;
  const { confirmed } = await confirmDialog({
    title: "Sair sem salvar?",
    message: "Você preencheu um formulário e não salvou. Se sair agora, essas alterações serão perdidas.",
    confirmLabel: "Sair sem salvar",
    cancelLabel: "Continuar editando",
    danger: true,
  });
  if (confirmed) drafts.clear();
  return confirmed;
}

// Closing the tab or reloading: the browser shows its own generic question.
window.addEventListener("beforeunload", (event) => {
  if (!hasUnsavedWork()) return;
  event.preventDefault();
  event.returnValue = "";
});
