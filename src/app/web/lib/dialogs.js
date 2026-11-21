// Confirmation dialog on the native <dialog>: showModal() already gives Esc, a focus trap, the
// inert page behind it and the focus going back to the opener, so none of that is rebuilt by hand.

import { h } from "./dom.js";

let dialogCount = 0;

// Resolves to {confirmed}. Esc, a click outside and the cancel button all mean "not confirmed".
// The first focus goes to the cancel button, never to the destructive one.
export function confirmDialog({ title, message, confirmLabel, cancelLabel = "Cancelar", danger = false }) {
  return new Promise((resolve) => {
    const id = `dialog-${++dialogCount}`;
    const cancel = h("button", { type: "button", class: "secondary" }, cancelLabel);
    const confirm = h("button", { type: "button", class: danger ? "danger" : "" }, confirmLabel);
    const dialog = h(
      "dialog", { class: "confirm", role: "alertdialog", "aria-labelledby": `${id}-title`, "aria-describedby": `${id}-message` },
      h(
        "div", { class: "dialog-body" },
        h("h2", { id: `${id}-title` }, title),
        h("p", { id: `${id}-message` }, message),
        h("div", { class: "row dialog-actions" }, cancel, confirm),
      ),
    );
    cancel.addEventListener("click", () => dialog.close("cancel"));
    confirm.addEventListener("click", () => dialog.close("confirm"));
    // A click on the backdrop lands on the <dialog> itself; the content sits in .dialog-body.
    dialog.addEventListener("click", (event) => {
      if (event.target === dialog) dialog.close("cancel");
    });
    dialog.addEventListener("close", () => {
      const confirmed = dialog.returnValue === "confirm";
      dialog.remove();
      resolve({ confirmed });
    });
    document.body.append(dialog);
    dialog.showModal();
    cancel.focus();
  });
}
