// The number of enrolled people shown in the menu. A view that already fetched the list calls
// setCount; one that changes the list (enrollment, deletion) calls refreshCount.

import { api } from "./api.js";
import { h } from "./dom.js";

export const countBadge = h("span", { class: "menu__count", hidden: true });

export function setCount(count) {
  countBadge.textContent = String(count);
  countBadge.hidden = false;
}

export async function refreshCount() {
  const result = await api.listPersons();
  if (result.ok) setCount(result.data.length);
}
