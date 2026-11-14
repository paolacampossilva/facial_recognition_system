// DOM helpers: the page is built only with nodes and textContent, never with markup strings.

export function h(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [name, value] of Object.entries(attrs)) {
    if (name === "class") node.className = value;
    else if (name.startsWith("on")) node.addEventListener(name.slice(2), value);
    else if (value === true) node.setAttribute(name, "");
    else if (value !== false && value != null) node.setAttribute(name, value);
  }
  for (const child of children.flat()) {
    if (child == null || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

const NOTICE_GLYPH = { error: "✕", warn: "!", success: "✓", info: "i" };

// A message box. Errors and warnings are announced at once (role alert), the rest politely.
export function notice(kind, text) {
  const role = kind === "error" || kind === "warn" ? "alert" : "status";
  return h("div", { class: `notice notice-${kind}`, role },
    h("span", { class: "notice-icon", "aria-hidden": "true" }, NOTICE_GLYPH[kind]),
    h("span", {}, text));
}

export function emptyState(title, text) {
  return h("div", { class: "empty-block" }, h("strong", {}, title), text && h("p", { class: "muted" }, text));
}

export const loading = (text = "Carregando…") =>
  h("p", { class: "muted", role: "status" }, h("span", { class: "spinner", "aria-hidden": "true" }), text);

export const icon = (name) => h("span", { class: `icon icon--${name}`, "aria-hidden": "true" });

// A submit button while its call runs: disabled with a spinner, back to normal in the finally.
export function setBusy(button, busy) {
  button.disabled = busy;
  button.querySelector(".spinner")?.remove();
  if (busy) button.prepend(h("span", { class: "spinner", "aria-hidden": "true" }));
}
