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

// An empty state with an optional action (a button or link) that leads out of it.
export function emptyState(title, text, action = null) {
  return h("div", { class: "empty-block" },
    h("span", { class: "icon icon--xl icon--users", "aria-hidden": "true" }),
    h("strong", {}, title),
    text && h("p", { class: "muted" }, text),
    action);
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

// The title of a screen and one line that says what it is for.
export const pageHead = (title, text) =>
  h("header", { class: "page-head" }, h("h1", {}, title), text && h("p", { class: "muted" }, text));

// A card: an optional heading with a hint, then the content.
export const card = (title, hint, ...children) =>
  h("section", { class: "card" },
    title && h("div", { class: "card__head" }, h("h2", {}, title), hint && h("p", { class: "muted" }, hint)),
    ...children);

// The first letter of a name in a circle (decorative: the name is always written next to it).
export const avatar = (name, large = false) =>
  h("span", { class: large ? "avatar avatar--lg" : "avatar", "aria-hidden": "true" },
    Array.from(name.trim())[0]?.toUpperCase() ?? "?");

// A link that looks like a button, for moving to another screen.
export const linkButton = (href, label, secondary = false) =>
  h("a", { href, class: secondary ? "button secondary" : "button" }, label);
