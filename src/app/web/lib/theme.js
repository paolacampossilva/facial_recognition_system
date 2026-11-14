// Light and dark theme. The choice is saved under the single key "theme"; without a choice the
// system preference applies (in CSS). theme-init.js applies the saved choice before the first paint.

const KEY = "theme";
const LABEL = { light: "Modo escuro", dark: "Modo claro" };

export const nextTheme = (effective) => (effective === "dark" ? "light" : "dark");
export const labelFor = (effective) => LABEL[effective];

export function savedTheme() {
  try {
    const value = localStorage.getItem(KEY);
    if (value === "light" || value === "dark") return value;
  } catch {
    // Without storage there is no saved choice.
  }
  return null;
}

const systemIsDark = () => window.matchMedia("(prefers-color-scheme: dark)").matches;
export const effectiveTheme = (saved, dark) => saved ?? (dark ? "dark" : "light");

export function applyTheme(saved) {
  if (saved === null) document.documentElement.removeAttribute("data-theme");
  else document.documentElement.setAttribute("data-theme", saved);
}

let saved = null;
let started = false;
const buttons = new Set();

function paint(button) {
  const effective = effectiveTheme(saved, systemIsDark());
  const text = labelFor(effective);
  button.setAttribute("aria-label", text);
  button.dataset.goes = nextTheme(effective);
  button.querySelector(".theme-btn__text").textContent = text;
}

function repaintAll() {
  for (const bound of buttons) paint(bound);
}

export function initTheme(button) {
  if (!started) {
    started = true;
    saved = savedTheme();
    applyTheme(saved);
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", repaintAll);
  }
  button.replaceChildren(
    Object.assign(document.createElement("span"), { className: "theme-btn__icon" }),
    Object.assign(document.createElement("span"), { className: "theme-btn__text" }),
  );
  button.firstChild.setAttribute("aria-hidden", "true");
  buttons.add(button);
  paint(button);
  button.addEventListener("click", () => {
    saved = nextTheme(effectiveTheme(saved, systemIsDark()));
    applyTheme(saved);
    repaintAll();
    try {
      localStorage.setItem(KEY, saved);
    } catch {
      // The choice lasts until the page is reloaded.
    }
  });
}
