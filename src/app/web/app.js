// Shell: three screens in the side menu with hash routing, the theme button and the
// "leave without saving" guard.

import { countBadge, refreshCount } from "./lib/counter.js";
import { confirmLeave } from "./lib/drafts.js";
import { h, icon } from "./lib/dom.js";
import { initShell } from "./lib/shell.js";
import { initTheme } from "./lib/theme.js";
import { enrollView } from "./views/enroll.js";
import { personsView } from "./views/persons.js";
import { recognizeView } from "./views/recognize.js";

const PRODUCT = "Reconhecimento facial";

// Each view renders into the root and returns a function that releases what it holds (the camera).
const ROUTES = [
  { path: "cadastro", label: "Cadastro", icon: "user-check", view: enrollView },
  { path: "reconhecimento", label: "Reconhecimento", icon: "scan-face", view: recognizeView },
  { path: "pessoas", label: "Pessoas cadastradas", icon: "users", view: personsView },
];

const menu = document.getElementById("menu");
const root = document.getElementById("view");
let current = null;

const routeFromHash = () => {
  const name = location.hash.replace(/^#\//, "");
  return ROUTES.find((route) => route.path === name) ?? ROUTES[0];
};

function paintMenu(active) {
  menu.replaceChildren(...ROUTES.map((route) =>
    h("a", { href: `#/${route.path}`, "aria-current": route === active ? "page" : null },
      icon(route.icon), route.label, route.path === "pessoas" && countBadge)));
}

async function show() {
  const next = routeFromHash();
  if (current?.route === next) return;
  if (current && !(await confirmLeave())) {
    location.hash = `#/${current.route.path}`;
    return;
  }
  current?.dispose?.();
  current = { route: next, dispose: null };
  document.title = `${next.label} · ${PRODUCT}`;
  paintMenu(next);
  root.replaceChildren();
  current.dispose = next.view(root);
}

initTheme(document.getElementById("theme"));
initShell();
window.addEventListener("hashchange", show);
refreshCount();
show();
