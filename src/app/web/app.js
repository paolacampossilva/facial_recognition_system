// Shell: three tabs with hash routing, the theme button and the "leave without saving" guard.

import { confirmLeave } from "./lib/drafts.js";
import { h, icon } from "./lib/dom.js";
import { initTheme } from "./lib/theme.js";
import { enrollView } from "./views/enroll.js";
import { personsView } from "./views/persons.js";
import { recognizeView } from "./views/recognize.js";

// Each view renders into the root and returns a function that releases what it holds (the camera).
const ROUTES = [
  { path: "cadastro", label: "Cadastro", icon: "user-check", view: enrollView },
  { path: "reconhecimento", label: "Reconhecimento", icon: "scan-face", view: recognizeView },
  { path: "pessoas", label: "Pessoas cadastradas", icon: "users", view: personsView },
];

const tabs = document.getElementById("tabs");
const root = document.getElementById("view");
let current = null;

const routeFromHash = () => {
  const name = location.hash.replace(/^#\//, "");
  return ROUTES.find((route) => route.path === name) ?? ROUTES[0];
};

function paintTabs(active) {
  tabs.replaceChildren(...ROUTES.map((route) =>
    h("a", { href: `#/${route.path}`, "aria-current": route === active ? "page" : null },
      icon(route.icon), route.label)));
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
  paintTabs(next);
  root.replaceChildren();
  current.dispose = next.view(root);
}

initTheme(document.getElementById("theme"));
window.addEventListener("hashchange", show);
show();
