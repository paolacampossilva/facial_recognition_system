// Shell: three screens in the side menu with hash routing, the theme button and the
// "leave without saving" guard.

import { countBadge, refreshCount } from "./lib/counter.js";
import { confirmLeave } from "./lib/drafts.js";
import { h, icon, notice } from "./lib/dom.js";
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

// The links are built once and only marked, so the focus stays on the one the person used.
const menuLinks = ROUTES.map((route) =>
  h("a", { href: `#/${route.path}` }, icon(route.icon), route.label, route.path === "pessoas" && countBadge));
menu.replaceChildren(...menuLinks);

function paintMenu(active) {
  ROUTES.forEach((route, index) => {
    if (route === active) menuLinks[index].setAttribute("aria-current", "page");
    else menuLinks[index].removeAttribute("aria-current");
  });
}

// After a change of screen the focus goes to its title, so a screen reader announces where the
// person is and the keyboard continues from there. Not on the first load: nothing was asked yet.
let firstScreen = true;

function focusTitle() {
  const title = root.querySelector("h1");
  if (title && !firstScreen) {
    title.tabIndex = -1;
    title.focus({ preventScroll: true });
  }
  firstScreen = false;
}

async function go(next) {
  if (current?.route === next) return;
  if (current && !(await confirmLeave())) {
    location.hash = `#/${current.route.path}`;
    return;
  }
  current?.dispose?.();
  current = null;
  document.title = `${next.label} · ${PRODUCT}`;
  paintMenu(next);
  root.replaceChildren();
  try {
    current = { route: next, dispose: next.view(root) };
    focusTitle();
  } catch {
    // A screen that fails to build must not leave a blank page that blocks every later visit.
    root.replaceChildren(h("div", { class: "page" }, notice("error", "Algo deu errado. Tente de novo")));
  }
}

// One navigation at a time: the browser's Back and Forward do not wait for the "leave without
// saving" dialog, so a second change that arrives meanwhile is handled right after the first.
let navigating = false;
let queued = false;

async function show() {
  if (navigating) {
    queued = true;
    return;
  }
  navigating = true;
  try {
    do {
      queued = false;
      await go(routeFromHash());
    } while (queued);
  } finally {
    navigating = false;
  }
}

initTheme(document.getElementById("theme"));
initShell();
window.addEventListener("hashchange", show);
refreshCount();
show();
