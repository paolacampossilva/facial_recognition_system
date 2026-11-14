// The side bar on narrow screens: below 860px it is an off-canvas drawer opened by the menu button.
// Esc, a click on the backdrop and navigating close it; closing by Esc or backdrop returns the focus
// to the button. Off-canvas, the bar is inert, so Tab never reaches an invisible link.

const NARROW = window.matchMedia("(max-width: 860px)");

export function initShell() {
  const toggle = document.getElementById("menu-toggle");
  const scrim = document.getElementById("scrim");
  const sidebar = document.getElementById("sidebar");

  const isOpen = () => document.body.classList.contains("sidebar-open");

  function setOpen(open) {
    document.body.classList.toggle("sidebar-open", open);
    toggle.setAttribute("aria-expanded", String(open));
    toggle.setAttribute("aria-label", open ? "Fechar o menu" : "Abrir o menu");
    scrim.hidden = !open;
    sidebar.inert = NARROW.matches && !open;
  }

  function close({ restoreFocus }) {
    if (!isOpen()) return;
    setOpen(false);
    if (restoreFocus) toggle.focus();
  }

  toggle.addEventListener("click", () => {
    setOpen(!isOpen());
    if (isOpen()) sidebar.querySelector("a")?.focus();
  });
  scrim.addEventListener("click", () => close({ restoreFocus: true }));
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") close({ restoreFocus: true });
  });
  window.addEventListener("hashchange", () => close({ restoreFocus: false }));
  NARROW.addEventListener("change", () => setOpen(false));
  setOpen(false);
}
