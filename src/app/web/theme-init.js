// Classic script in the <head>: applies the saved theme before the first paint, so a saved
// "light" choice on a dark system does not flash. The content policy allows it (script-src 'self').
try {
  const saved = localStorage.getItem("theme");
  if (saved === "light" || saved === "dark") {
    document.documentElement.setAttribute("data-theme", saved);
  }
} catch {
  // Without storage there is no saved choice; the system preference applies.
}
