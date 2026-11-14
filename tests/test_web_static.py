"""Checks on the front-end files: security rules, tokens, the violet accent and its routes."""

import colorsys
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from app.api.app import WEB_DIR, create_app
from app.fakes import FakeServices

pytestmark = pytest.mark.unit

THEME_FILES: frozenset[str] = frozenset({"theme.js", "theme-init.js"})
FORBIDDEN_CALLS: tuple[str, ...] = (
    "innerHTML",
    "outerHTML",
    "insertAdjacentHTML",
    "document.write",
    "eval(",
    "new Function",
    "sessionStorage",
    "document.cookie",
)
LOCAL_ALIASES: frozenset[str] = frozenset({"--icon", "--notice-color"})
MIN_CONTRAST: float = 4.5
VIOLET_HUE_RANGE: tuple[float, float] = (250.0, 290.0)


def scripts() -> list[Path]:
    return sorted(WEB_DIR.rglob("*.js"))


def code_of(path: Path) -> str:
    """The source without whole-line comments, so notes do not trip the checks."""
    lines = path.read_text().splitlines()
    return "\n".join(line for line in lines if not line.lstrip().startswith("//"))


def theme_block(css: str, selector: str) -> dict[str, str]:
    """The custom properties declared in the first rule that starts with `selector`."""
    start = css.index(selector + " {")
    body = css[css.index("{", start) + 1 : css.index("}", start)]
    return dict(re.findall(r"(--[\w-]+):\s*([^;]+);", body))


def hex_to_rgb(value: str) -> tuple[float, float, float]:
    digits = value.strip().lstrip("#")
    red, green, blue = (int(digits[index : index + 2], 16) / 255 for index in (0, 2, 4))
    return red, green, blue


def luminance(value: str) -> float:
    def linear(channel: float) -> float:
        return channel / 12.92 if channel <= 0.03928 else ((channel + 0.055) / 1.055) ** 2.4

    red, green, blue = (linear(channel) for channel in hex_to_rgb(value))
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def contrast(first: str, second: str) -> float:
    lighter, darker = sorted((luminance(first), luminance(second)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


def hue_degrees(value: str) -> float:
    return colorsys.rgb_to_hls(*hex_to_rgb(value))[0] * 360


def test_no_script_injects_markup_or_uses_forbidden_calls() -> None:
    for script in scripts():
        code = code_of(script)
        for forbidden in FORBIDDEN_CALLS:
            assert forbidden not in code, f"{script.name} uses {forbidden}"


def test_local_storage_is_only_touched_by_the_theme_files() -> None:
    for script in scripts():
        if script.name not in THEME_FILES:
            assert "localStorage" not in code_of(script), script.name


def test_the_theme_is_saved_under_the_single_key_theme() -> None:
    for name in THEME_FILES:
        source = next(script for script in scripts() if script.name == name).read_text()
        assert 'getItem("theme")' in source or 'const KEY = "theme"' in source


def test_no_browser_dialogs_and_the_confirmation_is_a_native_dialog() -> None:
    for script in scripts():
        assert not re.search(r"\b(alert|confirm|prompt)\(", code_of(script)), script.name
    dialogs = (WEB_DIR / "lib" / "dialogs.js").read_text()
    assert "showModal()" in dialogs
    assert 'role: "alertdialog"' in dialogs


def test_scripts_import_only_from_the_same_folder_tree() -> None:
    for script in scripts():
        source = code_of(script)
        assert "import(" not in source, script.name
        for target in re.findall(r'from "([^"]+)"', source):
            assert target.startswith("./") or target.startswith("../"), f"{script.name}: {target}"


def test_no_script_sets_inline_styles_or_loads_external_urls() -> None:
    for script in scripts():
        code = code_of(script)
        assert "style:" not in code and 'setAttribute("style"' not in code, script.name
        assert not re.search(r"https?://", code), script.name


def test_the_page_loads_no_external_resource() -> None:
    html = (WEB_DIR / "index.html").read_text()
    assert not re.search(r'(?:src|href)="https?:', html)
    assert html.index("tokens.css") < html.index("styles.css")
    assert 'src="theme-init.js"' in html


def test_colors_come_only_from_tokens() -> None:
    styles = (WEB_DIR / "styles.css").read_text()
    assert not re.search(r"#[0-9a-fA-F]{3,8}\b", styles)
    assert not re.search(r"\brgba?\(", styles)
    # Only local aliases (the icon of a button, the color of a notice) may be declared here.
    declared = set(re.findall(r"(?:^|[{;])\s*(--[\w-]+)\s*:", styles, flags=re.MULTILINE))
    assert declared <= LOCAL_ALIASES, declared - LOCAL_ALIASES
    for script in scripts():
        assert not re.search(r"#[0-9a-fA-F]{6}\b", code_of(script)), script.name


def test_dark_by_system_and_dark_by_choice_are_identical() -> None:
    css = (WEB_DIR / "tokens.css").read_text()
    by_system = theme_block(css, ':root:not([data-theme="light"])')
    by_choice = theme_block(css, ':root[data-theme="dark"]')
    assert by_system == by_choice
    assert set(by_choice) <= set(theme_block(css, ":root"))


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_the_accent_is_violet_with_readable_contrast(theme: str) -> None:
    css = (WEB_DIR / "tokens.css").read_text()
    tokens = theme_block(css, ":root" if theme == "light" else ':root[data-theme="dark"]')
    low, high = VIOLET_HUE_RANGE
    assert low <= hue_degrees(tokens["--accent"]) <= high
    assert contrast(tokens["--accent"], tokens["--bg"]) >= MIN_CONTRAST
    assert contrast(tokens["--on-accent"], tokens["--accent"]) >= MIN_CONTRAST
    assert contrast(tokens["--accent-ink"], tokens["--accent-soft"]) >= MIN_CONTRAST


def test_danger_is_not_violet() -> None:
    css = (WEB_DIR / "tokens.css").read_text()
    for selector in (":root", ':root[data-theme="dark"]'):
        low, high = VIOLET_HUE_RANGE
        assert not low <= hue_degrees(theme_block(css, selector)["--danger"]) <= high


def test_every_route_the_front_end_calls_exists_in_the_api() -> None:
    source = code_of(WEB_DIR / "lib" / "api.js")
    called = {
        re.sub(r"\$\{[^}]+\}", "{person_id}", path)
        for path in re.findall(r'call\(\s*[`"](/[^`"]*)[`"]', source)
    }
    api_paths = create_app(FakeServices(), 0.766).openapi()["paths"]
    documented = {path.removeprefix("/api") for path in api_paths}
    assert called
    assert called <= documented


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_theme_module_chooses_labels_and_survives_blocked_storage() -> None:
    script = f"""
    globalThis.localStorage = {{ getItem() {{ throw new Error("blocked"); }} }};
    const theme = await import("{(WEB_DIR / "lib" / "theme.js").as_uri()}");
    const checks = [
      theme.nextTheme("light") === "dark",
      theme.nextTheme("dark") === "light",
      theme.labelFor("light") === "Modo escuro",
      theme.labelFor("dark") === "Modo claro",
      theme.effectiveTheme(null, true) === "dark",
      theme.effectiveTheme("light", true) === "light",
      theme.savedTheme() === null,
    ];
    process.exit(checks.every(Boolean) ? 0 : 1);
    """
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_meter_state_marks_the_limit_and_degrades_without_it() -> None:
    script = f"""
    const meter = await import("{(WEB_DIR / "lib" / "meter.js").as_uri()}");
    const above = meter.meterState(0.91, 0.766, true);
    const below = meter.meterState(0.42, 0.766, false);
    const noLimit = meter.meterState(0.91, null, true);
    const outOfRange = meter.meterState(-0.2, 1.4, false);
    const checks = [
      above.tone === "above" && above.mark === 0.766 && above.fill === 0.91,
      above.limitText === "Acima do limite (0,766)",
      below.tone === "below" && below.limitText === "Abaixo do limite (0,766)",
      noLimit.mark === null && noLimit.limitText === null && noLimit.tone === "above",
      outOfRange.fill === 0 && outOfRange.mark === 1,
    ];
    process.exit(checks.every(Boolean) ? 0 : 1);
    """
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_the_shell_has_a_272px_sidebar_that_turns_off_canvas_below_860px() -> None:
    tokens = (WEB_DIR / "tokens.css").read_text()
    styles = (WEB_DIR / "styles.css").read_text()
    html = (WEB_DIR / "index.html").read_text()
    assert theme_block(tokens, ":root")["--sidebar-w"] == "272px"
    assert "@media (max-width: 860px)" in styles
    assert "body.sidebar-open .sidebar" in styles
    assert 'id="sidebar"' in html
    assert 'id="menu-toggle"' in html
    assert 'aria-expanded="false"' in html
    assert 'aria-controls="sidebar"' in html
    assert 'aria-label="Abrir o menu"' in html


def test_the_off_canvas_sidebar_is_inert_and_closes_with_escape() -> None:
    shell = code_of(WEB_DIR / "lib" / "shell.js")
    assert "sidebar.inert" in shell
    assert '"Escape"' in shell
    assert "toggle.focus()" in shell


def test_every_screen_sets_the_document_title() -> None:
    app = code_of(WEB_DIR / "app.js")
    assert "document.title" in app
    labels = re.findall(r'label: "([^"]+)"', app)
    assert labels == ["Cadastro", "Reconhecimento", "Pessoas cadastradas"]


def test_each_screen_has_a_page_title_and_the_menu_shows_the_people_count() -> None:
    for view in ("enroll", "recognize", "persons"):
        source = code_of(WEB_DIR / "views" / f"{view}.js")
        assert "pageHead(" in source or 'h("h1"' in source, view
    assert "countBadge" in code_of(WEB_DIR / "app.js")


def test_icons_used_by_the_css_exist_and_every_icon_file_is_used() -> None:
    styles = (WEB_DIR / "styles.css").read_text()
    used = set(re.findall(r'url\("assets/icons/([\w-]+)\.svg"\)', styles))
    present = {path.stem for path in (WEB_DIR / "assets" / "icons").glob("*.svg")}
    assert used == present


def test_every_icon_a_script_asks_for_has_a_css_class() -> None:
    styles = (WEB_DIR / "styles.css").read_text()
    asked: set[str] = set()
    for script in scripts():
        source = code_of(script)
        asked |= set(re.findall(r'icon\("([\w-]+)"\)', source))
        asked |= set(re.findall(r'icon: "([\w-]+)"', source))
        asked |= set(re.findall(r"icon--([a-z][\w-]*)", source))
    asked.discard("xl")  # size modifier, not an icon
    assert asked
    for name in asked:
        assert f".icon--{name} " in styles, name


def test_the_enrollment_form_shows_one_error_per_field() -> None:
    source = code_of(WEB_DIR / "views" / "enroll.js")
    assert "aria-invalid" in source
    assert "aria-describedby" in source
    assert "nameError" in source and "photosError" in source
    errors = code_of(WEB_DIR / "lib" / "errors.js")
    assert 'NAME_REQUIRED = "Digite o nome da pessoa."' in errors
    assert 'PHOTOS_REQUIRED = "Adicione de 1 a 10 fotos."' in errors


def test_icons_are_centered_by_a_flex_box_and_not_by_the_text_baseline() -> None:
    styles = (WEB_DIR / "styles.css").read_text()
    assert re.search(r"\.icon \{[^}]*display: inline-flex[^}]*align-items: center", styles)


def test_discarding_a_photo_asks_for_confirmation_with_a_danger_button() -> None:
    source = code_of(WEB_DIR / "lib" / "photos.js")
    assert "confirmDialog(" in source
    assert 'confirmLabel: "Descartar foto"' in source
    assert 'cancelLabel: "Manter foto"' in source
    assert "danger: true" in source


def test_camera_buttons_are_highlighted_and_the_preview_is_mirrored() -> None:
    source = code_of(WEB_DIR / "lib" / "photos.js")
    assert re.search(r'captureButton = h\("button", \{[^}]*class: "secondary"', source)
    assert re.search(r'closeButton = h\("button", \{[^}]*class: "danger"', source)
    styles = (WEB_DIR / "styles.css").read_text()
    assert re.search(r"\.camera \{[^}]*transform: scaleX\(-1\)", styles)


def test_the_page_has_a_favicon_that_exists() -> None:
    html = (WEB_DIR / "index.html").read_text()
    match = re.search(r'<link rel="icon"[^>]*href="([^"]+)"', html)
    assert match
    assert (WEB_DIR / match.group(1)).is_file()


def test_the_enrollment_screen_gives_tips_for_good_photos() -> None:
    source = code_of(WEB_DIR / "views" / "enroll.js")
    assert "Dicas para boas fotos" in source
    assert "TIPS" in source
    assert "um só rosto" in code_of(WEB_DIR / "views" / "recognize.js")


def test_clearing_the_photos_also_clears_the_limit_warning_and_the_camera_can_be_stopped() -> None:
    source = code_of(WEB_DIR / "lib" / "photos.js")
    clear_body = source[source.index("clear() {") : source.index("stop: stopCamera")]
    assert "message.replaceChildren()" in clear_body
    assert "stop: stopCamera" in source
    enroll = code_of(WEB_DIR / "views" / "enroll.js")
    assert "picker.stop()" in enroll


def test_the_similarity_is_shown_as_a_big_percentage_with_the_original_value() -> None:
    source = code_of(WEB_DIR / "views" / "recognize.js")
    assert "formatPercent" in source and "* 100" in source
    assert "Valor original" in source
    styles = (WEB_DIR / "styles.css").read_text()
    assert re.search(r"\.meter__percent \{[^}]*font-size: 32px", styles)


def test_the_tips_box_has_a_violet_highlight_and_a_violet_title() -> None:
    styles = (WEB_DIR / "styles.css").read_text()
    assert re.search(r"\.tips \{[^}]*border: 1px solid var\(--accent\)", styles)
    assert re.search(r"\.tips summary \{[^}]*color: var\(--accent-ink\)", styles)
