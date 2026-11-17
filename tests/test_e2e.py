"""End-to-end tests of the web interface in headless Chrome (issue #50, spec 005).

The server is the real application (API, static interface, services and SQLite) with the face
model swapped for the scripted detector and embedder of `tests.app_support`: the color of a photo
picks the face, so every flow is deterministic and needs no GPU, weights or real photo. The real
model is covered by `tests/test_services_integration.py`.

Every test fails if the page logs a console error or throws, except the HTTP statuses the test
itself expects (the browser logs a failed request, such as a 422, as a console error).

Requires the `playwright` package and Google Chrome installed; without them the tests are skipped.
"""

import contextlib
import re
import threading
import time
from collections.abc import Iterator
from types import SimpleNamespace

import pytest
import uvicorn

from app.api.app import create_app
from app.config import Settings
from app.core.deletion import DeletionService
from app.core.enrollment import EnrollmentService
from app.core.recognition import RecognitionService
from app.storage.repository import MEMORY_DATABASE, SqlitePersonRepository
from tests.app_support import (
    ANA_COLOR,
    BIA_COLOR,
    CARLA_COLOR,
    NO_FACE,
    THRESHOLD,
    TWO_FACES,
    build_encoder,
    free_port,
    photo,
)

playwright_api = pytest.importorskip("playwright.sync_api")
Browser = playwright_api.Browser
Page = playwright_api.Page
expect = playwright_api.expect

pytestmark = pytest.mark.e2e

SERVER_START_SECONDS: float = 10.0
FAILED_REQUEST = re.compile(r"Failed to load resource: the server responded with a status of (\d+)")
WIDE: dict[str, int] = {"width": 1280, "height": 900}
NARROW: dict[str, int] = {"width": 600, "height": 900}


def png(color: tuple[int, int, int], name: str, width: int | None = None) -> dict[str, object]:
    """A photo for the file input: Ana, Bia and Carla are colors, the width picks the faces."""
    content = photo(color) if width is None else photo(color, width)
    return {"name": name, "mimeType": "image/png", "buffer": content}


@pytest.fixture(scope="module")
def browser() -> Iterator[Browser]:
    """Headless Chrome, with a fake camera that the page is allowed to use."""
    with playwright_api.sync_playwright() as playwright:
        try:
            chrome = playwright.chromium.launch(
                channel="chrome",
                headless=True,
                args=["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"],
            )
        except playwright_api.Error as error:
            pytest.skip(f"Google Chrome is not available: {error.message.splitlines()[0]}")
        yield chrome
        chrome.close()


@pytest.fixture
def server() -> Iterator[str]:
    """The application on a free port, with an empty in-memory database; yields its URL."""
    repository = SqlitePersonRepository(MEMORY_DATABASE)
    encoder = build_encoder()
    services = SimpleNamespace(
        enrollment=EnrollmentService(encoder, repository, THRESHOLD),
        recognition=RecognitionService(encoder, repository, THRESHOLD),
        deletion=DeletionService(repository),
        persons=repository,
    )
    port = free_port()
    config = uvicorn.Config(
        create_app(services, Settings(threshold=THRESHOLD)),
        host="127.0.0.1",
        port=port,
        log_level="warning",
    )
    running = uvicorn.Server(config)
    thread = threading.Thread(target=running.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + SERVER_START_SECONDS
    while not running.started:
        if time.monotonic() > deadline:
            pytest.fail("the server did not start")
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    running.should_exit = True
    thread.join(timeout=SERVER_START_SECONDS)
    repository.close()


class ConsoleWatch:
    """Collects console errors and uncaught exceptions of a page."""

    def __init__(self, page: Page) -> None:
        self.errors: list[str] = []
        self.expected_statuses: set[int] = set()
        page.on("console", lambda message: self._on_console(message.type, message.text))
        page.on("pageerror", lambda error: self.errors.append(f"uncaught: {error}"))

    def _on_console(self, kind: str, text: str) -> None:
        if kind != "error":
            return
        failed = FAILED_REQUEST.search(text)
        if failed and int(failed.group(1)) in self.expected_statuses:
            return
        self.errors.append(text)

    def expect_statuses(self, *statuses: int) -> None:
        """HTTP errors the test provokes on purpose, so their console lines are not failures."""
        self.expected_statuses.update(statuses)


@pytest.fixture
def console() -> list[ConsoleWatch]:
    """The watches of the pages a test opens; any error left in them fails the test."""
    watches: list[ConsoleWatch] = []
    yield watches
    errors = [error for watch in watches for error in watch.errors]
    assert errors == [], f"the page logged console errors: {errors}"


@pytest.fixture
def open_page(browser: Browser, server: str, console: list[ConsoleWatch]):
    """Opens the application in a new browser context; returns the page and its console watch."""
    contexts = []

    def opener(viewport: dict[str, int] = WIDE, camera: bool = False) -> tuple[Page, ConsoleWatch]:
        context = browser.new_context(viewport=viewport, base_url=server)
        if camera:
            context.grant_permissions(["camera"])
        contexts.append(context)
        page = context.new_page()
        watch = ConsoleWatch(page)
        console.append(watch)
        page.goto("/")
        expect(page.locator("h1")).to_have_text("Cadastro")
        return page, watch

    yield opener
    for context in contexts:
        context.close()


def go_to(page: Page, screen: str) -> None:
    page.locator("#menu").get_by_role("link", name=re.compile(f"^{screen}")).click()
    expect(page.locator("h1")).to_have_text(screen)


def enroll(page: Page, name: str, *photos: dict[str, object]) -> None:
    """Fills in the enrollment form and sends it; the caller checks the outcome."""
    go_to(page, "Cadastro")
    page.locator("#person-name").fill(name)
    page.locator("input[type=file]").set_input_files(list(photos))
    page.get_by_role("button", name="Cadastrar").click()


def recognize(page: Page, picture: dict[str, object]) -> None:
    page.locator("input[type=file]").set_input_files([picture])
    page.get_by_role("button", name="Reconhecer").click()


CAMERA_ATTEMPTS: int = 3


def open_camera(page: Page) -> None:
    """Opens the camera and waits for its image.

    Chrome's fake camera sometimes refuses a stream right after another one was stopped (the app
    then shows its "camera unavailable" message, as it should). That is the test device, not the
    application, so the opening is retried a few times; it must succeed in the end.
    """
    camera = page.locator("video.camera")
    button = page.get_by_role("button", name="Usar a câmera")
    for _ in range(CAMERA_ATTEMPTS - 1):
        if button.is_visible():  # hidden once a slow opening succeeds
            button.click()
        with contextlib.suppress(AssertionError):
            expect(camera).to_be_visible(timeout=2000)
            return
    if button.is_visible():
        button.click()
    expect(camera).to_be_visible()


def menu_count(page: Page):
    return page.locator("#menu .menu__count")


def test_enroll_recognize_unknown_and_delete(open_page) -> None:
    """The whole flow of the issue: A1, A6, A7, A8, A9 to A11, A25 and A31 of spec 005."""
    page, _ = open_page()

    # A8 and A25: with nobody enrolled, both screens show the empty state with the way out.
    go_to(page, "Reconhecimento")
    expect(page.get_by_text("Nenhuma pessoa cadastrada.")).to_be_visible()
    expect(page.get_by_role("link", name="Ir para o Cadastro")).to_be_visible()
    go_to(page, "Pessoas cadastradas")
    expect(page.get_by_text("Nenhuma pessoa cadastrada.")).to_be_visible()
    expect(menu_count(page)).to_have_text("0")

    # A1: two photos of Ana; the photo list and the form are cleared, and the count goes up.
    enroll(page, "Ana", png(ANA_COLOR, "ana1.png"), png(ANA_COLOR, "ana2.png"))
    expect(page.get_by_text("Cadastro de Ana concluído.")).to_be_visible()
    expect(page.get_by_text("0 de 10 fotos")).to_be_visible()
    expect(page.locator("#person-name")).to_have_value("")
    expect(menu_count(page)).to_have_text("1")

    # A6 and A31: Ana is recognized, the bar is above the limit read from GET /api/config.
    go_to(page, "Reconhecimento")
    recognize(page, png(ANA_COLOR, "query.png"))
    result = page.get_by_role("status").filter(has_text="Pessoa reconhecida")
    expect(result).to_contain_text("Ana")
    expect(result).to_contain_text("100,0%")
    expect(result).to_contain_text("Acima do limite (0,766)")

    # A7 and A31: an unknown face shows no name at all, and the bar is below the limit.
    recognize(page, png(BIA_COLOR, "query.png"))
    result = page.get_by_role("status").filter(has_text="Pessoa desconhecida")
    expect(result).to_contain_text("Abaixo do limite (0,766)")
    expect(result).not_to_contain_text("Ana")

    # A9 and A10: the dialog names the action, and Esc keeps the person.
    go_to(page, "Pessoas cadastradas")
    page.get_by_role("button", name="Excluir Ana").click()
    dialog = page.get_by_role("alertdialog")
    expect(dialog.get_by_role("button", name="Excluir Ana")).to_be_visible()
    expect(dialog.get_by_role("button", name="Cancelar")).to_be_focused()
    page.keyboard.press("Escape")
    expect(dialog).to_have_count(0)
    expect(page.get_by_text("2 fotos usadas")).to_be_visible()

    # A11: confirming deletes; the list, the count and the recognition all forget Ana.
    page.get_by_role("button", name="Excluir Ana").click()
    page.get_by_role("alertdialog").get_by_role("button", name="Excluir Ana").click()
    expect(page.get_by_text("Cadastro de Ana excluído.")).to_be_visible()
    expect(page.get_by_text("Nenhuma pessoa cadastrada.")).to_be_visible()
    expect(menu_count(page)).to_have_text("0")
    go_to(page, "Reconhecimento")
    expect(page.get_by_text("Nenhuma pessoa cadastrada.")).to_be_visible()


@pytest.mark.parametrize(
    ("picture", "message"),
    [
        (png(ANA_COLOR, "sem-rosto.png", NO_FACE), "Nenhum rosto encontrado na imagem"),
        (png(ANA_COLOR, "dois.png", TWO_FACES), "Mais de um rosto na imagem"),
        (
            {"name": "texto.png", "mimeType": "image/png", "buffer": b"not an image"},
            "Arquivo de imagem inválido",
        ),
    ],
    ids=["no-face", "two-faces", "not-an-image"],
)
def test_a_bad_photo_shows_its_message_and_enrolls_nobody(
    open_page, picture: dict[str, object], message: str
) -> None:
    """A2: the image errors of spec 001 reach the person in Portuguese, and nothing is saved."""
    page, watch = open_page()
    watch.expect_statuses(422)

    enroll(page, "Ana", picture)

    expect(page.locator(".notice").filter(has_text=message)).to_be_visible()
    expect(page.get_by_role("button", name="Cadastrar")).to_be_enabled()
    expect(menu_count(page)).to_have_text("0")


def test_duplicates_by_name_and_by_face_are_refused(open_page) -> None:
    """A3 and A4: the same name in other case, and Ana's face under another name."""
    page, watch = open_page()
    watch.expect_statuses(409)
    enroll(page, "Ana", png(ANA_COLOR, "ana.png"))
    expect(page.get_by_text("Cadastro de Ana concluído.")).to_be_visible()

    enroll(page, "ANA", png(BIA_COLOR, "bia.png"))
    expect(page.get_by_text("Já existe uma pessoa cadastrada com esse nome")).to_be_visible()

    # Carla's face is 0.9 similar to Ana's, above the threshold.
    page.get_by_role("button", name="Descartar a foto 1").click()
    page.get_by_role("alertdialog").get_by_role("button", name="Descartar foto").click()
    page.locator("#person-name").fill("Carla")
    page.locator("input[type=file]").set_input_files([png(CARLA_COLOR, "carla.png")])
    page.get_by_role("button", name="Cadastrar").click()
    expect(page.get_by_text("Esse rosto já está cadastrado como Ana")).to_be_visible()
    expect(menu_count(page)).to_have_text("1")


def test_an_empty_form_marks_each_field_and_sends_nothing(open_page) -> None:
    """A26: the field errors come from the browser, next to each field."""
    page, _ = open_page()

    page.get_by_role("button", name="Cadastrar").click()

    name = page.locator("#person-name")
    name_error = page.locator("#person-name-error")
    expect(name_error).to_have_text("Digite o nome da pessoa.")
    expect(page.locator("#photos-error")).to_have_text("Adicione de 1 a 10 fotos.")
    expect(name).to_have_attribute("aria-invalid", "true")
    expect(name).to_have_attribute("aria-describedby", "person-name-error")
    expect(name).to_be_focused()
    name.fill("Ana")
    expect(name_error).to_be_hidden()
    expect(name).not_to_have_attribute("aria-invalid", "true")


def test_discarding_a_photo_asks_first(open_page) -> None:
    """A27: Esc and "Manter foto" keep the photo, "Descartar foto" removes it."""
    page, _ = open_page()
    page.locator("input[type=file]").set_input_files([png(ANA_COLOR, "ana.png")])
    expect(page.get_by_text("1 de 10 fotos")).to_be_visible()

    page.get_by_role("button", name="Descartar a foto 1").click()
    page.keyboard.press("Escape")
    page.get_by_role("button", name="Descartar a foto 1").click()
    page.get_by_role("alertdialog").get_by_role("button", name="Manter foto").click()
    expect(page.get_by_text("1 de 10 fotos")).to_be_visible()

    page.get_by_role("button", name="Descartar a foto 1").click()
    page.get_by_role("alertdialog").get_by_role("button", name="Descartar foto").click()
    expect(page.get_by_text("0 de 10 fotos")).to_be_visible()


def test_leaving_a_filled_form_asks_before_losing_it(open_page) -> None:
    """A15: "Continuar editando" keeps the form, "Sair sem salvar" leaves."""
    page, _ = open_page()
    page.locator("#person-name").fill("Ana")

    page.locator("#menu").get_by_role("link", name="Reconhecimento").click()
    dialog = page.get_by_role("alertdialog")
    expect(dialog).to_contain_text("Sair sem salvar?")
    dialog.get_by_role("button", name="Continuar editando").click()
    expect(page.locator("h1")).to_have_text("Cadastro")
    expect(page.locator("#person-name")).to_have_value("Ana")

    page.locator("#menu").get_by_role("link", name="Reconhecimento").click()
    page.get_by_role("alertdialog").get_by_role("button", name="Sair sem salvar").click()
    expect(page.locator("h1")).to_have_text("Reconhecimento")


def track_camera(page: Page, delay_ms: int = 0) -> None:
    """Keeps every camera stream the page opens, to check later that its tracks were stopped.

    `delay_ms` makes the camera slow to open, as a real one can be.
    """
    page.evaluate(
        "(delay) => { const devices = navigator.mediaDevices;"
        " const open = devices.getUserMedia.bind(devices);"
        " window.streams = [];"
        " navigator.mediaDevices.getUserMedia = async (constraints) => {"
        " await new Promise((resolve) => setTimeout(resolve, delay));"
        " const stream = await open(constraints); window.streams.push(stream); return stream; }; }",
        delay_ms,
    )


CAMERA_STOP_MS: int = 3000
TRACK_STATES: str = "window.streams.flatMap((s) => s.getTracks().map((t) => t.readyState))"


def camera_states(page: Page, streams: int) -> list[str]:
    """The state of every track once the page opened `streams` cameras and had time to stop them."""
    page.wait_for_function("(count) => window.streams.length === count", arg=streams)
    # On a timeout the caller's assertion shows which track is still live.
    with contextlib.suppress(playwright_api.TimeoutError):
        page.wait_for_function(
            f"() => {TRACK_STATES}.every((state) => state === 'ended')", timeout=CAMERA_STOP_MS
        )
    return page.evaluate(f"() => {TRACK_STATES}")


def test_the_camera_takes_a_photo_and_turns_off_when_leaving(open_page) -> None:
    """A28 and R3: a capture becomes a photo of the list, and leaving closes the camera."""
    page, _ = open_page(camera=True)
    track_camera(page)

    open_camera(page)
    page.get_by_role("button", name="Capturar foto").click()
    expect(page.get_by_text("1 de 10 fotos")).to_be_visible()
    page.get_by_role("button", name="Fechar a câmera").click()
    expect(page.locator("video.camera")).to_be_hidden()
    expect(page.get_by_role("button", name="Usar a câmera")).to_be_visible()

    open_camera(page)
    page.locator("#menu").get_by_role("link", name="Reconhecimento").click()
    page.get_by_role("alertdialog").get_by_role("button", name="Sair sem salvar").click()
    expect(page.locator("h1")).to_have_text("Reconhecimento")

    assert camera_states(page, streams=2) == ["ended", "ended"]


def test_a_camera_still_opening_when_leaving_is_turned_off(open_page) -> None:
    """R27: leaving while the camera is still opening must not leave it on when it opens."""
    page, _ = open_page(camera=True)
    track_camera(page, delay_ms=500)

    page.get_by_role("button", name="Usar a câmera").click()
    page.locator("#menu").get_by_role("link", name="Reconhecimento").click()
    expect(page.locator("h1")).to_have_text("Reconhecimento")

    assert camera_states(page, streams=1) == ["ended"]


def test_a_double_click_on_the_camera_button_opens_one_camera(open_page) -> None:
    """R27: two clicks before the camera opens must not leave a second stream on with no owner."""
    page, _ = open_page(camera=True)
    track_camera(page, delay_ms=500)

    page.evaluate(
        "() => { const button = [...document.querySelectorAll('button')]"
        ".find((item) => item.textContent.includes('Usar a câmera'));"
        " button.click(); button.click(); }"
    )
    expect(page.locator("video.camera")).to_be_visible()
    page.get_by_role("button", name="Fechar a câmera").click()

    assert camera_states(page, streams=1) == ["ended"]


def test_a_camera_that_cannot_play_is_turned_off(open_page) -> None:
    """R27: when the video refuses to play after the camera was granted, the stream is stopped."""
    page, _ = open_page(camera=True)
    track_camera(page)
    page.evaluate(
        "() => { HTMLMediaElement.prototype.play = () =>"
        " Promise.reject(new DOMException('refused', 'NotAllowedError')); }"
    )

    page.get_by_role("button", name="Usar a câmera").click()

    expect(page.get_by_text("Não foi possível usar a câmera. Envie um arquivo")).to_be_visible()
    assert camera_states(page, streams=1) == ["ended"]


def test_a_camera_that_goes_away_closes_the_camera_box(open_page) -> None:
    """A device that ends by itself (unplugged, permission revoked) leaves no frozen preview."""
    page, _ = open_page(camera=True)
    track_camera(page)
    open_camera(page)

    page.evaluate(
        "() => window.streams[0].getTracks().forEach((track) => {"
        " track.stop(); track.dispatchEvent(new Event('ended')); })"
    )

    expect(page.locator("video.camera")).to_be_hidden()
    expect(page.get_by_role("button", name="Usar a câmera")).to_be_visible()
    expect(page.get_by_text("Não foi possível usar a câmera. Envie um arquivo")).to_be_visible()


def test_a_denied_camera_keeps_the_file_upload(open_page) -> None:
    """A13: without permission, the message shows and a file still works."""
    page, _ = open_page()
    # The fake camera of the browser always accepts, so the refusal is simulated in the page.
    page.evaluate(
        "() => { navigator.mediaDevices.getUserMedia = () =>"
        " Promise.reject(new DOMException('denied', 'NotAllowedError')); }"
    )

    page.get_by_role("button", name="Usar a câmera").click()

    expect(page.get_by_text("Não foi possível usar a câmera. Envie um arquivo")).to_be_visible()
    page.locator("input[type=file]").set_input_files([png(ANA_COLOR, "ana.png")])
    expect(page.get_by_text("1 de 10 fotos")).to_be_visible()


def test_the_theme_survives_a_reload_as_the_only_stored_key(open_page) -> None:
    """A16: the chosen theme is kept under "theme" and nothing else is stored."""
    page, _ = open_page()
    button = page.locator("#theme")
    goes_to = button.get_attribute("data-goes")

    button.click()
    page.reload()

    expect(page.locator("html")).to_have_attribute("data-theme", goes_to or "")
    assert page.evaluate("() => Object.keys(localStorage)") == ["theme"]


def test_each_screen_sets_its_title(open_page) -> None:
    """A22 and A24: the side bar has the three screens, and the tab title follows the screen."""
    page, _ = open_page()

    links = page.locator("#menu").get_by_role("link")
    expect(links).to_have_count(3)
    for screen in ["Reconhecimento", "Pessoas cadastradas", "Cadastro"]:
        go_to(page, screen)
        expect(page).to_have_title(f"{screen} · Reconhecimento facial")


def test_on_a_narrow_window_the_menu_opens_and_esc_closes_it(open_page) -> None:
    """A23: below 860px the side bar is a drawer; Esc closes it and the focus goes back."""
    page, _ = open_page(viewport=NARROW)
    toggle = page.locator("#menu-toggle")
    expect(toggle).to_be_visible()

    toggle.click()
    expect(toggle).to_have_attribute("aria-expanded", "true")
    page.keyboard.press("Escape")

    expect(toggle).to_have_attribute("aria-expanded", "false")
    expect(toggle).to_be_focused()


def test_deleting_someone_already_deleted_refreshes_the_list(open_page) -> None:
    """A12: the person was deleted elsewhere; the screen says so and updates the list."""
    page, watch = open_page()
    watch.expect_statuses(404)
    enroll(page, "Ana", png(ANA_COLOR, "ana.png"))
    expect(page.get_by_text("Cadastro de Ana concluído.")).to_be_visible()
    go_to(page, "Pessoas cadastradas")
    expect(page.get_by_text("1 foto usada")).to_be_visible()

    assert page.request.delete("/api/persons/1").ok
    page.get_by_role("button", name="Excluir Ana").click()
    page.get_by_role("alertdialog").get_by_role("button", name="Excluir Ana").click()

    expect(page.get_by_text("Pessoa não encontrada. A lista foi atualizada")).to_be_visible()
    expect(page.get_by_text("Nenhuma pessoa cadastrada.")).to_be_visible()


def test_the_send_button_is_busy_during_the_call_and_free_after(open_page) -> None:
    """A14: the button is disabled with a spinner while the request runs."""
    page, _ = open_page()
    page.locator("#person-name").fill("Ana")
    page.locator("input[type=file]").set_input_files([png(ANA_COLOR, "ana.png")])

    # The click runs the handler up to the request, so the state is read right after it.
    busy = page.evaluate(
        "() => { const button = [...document.querySelectorAll('button')]"
        ".find((b) => b.textContent.trim() === 'Cadastrar'); button.click();"
        " return [button.disabled, Boolean(button.querySelector('.spinner'))]; }"
    )

    assert busy == [True, True]
    expect(page.get_by_text("Cadastro de Ana concluído.")).to_be_visible()
    button = page.get_by_role("button", name="Cadastrar")
    expect(button).to_be_enabled()
    expect(button.locator(".spinner")).to_have_count(0)


def hex_to_rgb(color: str) -> tuple[float, float, float]:
    value = color.strip().lstrip("#")
    return tuple(int(value[index : index + 2], 16) / 255 for index in (0, 2, 4))  # type: ignore[return-value]


def luminance(color: str) -> float:
    def linear(channel: float) -> float:
        return channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4

    red, green, blue = (linear(channel) for channel in hex_to_rgb(color))
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def hue(color: str) -> float:
    red, green, blue = hex_to_rgb(color)
    high, low = max(red, green, blue), min(red, green, blue)
    if high == low:
        return 0.0
    if high == red:
        return (60 * ((green - blue) / (high - low))) % 360
    if high == green:
        return 60 * ((blue - red) / (high - low)) + 120
    return 60 * ((red - green) / (high - low)) + 240


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_the_accent_is_violet_with_readable_text_in_both_themes(open_page, theme: str) -> None:
    """A17: `--accent` is violet (not reddish) and the text on it has a contrast of 4.5 or more."""
    page, _ = open_page()
    page.evaluate(f"() => document.documentElement.setAttribute('data-theme', '{theme}')")

    accent, on_accent = page.evaluate(
        "() => { const style = getComputedStyle(document.documentElement);"
        " return [style.getPropertyValue('--accent'), style.getPropertyValue('--on-accent')]; }"
    )

    assert 250 <= hue(accent) <= 290
    lighter, darker = sorted([luminance(accent), luminance(on_accent)], reverse=True)
    assert (lighter + 0.05) / (darker + 0.05) >= 4.5


def test_the_enrollment_shows_the_photo_tips_open(open_page) -> None:
    """A29: the tips for good photos are open in the photo card, five of them."""
    page, _ = open_page()

    tips = page.locator("details.tips")

    expect(tips).to_have_attribute("open", "")
    expect(tips.locator("summary")).to_have_text("Dicas para boas fotos")
    expect(tips.locator("li")).to_have_count(5)


def test_enrolling_closes_the_camera_and_clears_the_photo_limit_warning(open_page) -> None:
    """A30: past 10 photos a warning shows; a successful enrollment clears it and the camera."""
    page, _ = open_page(camera=True)
    open_camera(page)
    photos = [png(ANA_COLOR, f"ana{index}.png") for index in range(11)]

    page.locator("input[type=file]").set_input_files(photos)
    expect(page.get_by_text("O limite é de 10 fotos.")).to_be_visible()
    expect(page.get_by_text("10 de 10 fotos")).to_be_visible()
    page.locator("#person-name").fill("Ana")
    page.get_by_role("button", name="Cadastrar").click()

    expect(page.get_by_text("Cadastro de Ana concluído.")).to_be_visible()
    expect(page.locator("video.camera")).to_be_hidden()
    expect(page.get_by_text("O limite é de 10 fotos.")).to_have_count(0)
    expect(page.get_by_text("0 de 10 fotos")).to_be_visible()
