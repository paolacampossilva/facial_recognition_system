// Photo chooser used by the enrollment and recognition screens: camera capture and file upload,
// a preview of each photo and a way to remove it. Photos live in memory as blobs.

import { captureFrame, cameraAvailable, closeCamera, openCamera } from "./camera.js";
import { confirmDialog } from "./dialogs.js";
import { h, icon, notice } from "./dom.js";
import { CAMERA_UNAVAILABLE } from "./errors.js";

export function photoPicker({ max, onChange = () => {} }) {
  let items = [];
  let stream = null;
  // Bumped by every stop: a camera that finishes opening after the person left (or closed it) is
  // turned off at once instead of staying on with nobody seeing it.
  let cameraTicket = 0;
  // True while a camera is opening: a second click must not open another one.
  let opening = false;

  const list = h("ul", { class: "photos", "aria-label": "Fotos escolhidas" });
  const empty = h("div", { class: "photos-empty" }, icon("camera"), "As fotos escolhidas aparecem aqui.");
  const message = h("div", { "aria-live": "polite" });
  const video = h("video", {
    class: "camera", autoplay: true, playsinline: true, muted: true, "aria-label": "Prévia da câmera",
  });
  video.muted = true; // the attribute alone does not always set the property
  const input = h("input", { type: "file", accept: "image/*", multiple: max > 1, hidden: true });
  const cameraButton = h("button", { type: "button", class: "secondary" }, icon("camera"), "Usar a câmera");
  const captureButton = h("button", { type: "button", class: "secondary" }, icon("camera"), "Capturar foto");
  const closeButton = h("button", { type: "button", class: "danger" }, icon("x"), "Fechar a câmera");
  const cameraBox = h("div", { class: "camera-box", hidden: true },
    video, h("div", { class: "row" }, captureButton, closeButton));
  const uploadButton = h("button", { type: "button", class: "secondary" }, icon("upload"), "Enviar arquivo");

  function render() {
    list.replaceChildren(...items.map((item, index) =>
      h("li", { class: "photo" },
        h("img", { src: item.url, alt: `Foto ${index + 1}` }),
        h("button", {
          type: "button", class: "secondary", "aria-label": `Descartar a foto ${index + 1}`,
          onclick: () => remove(item),
        }, icon("trash")))));
    empty.hidden = items.length > 0;
    onChange(items.length);
  }

  function add(blob, name) {
    if (max === 1) discard();
    if (items.length >= max) {
      message.replaceChildren(notice("warn", `O limite é de ${max} fotos.`));
      return;
    }
    message.replaceChildren();
    items.push({ blob, name, url: URL.createObjectURL(blob) });
    render();
  }

  async function remove(item) {
    const { confirmed } = await confirmDialog({
      title: "Descartar esta foto?",
      message: "A foto será removida da lista. Você pode adicionar outra depois.",
      confirmLabel: "Descartar foto",
      cancelLabel: "Manter foto",
      danger: true,
    });
    if (!confirmed) return;
    URL.revokeObjectURL(item.url);
    const position = items.indexOf(item);
    items = items.filter((other) => other !== item);
    message.replaceChildren();
    render();
    // The button that had the focus is gone: hand it to the neighbor photo, or to the upload button.
    const buttons = list.querySelectorAll("button");
    (buttons[Math.min(position, buttons.length - 1)] ?? uploadButton).focus();
  }

  function discard() {
    items.forEach((item) => URL.revokeObjectURL(item.url));
    items = [];
  }

  function stopCamera() {
    cameraTicket += 1;
    opening = false;
    cameraButton.disabled = false;
    closeCamera(stream, video);
    stream = null;
    cameraBox.hidden = true;
    cameraButton.hidden = false;
  }

  // The camera went away by itself (unplugged, permission taken back): show it, do not keep a frozen box.
  function watchEnded(opened) {
    opened.getTracks().forEach((track) => track.addEventListener("ended", () => {
      if (stream !== opened) return;
      stopCamera();
      message.replaceChildren(notice("error", CAMERA_UNAVAILABLE));
    }));
  }

  cameraButton.addEventListener("click", async () => {
    if (opening || stream) return;
    if (!cameraAvailable()) {
      message.replaceChildren(notice("error", CAMERA_UNAVAILABLE));
      return;
    }
    opening = true;
    cameraButton.disabled = true;
    const ticket = cameraTicket;
    const current = () => ticket === cameraTicket;
    try {
      const opened = await openCamera(video, current);
      if (!opened) return; // the person left or closed it meanwhile, and it is already off
      stream = opened;
      watchEnded(opened);
      message.replaceChildren();
      cameraBox.hidden = false;
      cameraButton.hidden = true;
      captureButton.focus(); // the button the person used is now hidden
    } catch {
      if (current()) message.replaceChildren(notice("error", CAMERA_UNAVAILABLE));
    } finally {
      // A stop in the meantime already reset this, and a newer opening owns the flag now.
      if (current()) {
        opening = false;
        cameraButton.disabled = false;
      }
    }
  });
  captureButton.addEventListener("click", async () => {
    if (!stream) return;
    const ticket = cameraTicket;
    const blob = await captureFrame(video);
    if (ticket !== cameraTicket) return; // the camera was closed while the frame was being saved
    if (blob) add(blob, "camera.jpg");
  });
  closeButton.addEventListener("click", () => {
    stopCamera();
    cameraButton.focus(); // the button that was clicked is hidden now
  });
  uploadButton.addEventListener("click", () => input.click());
  input.addEventListener("change", () => {
    for (const file of input.files) add(file, file.name);
    input.value = "";
  });

  const element = h("div", { class: "stack" },
    h("div", { class: "row" }, cameraButton, uploadButton, input),
    cameraBox,
    message,
    empty,
    list);

  return {
    element,
    count: () => items.length,
    files: () => items.map(({ blob, name }) => ({ blob, name })),
    // Drops every photo and the limit warning (after a successful submit, for instance).
    clear() {
      discard();
      message.replaceChildren();
      render();
    },
    // Turns the camera off without waiting for the person to click "Fechar a câmera".
    stop: stopCamera,
    dispose() {
      stopCamera();
      discard();
    },
  };
}
