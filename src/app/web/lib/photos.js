// Photo chooser used by the enrollment and recognition screens: camera capture and file upload,
// a preview of each photo and a way to remove it. Photos live in memory as blobs.

import { captureFrame, cameraAvailable, closeCamera, openCamera } from "./camera.js";
import { h, icon, notice } from "./dom.js";
import { CAMERA_UNAVAILABLE } from "./errors.js";

export function photoPicker({ max, onChange = () => {} }) {
  let items = [];
  let stream = null;

  const list = h("ul", { class: "photos", "aria-label": "Fotos escolhidas" });
  const message = h("div", { "aria-live": "polite" });
  const video = h("video", { class: "camera", autoplay: true, playsinline: true, muted: true, hidden: true });
  const input = h("input", { type: "file", accept: "image/*", multiple: max > 1, hidden: true });
  const cameraButton = h("button", { type: "button", class: "secondary" }, icon("camera"), "Usar a câmera");
  const captureButton = h("button", { type: "button", class: "secondary", hidden: true }, "Capturar foto");
  const closeButton = h("button", { type: "button", class: "secondary", hidden: true }, "Fechar a câmera");
  const uploadButton = h("button", { type: "button", class: "secondary" }, icon("upload"), "Enviar arquivo");

  function render() {
    list.replaceChildren(...items.map((item, index) =>
      h("li", { class: "photo" },
        h("img", { src: item.url, alt: `Foto ${index + 1}` }),
        h("button", {
          type: "button", class: "secondary", "aria-label": `Remover a foto ${index + 1}`,
          onclick: () => remove(index),
        }, icon("trash")))));
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

  function remove(index) {
    URL.revokeObjectURL(items[index].url);
    items = items.filter((_, position) => position !== index);
    message.replaceChildren();
    render();
  }

  function discard() {
    items.forEach((item) => URL.revokeObjectURL(item.url));
    items = [];
  }

  function stopCamera() {
    closeCamera(stream, video);
    stream = null;
    video.hidden = true;
    captureButton.hidden = true;
    closeButton.hidden = true;
    cameraButton.hidden = false;
  }

  cameraButton.addEventListener("click", async () => {
    if (!cameraAvailable()) {
      message.replaceChildren(notice("error", CAMERA_UNAVAILABLE));
      return;
    }
    try {
      stream = await openCamera(video);
    } catch {
      message.replaceChildren(notice("error", CAMERA_UNAVAILABLE));
      return;
    }
    message.replaceChildren();
    video.hidden = false;
    captureButton.hidden = false;
    closeButton.hidden = false;
    cameraButton.hidden = true;
  });
  captureButton.addEventListener("click", async () => {
    const blob = await captureFrame(video);
    if (blob) add(blob, "camera.jpg");
  });
  closeButton.addEventListener("click", stopCamera);
  uploadButton.addEventListener("click", () => input.click());
  input.addEventListener("change", () => {
    for (const file of input.files) add(file, file.name);
    input.value = "";
  });

  const element = h("div", { class: "stack" },
    h("div", { class: "row" }, cameraButton, uploadButton, input),
    video,
    h("div", { class: "row" }, captureButton, closeButton),
    message,
    list);

  return {
    element,
    count: () => items.length,
    files: () => items.map(({ blob, name }) => ({ blob, name })),
    clear() {
      discard();
      render();
    },
    dispose() {
      stopCamera();
      discard();
    },
  };
}
