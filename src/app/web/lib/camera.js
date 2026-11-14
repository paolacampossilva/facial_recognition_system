// The browser camera: one photo per click. The stream is stopped as soon as the person leaves,
// so the camera light goes off.

export const cameraAvailable = () => Boolean(navigator.mediaDevices?.getUserMedia);

export async function openCamera(video) {
  const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: false });
  video.srcObject = stream;
  await video.play();
  return stream;
}

export function closeCamera(stream, video) {
  stream?.getTracks().forEach((track) => track.stop());
  video.srcObject = null;
}

// Draws the current frame on a canvas and returns it as a JPEG blob, kept in memory only.
export function captureFrame(video) {
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d").drawImage(video, 0, 0);
  return new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.92));
}
