// The browser camera: one photo per click. The stream is stopped as soon as the person leaves,
// so the camera light goes off.

export const cameraAvailable = () => Boolean(navigator.mediaDevices?.getUserMedia);

const stopStream = (stream) => stream?.getTracks().forEach((track) => track.stop());

// Opens the camera into `video`. `isCurrent` says whether the person still wants this opening: when
// it is no longer current the stream is stopped and null is returned, and `video` is left alone.
// Any failure after the camera was granted stops its stream too, so no camera stays on unseen.
export async function openCamera(video, isCurrent = () => true) {
  const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: false });
  if (!isCurrent()) {
    stopStream(stream);
    return null;
  }
  try {
    video.srcObject = stream;
    await video.play();
  } catch (error) {
    stopStream(stream);
    if (video.srcObject === stream) video.srcObject = null;
    throw error;
  }
  if (!isCurrent()) {
    stopStream(stream);
    if (video.srcObject === stream) video.srcObject = null;
    return null;
  }
  return stream;
}

export function closeCamera(stream, video) {
  stopStream(stream);
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
