// Calls to the JSON API on the same origin. `call` never throws: it returns {ok, status, data},
// with status 0 when the server could not be reached.

const API = "/api";

async function call(path, options = {}) {
  try {
    const response = await fetch(API + path, { ...options, credentials: "same-origin" });
    let data = null;
    try {
      data = await response.json();
    } catch {
      // An empty or non-JSON body leaves data null.
    }
    return { ok: response.ok, status: response.status, data };
  } catch {
    return { ok: false, status: 0, data: null };
  }
}

function formWith(entries) {
  const form = new FormData();
  for (const [field, value, filename] of entries) {
    if (filename === undefined) form.append(field, value);
    else form.append(field, value, filename);
  }
  return form;
}

export const api = {
  listPersons: () => call("/persons"),
  enroll: (name, photos) =>
    call("/persons", {
      method: "POST",
      body: formWith([["name", name], ...photos.map((photo) => ["photos", photo.blob, photo.name])]),
    }),
  recognize: (photo) =>
    call("/recognitions", { method: "POST", body: formWith([["photo", photo.blob, photo.name]]) }),
  deletePerson: (id) => call(`/persons/${id}`, { method: "DELETE" }),
};
