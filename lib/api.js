// Multipart upload of a captured photo to the backend.
//
// - uploadSample: fire-and-forget upload to /sample with an optional label.
// - uploadAnalyse: upload to /analyse and read back the predicted label.

function joinUrl(baseUrl, path) {
  const b = (baseUrl || '').replace(/\/+$/, '');
  const p = path.startsWith('/') ? path : `/${path}`;
  return `${b}${p}`;
}

function fileFromUri(uri) {
  const name = uri.split('/').pop() || `photo-${Date.now()}.jpg`;
  return { uri, name, type: 'image/jpeg' };
}

export async function uploadSample({ baseUrl, uri, label }) {
  const form = new FormData();
  form.append('image', fileFromUri(uri));
  if (label && label.trim()) form.append('label', label.trim());

  const res = await fetch(joinUrl(baseUrl, '/sample'), {
    method: 'POST',
    body: form,
    headers: { Accept: 'application/json' },
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json().catch(() => ({}));
}

export async function uploadAnalyse({ baseUrl, uri }) {
  const form = new FormData();
  form.append('image', fileFromUri(uri));

  const res = await fetch(joinUrl(baseUrl, '/analyse'), {
    method: 'POST',
    body: form,
    headers: { Accept: 'application/json' },
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const data = await res.json().catch(() => ({}));
  // Backend is expected to return { label, confidence? }.
  return {
    label: data.label ?? data.prediction ?? 'unknown',
    confidence: data.confidence,
  };
}
