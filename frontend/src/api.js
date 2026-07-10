// Thin client for the Python backend. Same-origin in dev (vite proxy) and in
// the packaged app; VITE_API_BASE overrides for unusual setups.
const BASE = import.meta.env.VITE_API_BASE || ''

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, options)
  const body = await res.json().catch(() => ({ detail: res.statusText }))
  if (!res.ok) {
    const detail = Array.isArray(body.detail) // FastAPI 422s carry a list of {loc, msg, ...}
      ? body.detail.map((d) => d.msg || JSON.stringify(d)).join('; ')
      : body.detail
    throw new Error(detail || `Request failed (${res.status})`)
  }
  return body
}

export const getStatus = () => request('/api/status')
export const getFiles = () => request('/api/files')
export const startIngest = () => request('/api/ingest', { method: 'POST' })
export const getSettings = () => request('/api/settings')

export const putSettings = (settings) =>
  request('/api/settings', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(settings),
  })

export const ask = (question, files) =>
  request('/api/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, files: files?.length ? files : null }),
  })

export async function uploadFiles(fileList) {
  const failed = []
  for (const file of fileList) {
    const form = new FormData()
    form.append('file', file)
    try {
      await request('/api/upload', { method: 'POST', body: form })
    } catch (e) {
      failed.push(`${file.name}: ${e.message}`)
    }
  }
  // callers still ingest whatever made it up; failures are reported, not silent
  if (failed.length) throw new Error(`Failed to upload — ${failed.join(', ')}`)
}

export const pdfUrl = (name) => `${BASE}/api/pdf/${encodeURIComponent(name)}`
