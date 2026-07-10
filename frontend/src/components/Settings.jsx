import { useEffect, useState } from 'react'
import { getSettings, putSettings } from '../api.js'

export default function Settings({ onClose }) {
  const [settings, setSettings] = useState(null)
  const [apiKey, setApiKey] = useState('')
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    getSettings().then(setSettings).catch((e) => setError(e.message))
  }, [])

  const save = async () => {
    setSaving(true)
    setError(null)
    try {
      const updated = await putSettings({
        provider: settings.provider,
        model: settings.model,
        api_key: apiKey || null, // never sent back down; only up, and only if entered
      })
      setSettings(updated)
      setApiKey('')
      onClose()
    } catch (e) {
      setError(e.message)
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <h2>Settings</h2>
        {!settings && !error && <p className="hint">Loading…</p>}
        {settings && (
          <>
            <label>
              Provider
              <select
                value={settings.provider}
                onChange={(e) => setSettings({ ...settings, provider: e.target.value })}
              >
                {settings.providers.map((p) => <option key={p} value={p}>{p}</option>)}
              </select>
            </label>
            <label>
              Model
              <input
                value={settings.model}
                onChange={(e) => setSettings({ ...settings, model: e.target.value })}
              />
            </label>
            <label>
              API key {settings.has_key && <span className="hint">(one is stored — leave blank to keep it)</span>}
              <input
                type="password"
                value={apiKey}
                placeholder="paste to set or replace"
                autoComplete="off"
                onChange={(e) => setApiKey(e.target.value)}
              />
            </label>
          </>
        )}
        {error && <p className="error">{error}</p>}
        <div className="modal-actions">
          <button className="ghost" onClick={onClose}>Cancel</button>
          <button className="primary" onClick={save} disabled={!settings || saving}>
            {saving ? 'Saving…' : 'Save'}
          </button>
        </div>
      </div>
    </div>
  )
}
