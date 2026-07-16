import { useEffect, useState } from 'react'
import { getSettings, putSettings } from '../api.js'

// Display names for provider ids; the ids themselves are what LangChain expects.
const PROVIDER_LABELS = {
  openai: 'OpenAI',
  anthropic: 'Claude',
  google_genai: 'Gemini',
  ollama: 'Local (Ollama)',
}

// Common models per provider, best default first. The dropdown also keeps
// whatever model is already saved, so custom models from .env aren't lost.
const PROVIDER_MODELS = {
  anthropic: ['claude-sonnet-5', 'claude-opus-4-8', 'claude-haiku-4-5'],
  openai: ['gpt-5.6-terra', 'gpt-5.6-sol', 'gpt-5.6-luna', 'gpt-5.4-mini'],
  google_genai: ['gemini-3.5-flash', 'gemini-3-pro', 'gemini-2.5-flash', 'gemini-2.5-pro'],
  ollama: ['llama3.2', 'qwen3:8b', 'gemma3:12b', 'mistral', 'deepseek-r1:8b'],
}

export default function Settings({ onClose }) {
  const [settings, setSettings] = useState(null)
  const [provider, setProvider] = useState('') // '' = not chosen yet: model + key stay locked
  const [model, setModel] = useState('')
  const [apiKey, setApiKey] = useState('')
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    getSettings().then((s) => {
      setSettings(s)
      if (s.configured) { // returning user: show their saved choice
        setProvider(s.provider)
        setModel(s.model)
      }
    }).catch((e) => setError(e.message))
  }, [])

  const chooseProvider = (p) => {
    setProvider(p)
    // saved model still applies if it belongs to this provider; otherwise best default
    setModel(settings.configured && p === settings.provider
      ? settings.model
      : PROVIDER_MODELS[p][0])
  }

  const needsKey = provider && provider !== 'ollama'
  const models = provider ? PROVIDER_MODELS[provider].slice() : []
  if (model && !models.includes(model)) models.unshift(model)

  const save = async () => {
    setSaving(true)
    setError(null)
    try {
      const updated = await putSettings({
        provider,
        model,
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
              <select value={provider} onChange={(e) => chooseProvider(e.target.value)}>
                <option value="" disabled>Choose a provider…</option>
                {settings.providers.map((p) => (
                  <option key={p} value={p}>{PROVIDER_LABELS[p] || p}</option>
                ))}
              </select>
            </label>
            <label>
              Model
              <select
                value={model}
                disabled={!provider}
                onChange={(e) => setModel(e.target.value)}
              >
                {!provider && <option value="">Pick a provider first</option>}
                {models.map((m) => <option key={m} value={m}>{m}</option>)}
              </select>
            </label>
            <label>
              API key
              {needsKey && settings.has_key && provider === settings.provider &&
                <span className="hint"> (one is stored — leave blank to keep it)</span>}
              {provider === 'ollama' &&
                <span className="hint"> (runs on your machine — no key needed)</span>}
              <input
                type="password"
                value={apiKey}
                placeholder={needsKey ? 'paste to set or replace' : ''}
                autoComplete="off"
                disabled={!needsKey}
                onChange={(e) => setApiKey(e.target.value)}
              />
            </label>
          </>
        )}
        {error && <p className="error">{error}</p>}
        <div className="modal-actions">
          <button className="ghost" onClick={onClose}>Cancel</button>
          <button className="primary" onClick={save} disabled={!settings || !provider || saving}>
            {saving ? 'Saving…' : 'Save'}
          </button>
        </div>
      </div>
    </div>
  )
}
