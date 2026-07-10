import { useRef, useState } from 'react'
import { startIngest, uploadFiles } from '../api.js'

export default function Sidebar({
  conversations, activeId, onSelectConversation, onNewConversation, onDeleteConversation,
  files, scope, setScope, ingest, refreshFiles, onOpenSettings,
}) {
  const fileInput = useRef(null)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState(null)

  const toggleScope = (name) =>
    setScope((s) => (s.includes(name) ? s.filter((n) => n !== name) : [...s, name]))

  const handleUpload = async (fileList) => {
    if (!fileList?.length) return
    setError(null)
    setUploading(true)
    try {
      await uploadFiles([...fileList])
      await startIngest()
      await refreshFiles()
    } catch (e) {
      setError(e.message)
    } finally {
      setUploading(false)
      if (fileInput.current) fileInput.current.value = ''
    }
  }

  const rescan = async () => {
    setError(null)
    try {
      await startIngest()
      await refreshFiles()
    } catch (e) {
      setError(e.message)
    }
  }

  return (
    <aside className="sidebar">
      <div className="sidebar-head">
        <h1>Research RAG</h1>
        <button className="ghost" onClick={onOpenSettings} title="Settings">⚙</button>
      </div>

      <section>
        <div className="section-head">
          <h2>Conversations</h2>
          <button className="ghost" onClick={onNewConversation} title="New conversation">＋</button>
        </div>
        <ul className="conversations">
          {conversations.map((c) => (
            <li key={c.id} className={c.id === activeId ? 'active' : ''}>
              <button className="conv-title" onClick={() => onSelectConversation(c.id)}>
                {c.title}
              </button>
              <button className="ghost small" title="Delete"
                      onClick={() => onDeleteConversation(c.id)}>✕</button>
            </li>
          ))}
        </ul>
      </section>

      <section className="grow">
        <div className="section-head">
          <h2>File collection</h2>
          <button className="ghost" onClick={rescan} title="Re-scan corpus folder">⟳</button>
        </div>
        <p className="hint">
          {scope.length ? `Searching ${scope.length} selected file(s)` : 'Searching all files'}
          {scope.length > 0 && (
            <button className="link" onClick={() => setScope([])}>clear</button>
          )}
        </p>
        <ul className="files">
          {files.map((f) => (
            <li key={f.name}>
              <label title={f.name}>
                <input
                  type="checkbox"
                  checked={scope.includes(f.name)}
                  onChange={() => toggleScope(f.name)}
                />
                <span className="file-name">{f.name}</span>
                <span className="chunks">{f.chunks}</span>
              </label>
            </li>
          ))}
          {!files.length && <li className="hint">No documents ingested yet.</li>}
        </ul>
      </section>

      <section>
        <div
          className="dropzone"
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => { e.preventDefault(); handleUpload(e.dataTransfer.files) }}
          onClick={() => fileInput.current?.click()}
        >
          {uploading ? 'Uploading…' : 'Drop PDFs here or click to upload'}
        </div>
        <input ref={fileInput} type="file" accept=".pdf" multiple hidden
               onChange={(e) => handleUpload(e.target.files)} />
        {ingest?.running && (
          <p className="ingest-progress">
            Ingesting {ingest.processed}/{ingest.total}…
          </p>
        )}
        {(error || ingest?.error) && <p className="error">{error || ingest.error}</p>}
      </section>
    </aside>
  )
}
