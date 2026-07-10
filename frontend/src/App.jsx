import { useCallback, useEffect, useState } from 'react'
import Sidebar from './components/Sidebar.jsx'
import Chat from './components/Chat.jsx'
import PdfPanel from './components/PdfPanel.jsx'
import Settings from './components/Settings.jsx'
import { getFiles, getStatus } from './api.js'

const STORE_KEY = 'rag-conversations-v1'

function loadConversations() {
  try {
    const parsed = JSON.parse(localStorage.getItem(STORE_KEY))
    if (Array.isArray(parsed) && parsed.length) return parsed
  } catch { /* fresh start */ }
  return [{ id: crypto.randomUUID(), title: 'New conversation', messages: [] }]
}

const initialConversations = loadConversations()

export default function App() {
  const [conversations, setConversations] = useState(initialConversations)
  const [activeId, setActiveId] = useState(initialConversations[0].id)
  const [files, setFiles] = useState([])
  const [scope, setScope] = useState([]) // selected document names; empty = search all
  const [ingest, setIngest] = useState(null)
  const [showSettings, setShowSettings] = useState(false)
  // the source being viewed: {pdf, boxes, chunk_id, ...} or null (panel hidden)
  const [activeSource, setActiveSource] = useState(null)

  useEffect(() => {
    localStorage.setItem(STORE_KEY, JSON.stringify(conversations))
  }, [conversations])

  const refreshFiles = useCallback(async () => {
    try {
      const [fileList, status] = await Promise.all([getFiles(), getStatus()])
      setFiles(fileList)
      setIngest(status.ingest)
    } catch (e) {
      // keep the previous running state: one transient failure must not stop
      // the ingest poll or re-enable the composer mid-ingest
      setIngest((prev) => ({ ...(prev || { running: false }), error: `Backend unreachable: ${e.message}` }))
    }
  }, [])

  useEffect(() => { refreshFiles() }, [refreshFiles])

  // poll while an ingest runs so progress and the file list stay live
  useEffect(() => {
    if (!ingest?.running) return
    const timer = setInterval(refreshFiles, 2000)
    return () => clearInterval(timer)
  }, [ingest?.running, refreshFiles])

  const active = conversations.find((c) => c.id === activeId) || conversations[0]

  const updateActive = (updater) =>
    setConversations((all) => all.map((c) => (c.id === active.id ? updater(c) : c)))

  const newConversation = () => {
    const conv = { id: crypto.randomUUID(), title: 'New conversation', messages: [] }
    setConversations((all) => [conv, ...all])
    setActiveId(conv.id)
    setActiveSource(null)
  }

  const deleteConversation = (id) => {
    setConversations((all) => {
      const rest = all.filter((c) => c.id !== id)
      return rest.length ? rest : [{ id: crypto.randomUUID(), title: 'New conversation', messages: [] }]
    })
    if (id === activeId) setActiveSource(null)
  }

  useEffect(() => {
    if (!conversations.some((c) => c.id === activeId)) setActiveId(conversations[0].id)
  }, [conversations, activeId])

  return (
    <div className={`app ${activeSource ? 'with-viewer' : ''}`}>
      <Sidebar
        conversations={conversations}
        activeId={active.id}
        onSelectConversation={(id) => { setActiveId(id); setActiveSource(null) }}
        onNewConversation={newConversation}
        onDeleteConversation={deleteConversation}
        files={files}
        scope={scope}
        setScope={setScope}
        ingest={ingest}
        refreshFiles={refreshFiles}
        onOpenSettings={() => setShowSettings(true)}
      />
      <Chat
        key={active.id}  /* per-conversation instance: busy state can't leak across chats */
        conversation={active}
        updateConversation={updateActive}
        scope={scope}
        onCitationClick={setActiveSource}
        ingestRunning={!!ingest?.running}
      />
      {activeSource && (
        <PdfPanel source={activeSource} onClose={() => setActiveSource(null)} />
      )}
      {showSettings && <Settings onClose={() => setShowSettings(false)} />}
    </div>
  )
}
