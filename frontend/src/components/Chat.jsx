import { useEffect, useRef, useState } from 'react'
import { ask } from '../api.js'

/** Render answer text with [n] turned into clickable citation chips. */
function AnswerText({ text, sources, onCitationClick }) {
  const parts = text.split(/(\[\d+\])/g)
  return (
    <div className="answer-text">
      {parts.map((part, i) => {
        const m = part.match(/^\[(\d+)\]$/)
        if (!m) return <span key={i}>{part}</span>
        const source = sources?.find((s) => s.n === Number(m[1]))
        if (!source) return <span key={i}>{part}</span>
        return (
          <button key={i} className="citation" title={`${source.pdf} — ${source.headings}`}
                  onClick={() => onCitationClick(source)}>
            {m[1]}
          </button>
        )
      })}
    </div>
  )
}

function SourceList({ sources, onCitationClick }) {
  if (!sources?.length) return null
  return (
    <details className="sources">
      <summary>{sources.length} source{sources.length > 1 ? 's' : ''}</summary>
      <ul>
        {sources.map((s) => (
          <li key={s.n}>
            <button className="citation" onClick={() => onCitationClick(s)}>{s.n}</button>
            <span className="source-meta" title={s.text?.slice(0, 400)}>
              {s.pdf}{s.headings ? ` — ${s.headings}` : ''}
            </span>
          </li>
        ))}
      </ul>
    </details>
  )
}

export default function Chat({ conversation, updateConversation, scope, onCitationClick, ingestRunning }) {
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const bottom = useRef(null)

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: 'smooth' })
  }, [conversation.messages.length, busy])

  const send = async () => {
    const question = input.trim()
    if (!question || busy) return
    setInput('')
    setBusy(true)
    updateConversation((c) => ({
      ...c,
      title: c.messages.length ? c.title : question.slice(0, 60),
      messages: [...c.messages, { role: 'user', text: question, scope: [...scope] }],
    }))
    try {
      const res = await ask(question, scope)
      updateConversation((c) => ({
        ...c,
        messages: [...c.messages, { role: 'assistant', text: res.answer, sources: res.sources }],
      }))
    } catch (e) {
      updateConversation((c) => ({
        ...c,
        messages: [...c.messages, { role: 'error', text: e.message }],
      }))
      setInput((cur) => cur || question) // don't lose the typed question on failure
    } finally {
      setBusy(false)
    }
  }

  return (
    <main className="chat">
      <div className="messages">
        {!conversation.messages.length && (
          <div className="empty">
            Ask a question about your documents. Select files in the sidebar to
            restrict the search; leave them unchecked to search everything.
          </div>
        )}
        {conversation.messages.map((m, i) => (
          <div key={i} className={`message ${m.role}`}>
            {m.role === 'user' && m.scope?.length > 0 && (
              <div className="scope-tag">in {m.scope.length} selected file(s)</div>
            )}
            {m.role === 'assistant'
              ? <AnswerText text={m.text} sources={m.sources} onCitationClick={onCitationClick} />
              : <div className="answer-text">{m.text}</div>}
            {m.role === 'assistant' && (
              <SourceList sources={m.sources} onCitationClick={onCitationClick} />
            )}
          </div>
        ))}
        {busy && <div className="message assistant thinking">Searching the corpus…</div>}
        <div ref={bottom} />
      </div>
      <div className="composer">
        <textarea
          value={input}
          placeholder={ingestRunning ? 'Ingest in progress…' : 'Ask about your documents…'}
          disabled={ingestRunning}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
              e.preventDefault()
              send()
            }
          }}
        />
        <button className="primary" onClick={send} disabled={busy || ingestRunning || !input.trim()}>
          Send
        </button>
      </div>
    </main>
  )
}
