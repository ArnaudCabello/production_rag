import { useEffect, useMemo, useRef, useState } from 'react'
import { Document, Page, pdfjs } from 'react-pdf'
import { pdfUrl } from '../api.js'

// Bundle the worker locally — the desktop app must not fetch from a CDN.
pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  'pdfjs-dist/build/pdf.worker.min.mjs', import.meta.url,
).toString()

/**
 * source.boxes: [[page, l, t, r, b], ...] in PDF points, origin bottom-left
 * (docling's convention). CSS wants top-left, so y flips against page height.
 */
export default function PdfPanel({ source, onClose }) {
  const container = useRef(null)
  const pageRefs = useRef({})
  const [numPages, setNumPages] = useState(null)
  const [pageDims, setPageDims] = useState({}) // pageNumber -> {width, height} at scale 1
  const [panelWidth, setPanelWidth] = useState(600)

  useEffect(() => {
    const el = container.current
    if (!el) return
    // quantized so a 1px drag doesn't re-rasterize every page canvas
    const measure = () => setPanelWidth(Math.max(320, Math.round((el.clientWidth - 24) / 16) * 16))
    measure()
    const obs = new ResizeObserver(measure)
    obs.observe(el)
    return () => obs.disconnect()
  }, [])

  // a different document means different page sizes — never reuse cached dims
  useEffect(() => {
    setPageDims({})
    setNumPages(null)
  }, [source.pdf])

  const boxesByPage = useMemo(() => {
    const byPage = {}
    for (const [page, l, t, r, b] of source.boxes || []) {
      ;(byPage[page] = byPage[page] || []).push({ l, t, r, b })
    }
    return byPage
  }, [source])

  const firstPage = useMemo(() => {
    const pages = Object.keys(boxesByPage).map(Number)
    return pages.length ? Math.min(...pages) : 1
  }, [boxesByPage])

  // scroll the first highlighted page into view once it has rendered
  const [scrolled, setScrolled] = useState(false)
  useEffect(() => { setScrolled(false) }, [source.chunk_id, source.pdf])
  useEffect(() => {
    if (scrolled || !pageDims[firstPage]) return
    pageRefs.current[firstPage]?.scrollIntoView({ block: 'start' })
    setScrolled(true)
  }, [pageDims, firstPage, scrolled])

  const onPageLoad = (page) => {
    const { width, height } = page.getViewport({ scale: 1 })
    setPageDims((d) => (d[page.pageNumber] ? d : { ...d, [page.pageNumber]: { width, height } }))
  }

  return (
    <aside className="pdf-panel" ref={container}>
      <div className="pdf-head">
        <span className="pdf-title" title={source.pdf}>{source.pdf}</span>
        <button className="ghost" onClick={onClose} title="Close viewer">✕</button>
      </div>
      {!(source.boxes || []).length && (
        <div className="pdf-notice">This source has no location data — showing the document from page 1.</div>
      )}
      <div className="pdf-scroll">
        <Document
          file={pdfUrl(source.pdf)}
          onLoadSuccess={({ numPages: n }) => setNumPages(n)}
          loading={<div className="hint">Loading PDF…</div>}
          error={<div className="error">Could not load the PDF.</div>}
        >
          {Array.from({ length: numPages || 0 }, (_, i) => {
            const pageNumber = i + 1
            const dims = pageDims[pageNumber]
            const scale = dims ? panelWidth / dims.width : 1
            return (
              <div
                key={pageNumber}
                className="pdf-page"
                ref={(el) => { pageRefs.current[pageNumber] = el }}
              >
                <Page
                  pageNumber={pageNumber}
                  width={panelWidth}
                  onLoadSuccess={onPageLoad}
                  renderTextLayer={false}
                  renderAnnotationLayer={false}
                />
                {dims && (boxesByPage[pageNumber] || []).map((box, j) => (
                  <div
                    key={j}
                    className="highlight"
                    style={{
                      left: box.l * scale,
                      top: (dims.height - box.t) * scale,
                      width: (box.r - box.l) * scale,
                      height: (box.t - box.b) * scale,
                    }}
                  />
                ))}
              </div>
            )
          })}
        </Document>
      </div>
    </aside>
  )
}
