// End-to-end smoke: seeds a conversation with a real provenance box, clicks the
// citation, and asserts the PDF viewer draws a highlight. Requires the backend
// on :8642 (with ReAct.pdf ingested) and `npm run dev` on :5173.
//   node e2e/smoke.mjs
// Set PW_CHROMIUM to a Chromium binary if Playwright's own download is absent.
import { chromium } from 'playwright'

const launchOptions = process.env.PW_CHROMIUM ? { executablePath: process.env.PW_CHROMIUM } : {}

const conversation = [{
  id: 'seed-1',
  title: 'What does Figure 1 of the ReAct paper show?',
  messages: [
    { role: 'user', text: 'What does Figure 1 of the ReAct paper show?', scope: [] },
    {
      role: 'assistant',
      text: 'Figure 1 compares four prompting methods [1].',
      sources: [{
        n: 1,
        chunk_id: 'seed-chunk',
        pdf: 'ReAct.pdf',
        headings: '1 INTRODUCTION',
        text: 'Figure 1: (1) Comparison of 4 prompting methods...',
        boxes: [[2, 108.0, 375.2, 505.2, 322.8]],
      }],
    },
  ],
}]

const browser = await chromium.launch(launchOptions)
try {
  const page = await browser.newPage({ viewport: { width: 1500, height: 950 } })
  const errors = []
  page.on('pageerror', (err) => errors.push(err.message))

  await page.goto('http://127.0.0.1:5173')
  await page.evaluate((conv) => localStorage.setItem('rag-conversations-v1', JSON.stringify(conv)), conversation)
  await page.reload()
  await page.waitForSelector('.files li label', { timeout: 15000 })

  await page.click('.answer-text .citation')
  await page.waitForSelector('.pdf-page canvas', { timeout: 30000 })
  await page.waitForSelector('.highlight', { timeout: 30000 })
  const box = await page.locator('.highlight').first().boundingBox()
  if (!box || box.width < 50 || box.height < 10) throw new Error(`implausible highlight box: ${JSON.stringify(box)}`)
  if (errors.length) throw new Error(`page errors: ${errors.join(' | ')}`)
  console.log('SMOKE OK — highlight box:', JSON.stringify(box))
} finally {
  await browser.close()
}
