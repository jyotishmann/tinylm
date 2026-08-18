// src/services/api.js
// Axios instance + all API call functions.
// Components and hooks import these functions, never axios directly.
// All calls use relative URLs — Vite proxy routes them to :8000.

import axios from 'axios'

// ── Axios instance ────────────────────────────────────────────────────
const apiClient = axios.create({
  baseURL: '',         // Relative: Vite proxy handles routing to :8000
  timeout: 60_000,     // 60s — generation can take time on CPU
  headers: { 'Content-Type': 'application/json' },
})

// ── Response interceptor: normalise all errors to {message, code} ─────
// This means every catch(e) in a hook sees e.message — never nested paths.
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    // Try to extract the message from our error schema
    // Backend sends: {"detail": {"code": "...", "message": "..."}}
    // FastAPI validation sends: {"detail": [...]} (array)
    const detail = error.response?.data?.detail
    let message = 'An unexpected error occurred'

    if (typeof detail === 'string') {
      message = detail
    } else if (detail?.message) {
      message = detail.message
    } else if (Array.isArray(detail) && detail[0]?.msg) {
      message = `Validation error: ${detail[0].msg}`
    } else if (error.message) {
      message = error.message
    }

    return Promise.reject({
      message,
      code: error.response?.status ?? 0,
      raw:  error,
    })
  }
)

// ── API functions ─────────────────────────────────────────────────────

/** GET /api/model/info — called once on app load */
export const getModelInfo = () =>
  apiClient.get('/api/model/info').then((r) => r.data)

/**
 * POST /api/generate — synchronous text generation
 * @param {{ prompt, maxTokens, temperature, topK, topP, seed }} params
 */
export const generate = (params) =>
  apiClient
    .post('/api/generate', {
      prompt:         params.prompt,
      max_new_tokens: params.maxTokens,
      temperature:    params.temperature,
      top_k:          params.topK,
      top_p:          params.topP,
      seed:           params.seed ?? null,
    })
    .then((r) => r.data)

/**
 * POST /api/tokenize — tokenise text for the TokenViewer
 * @param {string} text
 */
export const tokenize = (text) =>
  apiClient.post('/api/tokenize', { text }).then((r) => r.data)

/**
 * POST /api/attention — extract attention weights for heatmap
 * @param {string} text
 * @param {number} layer — transformer block index (0-indexed)
 */
export const getAttentionWeights = (text, layer = 0) =>
  apiClient.post('/api/attention', { text, layer }).then((r) => r.data)

export default apiClient