// src/hooks/useStreamGenerate.js
// Manages WebSocket streaming generation state.
// Yields tokens as they arrive from WS /ws/generate.

import { useState, useCallback, useRef } from 'react'
import { WS_URL } from '../utils/constants'

export function useStreamGenerate() {
  const [buffer, setBuffer]         = useState('')   // accumulated token text
  const [isStreaming, setIsStreaming] = useState(false)
  const [stats, setStats]           = useState(null)
  const [error, setError]           = useState(null)
  const wsRef                        = useRef(null)

  const startStream = useCallback((params) => {
    // Close any existing connection first
    if (wsRef.current) {
      wsRef.current.close()
    }

    // Reset state before opening new connection
    setBuffer('')
    setIsStreaming(true)
    setError(null)
    setStats(null)

    const ws = new WebSocket(WS_URL)
    wsRef.current = ws

    // ── 1. On open: send the generation request ───────────────────────
    ws.onopen = () => {
      ws.send(JSON.stringify({
        prompt:         params.prompt,
        max_new_tokens: params.maxTokens,
        temperature:    params.temperature,
        top_k:          params.topK,
        top_p:          params.topP,
      }))
    }

    // ── 2. On message: handle each frame by type ──────────────────────
    ws.onmessage = (event) => {
      let frame
      try {
        frame = JSON.parse(event.data)
      } catch {
        return  // Malformed frame — skip silently
      }

      switch (frame.type) {
        case 'token':
          // Append token text — React batches these updates automatically
          setBuffer((prev) => prev + frame.token)
          break

        case 'done':
          setIsStreaming(false)
          setStats({
            totalTokens:    frame.total_tokens,
            timeMs:         frame.generation_time_ms,
            tokensPerSecond: frame.total_tokens / (frame.generation_time_ms / 1000),
          })
          ws.close()
          break

        case 'error':
          setError(frame.message)
          setIsStreaming(false)
          ws.close()
          break

        default:
          break
      }
    }

    // ── 3. On error: network or connection failure ─────────────────────
    ws.onerror = () => {
      setError('WebSocket connection failed. Is the backend running on :8000?')
      setIsStreaming(false)
    }

    // ── 4. On close: always fires — even on intentional close ─────────
    ws.onclose = () => {
      setIsStreaming(false)
    }
  }, [])

  // Called by the Stop button
  const stopStream = useCallback(() => {
    wsRef.current?.close()
    setIsStreaming(false)
  }, [])

  return { buffer, isStreaming, stats, error, startStream, stopStream }
}