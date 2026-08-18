// src/hooks/useGenerate.js
// Manages REST (non-streaming) text generation state.
// Calls POST /api/generate via the service layer.

import { useState, useCallback } from 'react'
import { generate as generateAPI } from '../services/api'

export function useGenerate() {
  const [generatedText, setGeneratedText] = useState('')
  const [isGenerating, setIsGenerating]   = useState(false)
  const [stats, setStats]                 = useState(null)
  const [error, setError]                 = useState(null)

  /**
   * Trigger text generation.
   * @param {{ prompt, maxTokens, temperature, topK, topP, seed }} params
   */
  const generate = useCallback(async (params) => {
    setIsGenerating(true)
    setGeneratedText('')
    setStats(null)
    setError(null)

    try {
      const data = await generateAPI(params)
      setGeneratedText(data.generated_text)
      setStats({
        promptTokens:    data.prompt_tokens,
        tokensGenerated: data.tokens_generated,
        timeMs:          data.generation_time_ms,
        tokensPerSecond: data.tokens_per_second,
      })
    } catch (e) {
      setError(e.message)
    } finally {
      setIsGenerating(false)
    }
  }, [])

  const reset = useCallback(() => {
    setGeneratedText('')
    setStats(null)
    setError(null)
  }, [])

  return { generate, generatedText, isGenerating, stats, error, reset }
}