// src/pages/TokenizePage.jsx
// BPE tokenisation visualiser.
// 300ms debounce on input → POST /api/tokenize → TokenGrid.

import { useState, useEffect } from 'react'
import { tokenize }            from '../services/api'
import { useDebounce }         from '../hooks/useDebounce'
import TokenizerInput          from '../components/tokenize/TokenizerInput'
import TokenGrid               from '../components/tokenize/TokenGrid'
import ErrorToast              from '../components/shared/ErrorToast'

export default function TokenizePage() {
  const [text, setText]     = useState('')
  const [tokens, setTokens] = useState([])
  const [error, setError]   = useState(null)
  const [isLoading, setIsLoading] = useState(false)

  // Only call API 300ms after user stops typing
  const debouncedText = useDebounce(text, 300)

  useEffect(() => {
    if (!debouncedText.trim()) {
      setTokens([])
      return
    }

    setIsLoading(true)
    setError(null)

    tokenize(debouncedText)
      .then((data) => setTokens(data.tokens))
      .catch((e)   => setError(e.message))
      .finally(()  => setIsLoading(false))
  }, [debouncedText])

  return (
    <div className="max-w-3xl mx-auto flex flex-col gap-6">
      <div>
        <h2 className="text-lg font-semibold text-gray-100">Tokeniser</h2>
        <p className="text-sm text-gray-500 mt-0.5">
          See how the BPE tokeniser splits text into sub-word tokens.
          Each colour is a different token; hover for the token ID.
        </p>
      </div>

      <TokenizerInput value={text} onChange={setText} />

      {/* Results */}
      <div className="min-h-24 p-4 bg-gray-900 rounded-lg border border-gray-800">
        {isLoading && (
          <p className="text-xs text-gray-500 animate-pulse">Tokenising…</p>
        )}
        {!isLoading && tokens.length > 0 && (
          <TokenGrid tokens={tokens} />
        )}
        {!isLoading && !tokens.length && text && (
          <p className="text-xs text-gray-600 italic">No tokens returned</p>
        )}
        {!text && (
          <p className="text-xs text-gray-700 italic">
            Type something above to see its tokenisation
          </p>
        )}
      </div>

      <ErrorToast message={error} onDismiss={() => setError(null)} />

      {/* Educational note */}
      {tokens.length > 0 && (
        <div className="text-xs text-gray-600 space-y-1 border-t border-gray-800 pt-4"> 
          <p>
            <span className="text-gray-400">▪ after a chip</span> means that token
            ends a word (the BPE <code className="text-gray-500">{'</w>'}</code> marker).
          </p>
          <p>
            Token IDs in <span className="font-mono text-gray-500">0–3</span> are \
            special tokens (PAD, UNK, BOS, EOS) and won't appear in normal text. 
          </p>
        </div>
      )}
    </div>
  )
}