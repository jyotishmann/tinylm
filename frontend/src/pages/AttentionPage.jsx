// src/pages/AttentionPage.jsx
// Attention weight visualisation page.
// POST /api/attention → D3 heatmap for each head.

import { useState }          from 'react'
import { getAttentionWeights } from '../services/api'
import AttentionInput        from '../components/attention/AttentionInput'
import HeadSelector          from '../components/attention/HeadSelector'
import AttentionHeatmap      from '../components/attention/AttentionHeatmap'
import ErrorToast            from '../components/shared/ErrorToast'

export default function AttentionPage({ modelInfo }) {
  const nLayers = modelInfo?.n_layers ?? 6
  const nHeads  = modelInfo?.n_heads  ?? 6

  const [text, setText]               = useState('The eldritch horror was cyclopean')
  const [layer, setLayer]             = useState(0)
  const [selectedHead, setSelectedHead] = useState(0)
  const [attnData, setAttnData]       = useState(null)
  const [isLoading, setIsLoading]     = useState(false)
  const [error, setError]             = useState(null)

  const handleExtract = () => {
    if (!text.trim()) return
    setIsLoading(true)
    setError(null)

    getAttentionWeights(text, layer)
      .then(setAttnData)
      .catch((e) => setError(e.message))
      .finally(() => setIsLoading(false))
  }

  return (
    <div className="max-w-5xl mx-auto flex flex-col gap-6">
      <div>
        <h2 className="text-lg font-semibold text-gray-100">
          Attention Visualiser
        </h2>
        <p className="text-sm text-gray-500 mt-0.5">
          See what each attention head attends to. Darker cells = higher weight.
          Rows are query positions, columns are key positions.
        </p>
      </div>

      {/* Input controls */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
        <AttentionInput
          text         = {text}
          onTextChange = {setText}
          layer        = {layer}
          onLayerChange = {setLayer}
          maxLayers    = {nLayers}
          onSubmit     = {handleExtract}
          isLoading    = {isLoading}
        />
      </div>

      <ErrorToast message={error} onDismiss={() => setError(null)} />

      {/* Results */}
      {attnData && (
        <div className="flex flex-col gap-4">
          <HeadSelector
            nHeads       = {nHeads}
            selectedHead = {selectedHead}
            onSelect     = {setSelectedHead}
          />

          <div className="bg-gray-900 border border-gray-800 rounded-xl p-5
                          overflow-x-auto">
            {/* Layer / head info */}
            <p className="text-xs text-gray-600 mb-4">
              Layer {attnData.layer} · Head {selectedHead} ·{' '}
              {attnData.tokens.length} tokens
            </p>

            <AttentionHeatmap
              weights      = {attnData.weights}
              tokens       = {attnData.tokens}
              selectedHead = {selectedHead}
            />
          </div>

          {/* Interpretation note */}
          <p className="text-xs text-gray-600">
            <span className="text-gray-400">Reading the heatmap:</span>{' '}
            Row i, column j shows how much position i attends to position j.
            Dark = low weight, bright indigo = high weight. Rows sum to 1.
            Early layers tend to show local patterns; later layers show
            longer-range semantic relationships.
          </p>
        </div>
      )}

      {/* Prompt to run first extraction */}
      {!attnData && !isLoading && (
        <div className="flex items-center justify-center min-h-32
                        border border-dashed border-gray-800 rounded-xl">
          <p className="text-sm text-gray-600 italic">
            Enter text and click Extract to see attention weights
          </p>
        </div>
      )}
    </div>
  )
}