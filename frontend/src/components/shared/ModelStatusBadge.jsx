// src/components/shared/ModelStatusBadge.jsx
// Green dot + param count when model is loaded.
// Red dot + "connecting..." when model data hasn't arrived.

import { formatParams } from '../../utils/format'

export default function ModelStatusBadge({ modelInfo, isLoading }) {
  if (isLoading) {
    return (
      <div className="flex items-center gap-2 text-sm text-gray-400">
        <span className="inline-block w-2 h-2 rounded-full bg-yellow-400 animate-pulse" />
        connecting...
      </div>
    )
  }

  if (!modelInfo) {
    return (
      <div className="flex items-center gap-2 text-sm text-red-400">
        <span className="inline-block w-2 h-2 rounded-full bg-red-500" />
        backend offline
      </div>
    )
  }

  return (
    <div className="flex items-center gap-2 text-sm text-gray-300">
      <span className="inline-block w-2 h-2 rounded-full bg-emerald-400" />
      <span className="font-mono text-emerald-300">
        {formatParams(modelInfo.n_params)} params
      </span>
      <span className="text-gray-600">·</span>
      <span className="text-gray-400">
        {modelInfo.n_layers}L · {modelInfo.n_heads}H · d{modelInfo.n_embd}
      </span>
    </div>
  )
}