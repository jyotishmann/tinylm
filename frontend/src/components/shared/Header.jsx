// src/components/shared/Header.jsx
// Top bar: project name, architecture badge, model status.
// Receives modelInfo from App.jsx (fetched once, shared down).

import ModelStatusBadge from './ModelStatusBadge'

export default function Header({ modelInfo, isLoading }) {
  return (
    <header className="flex items-center justify-between px-6 py-3
                        bg-gray-900 border-b border-gray-800 shrink-0">
      {/* Left: project name + tagline */}
      <div className="flex items-center gap-3">
        <div>
          <h1 className="text-base font-semibold tracking-tight text-gray-100">
            TinyLM{' '}
            <span className="text-gray-500 font-normal text-sm">EMG-01</span>
          </h1>
          <p className="text-xs text-gray-500 hidden sm:block">
            GPT trained from scratch · H.P. Lovecraft corpus
          </p>
        </div>
      </div>

      {/* Right: live model status */}
      <ModelStatusBadge modelInfo={modelInfo} isLoading={isLoading} />
    </header>
  )
}