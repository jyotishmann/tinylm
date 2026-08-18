// src/components/generation/GenerationControls.jsx
// Temperature, top-k, top-p, max-tokens sliders + REST/Stream mode toggle.

import Slider from '../shared/Slider'

export default function GenerationControls({
  params,
  onChange,
  mode,          // 'rest' | 'stream'
  onModeChange,
  disabled,
}) {
  const set = (key) => (value) => onChange({ ...params, [key]: value })

  return (
    <div className="flex flex-col gap-4">
      {/* Mode toggle */}
      <div className="flex items-center gap-1 p-0.5 bg-gray-800 rounded-lg w-fit">
        {['rest', 'stream'].map((m) => (
          <button
            key       = {m}
            onClick   = {() => onModeChange(m)}
            disabled  = {disabled}
            className = {`px-3 py-1.5 rounded-md text-xs font-medium transition-colors
                          ${mode === m
                            ? 'bg-indigo-600 text-white'
                            : 'text-gray-400 hover:text-gray-200'
                          } disabled:opacity-50`}
          >
            {m === 'rest' ? '⚡ REST' : '📡 Stream'}
          </button>
        ))}
      </div>

      {/* Sliders */}
      <Slider
        label       = "Temperature"
        value       = {params.temperature}
        onChange    = {set('temperature')}
        min         = {0.01}
        max         = {2.0}
        step        = {0.01}
        formatValue = {(v) => v.toFixed(2)}
      />
      <Slider
        label       = "Max Tokens"
        value       = {params.maxTokens}
        onChange    = {set('maxTokens')}
        min         = {1}
        max         = {500}
        step        = {1}
        formatValue = {(v) => String(v)}
      />
      <Slider
        label       = "Top-k"
        value       = {params.topK}
        onChange    = {set('topK')}
        min         = {0}
        max         = {200}
        step        = {1}
        formatValue = {(v) => v === 0 ? 'off' : String(v)}
      />
      <Slider
        label       = "Top-p"
        value       = {params.topP}
        onChange    = {set('topP')}
        min         = {0}
        max         = {1}
        step        = {0.01}
        formatValue = {(v) => v.toFixed(2)}
      />
    </div>
  )
}