// src/components/attention/AttentionInput.jsx
// Text input + layer selector + submit button for the attention page.

import Button from '../shared/Button'

export default function AttentionInput({
  text, onTextChange,
  layer, onLayerChange,
  maxLayers,
  onSubmit,
  isLoading,
}) {
  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-col gap-1">
        <label className="text-xs font-medium text-gray-400 uppercase tracking-wider">
          Text
        </label>
        <input
          type        = "text"
          value       = {text}
          onChange    = {(e) => onTextChange(e.target.value)}
          disabled    = {isLoading}
          maxLength   = {512}
          placeholder = 'e.g. "The eldritch horror was cyclopean"'
          className   = "bg-gray-800 border border-gray-700 rounded-lg \
                         px-3 py-2 text-sm text-gray-100 placeholder-gray-600 \
                         focus:outline-none focus:ring-2 focus:ring-indigo-500 \
                         focus:border-transparent disabled:opacity-50"
        />
      </div>

      <div className="flex items-end gap-3">
        <div className="flex flex-col gap-1 w-32">
          <label className="text-xs font-medium text-gray-400 uppercase tracking-wider">
            Layer
          </label>
          <select
            value     = {layer}
            onChange  = {(e) => onLayerChange(Number(e.target.value))}
            disabled  = {isLoading}
            className = "bg-gray-800 border border-gray-700 rounded-lg \
                         px-3 py-2 text-sm text-gray-100 \
                         focus:outline-none focus:ring-2 focus:ring-indigo-500 \
                         disabled:opacity-50"
          >
            {Array.from({ length: maxLayers || 6 }, (_, i) => (
              <option key={i} value={i}>Layer {i}</option>
            ))}
          </select>
        </div>

        <Button
          onClick   = {onSubmit}
          isLoading = {isLoading}
          disabled  = {!text.trim() || isLoading}
        >
          Extract
        </Button>
      </div>
    </div>
  )
}