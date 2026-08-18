// src/components/generation/OutputDisplay.jsx
// Displays generated text — handles both REST (full at once)
// and streaming (buffer that grows token by token).
// The `text` prop handles both cases: REST sets it fully,
// streaming sets it incrementally via the buffer.

export default function OutputDisplay({ text, isLoading, placeholder }) {
  if (!text && !isLoading) {
    return (
      <div className="min-h-32 flex items-center justify-center
                      border border-dashed border-gray-700 rounded-lg">
        <p className="text-sm text-gray-600 italic">
          {placeholder ?? 'Generated text will appear here...'}
        </p>
      </div>
    )
  }

  return (
    <div className="relative">
      <div className="min-h-32 p-4 bg-gray-800 rounded-lg border border-gray-700
                      text-sm text-gray-200 leading-relaxed whitespace-pre-wrap
                      font-mono">
        {text}
        {/* Blinking cursor while streaming */}
        {isLoading && (
          <span className="inline-block w-2 h-4 bg-indigo-400 ml-0.5 animate-pulse
                           align-text-bottom" />
        )}
      </div>
    </div>
  )
}