// src/components/tokenize/TokenizerInput.jsx
// Textarea for the tokenizer page. Simpler than PromptInput —
// no character limit here (the backend limits to 5000 chars).

export default function TokenizerInput({ value, onChange }) {
  return (
    <div className="flex flex-col gap-1">
      <label className="text-xs font-medium text-gray-400 uppercase tracking-wider">
        Text to tokenise
      </label>
      <textarea
        value       = {value}
        onChange    = {(e) => onChange(e.target.value)}
        rows        = {3}
        maxLength   = {5000}
        placeholder = 'Type any text — try "eldritch" or "non-Euclidean"'
        className   = "w-full bg-gray-800 border border-gray-700 rounded-lg \
                       px-3 py-2.5 text-sm text-gray-100 placeholder-gray-600 \
                       resize-none focus:outline-none focus:ring-2 \
                       focus:ring-indigo-500 focus:border-transparent leading-relaxed"
      />
      <p className="text-xs text-gray-600">
        Results update 300ms after you stop typing
      </p>
    </div>
  )
}