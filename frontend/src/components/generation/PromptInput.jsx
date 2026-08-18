// src/components/generation/PromptInput.jsx
// Textarea for the prompt with character counter.

const MAX_CHARS = 1000

export default function PromptInput({ value, onChange, disabled }) {
  const remaining = MAX_CHARS - value.length
  const isNearLimit = remaining < 100

  return (
    <div className="flex flex-col gap-1">
      <div className="flex justify-between items-baseline">
        <label className="text-xs font-medium text-gray-400 uppercase tracking-wider">
          Prompt
        </label>
        <span className={`text-xs font-mono ${isNearLimit ? 'text-yellow-400' : 'text-gray-600'}`}>
          {remaining}/{MAX_CHARS}
        </span>
      </div>
      <textarea
        value       = {value}
        onChange    = {(e) => onChange(e.target.value)}
        disabled    = {disabled}
        maxLength   = {MAX_CHARS}
        rows        = {4}
        placeholder = "The ancient city lay beneath the waves, its cyclopean..."
        className   = "w-full bg-gray-800 border border-gray-700 rounded-lg \
                       px-3 py-2.5 text-sm text-gray-100 placeholder-gray-600 \
                       resize-none focus:outline-none focus:ring-2 \
                       focus:ring-indigo-500 focus:border-transparent \
                       disabled:opacity-50 disabled:cursor-not-allowed \
                       leading-relaxed"
      />
    </div>
  )
}