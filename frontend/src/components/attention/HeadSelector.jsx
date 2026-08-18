// src/components/attention/HeadSelector.jsx
// Horizontal tab strip for selecting which attention head to display.

export default function HeadSelector({ nHeads, selectedHead, onSelect }) {
  return (
    <div className="flex flex-col gap-2">
      <label className="text-xs font-medium text-gray-400 uppercase tracking-wider">
        Attention Head
      </label>
      <div className="flex gap-1 flex-wrap">
        {Array.from({ length: nHeads }, (_, i) => (
          <button
            key       = {i}
            onClick   = {() => onSelect(i)}
            className = {`px-3 py-1 rounded text-xs font-mono transition-colors
                          ${selectedHead === i
                            ? 'bg-indigo-600 text-white'
                            : 'bg-gray-800 text-gray-400 hover:text-gray-200'
                          }`}
          >
            H{i}
          </button>
        ))}
      </div>
    </div>
  )
}