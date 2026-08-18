// src/components/generation/GenerationStats.jsx
// Shows token count, speed, and time after generation completes.

import { formatTime, formatSpeed } from '../../utils/format'

export default function GenerationStats({ stats }) {
  if (!stats) return null

  const items = [
    { label: 'Generated',    value: `${stats.tokensGenerated ?? stats.totalTokens} tokens` },
    { label: 'Speed',        value: formatSpeed(stats.tokensPerSecond) },
    { label: 'Time',         value: formatTime(stats.timeMs) },
    ...(stats.promptTokens != null
      ? [{ label: 'Prompt', value: `${stats.promptTokens} tokens` }]
      : []),
  ]

  return (
    <div className="flex flex-wrap gap-x-6 gap-y-1">
      {items.map(({ label, value }) => (
        <div key={label} className="flex items-baseline gap-1.5">
          <span className="text-xs text-gray-500 uppercase tracking-wider">{label}</span>
          <span className="text-xs font-mono text-gray-300">{value}</span>
        </div>
      ))}
    </div>
  )
}