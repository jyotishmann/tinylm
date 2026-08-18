// src/components/tokenize/TokenGrid.jsx
// Renders BPE tokens as coloured chips.
// Each chip shows the token text; hover shows the token ID.

import { TOKEN_COLOURS } from '../../utils/constants'
import { cleanToken }    from '../../utils/format'

export default function TokenGrid({ tokens }) {
  if (!tokens || tokens.length === 0) return null

  return (
    <div className="flex flex-col gap-3">
      {/* Summary */}
      <p className="text-xs text-gray-500">
        <span className="font-mono text-gray-300">{tokens.length}</span> tokens
      </p>

      {/* Token chips */}
      <div className="flex flex-wrap gap-1.5">
        {tokens.map((token, idx) => {
          const colourClass = TOKEN_COLOURS[idx % TOKEN_COLOURS.length]
          const display     = cleanToken(token.text)   // strip </w>
          const hasEow      = token.text.includes('</w>')

          return (
            <div
              key       = {`${token.id}-${idx}`}
              title     = {`ID: ${token.id}  |  raw: "${token.text}"`}
              className = {`relative inline-flex items-center gap-1 px-2 py-0.5
                            rounded text-xs font-mono cursor-default
                            select-none group ${colourClass}`}
            >
              {/* Token text */}
              <span>{display || '·'}</span>

              {/* Word-boundary marker — small dot shows </w> position */}
              {hasEow && (
                <span className="text-current opacity-30 text-[8px]">▪</span>
              )}

              {/* Hover tooltip showing token ID */}
              <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1
                              hidden group-hover:block z-10
                              bg-gray-900 border border-gray-700
                              text-gray-300 text-[10px] font-mono
                              px-2 py-0.5 rounded whitespace-nowrap shadow-lg">
                id={token.id}
              </div>
            </div>
          )
        })}
      </div>

      {/* Token ID list — for the interview ("what IDs does this encode to?") */}
      <div className="font-mono text-xs text-gray-600 break-all">
        [{tokens.map((t) => t.id).join(', ')}]
      </div>
    </div>
  )
}