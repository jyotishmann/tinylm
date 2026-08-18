// src/utils/format.js
// Formatting utilities for display values.

/** Format a millisecond duration as '1.23s' or '834ms' */
export const formatTime = (ms) => {
  if (!ms && ms !== 0) return '–'
  return ms >= 1000 ? `${(ms / 1000).toFixed(2)}s` : `${Math.round(ms)}ms`
}

/** Format a large integer with commas: 12700000 → '12.7M' */
export const formatParams = (n) => {
  if (!n) return '–'
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`
  if (n >= 1e3) return `${(n / 1e3).toFixed(0)}K`
  return String(n)
}

/** Format tokens per second: 109.7 → '110 tok/s' */
export const formatSpeed = (tps) => {
  if (!tps) return '–'
  return `${Math.round(tps)} tok/s`
}

/** Strip </w> from a token string for display */
export const cleanToken = (t) => t.replace(/<\/w>/g, '')