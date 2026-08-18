// src/utils/constants.js
// Application-wide constants.
// The WebSocket URL is constructed from window.location so it
// works with the Vite proxy in development and with a reverse
// proxy in production without code changes.

export const API_BASE = ''  // empty string → Vite proxy handles routing

export const WS_URL = (() => {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  const host     = window.location.host  // e.g. 'localhost:5173'
  return `${protocol}//${host}/ws/generate`
})()

// Default generation parameters — match backend defaults
export const DEFAULTS = {
  maxTokens:   200,
  temperature: 0.8,
  topK:        50,
  topP:        0.9,
}

// Colour palette for token chips (cycles through 8 colours)
export const TOKEN_COLOURS = [
  'bg-blue-900   text-blue-200',
  'bg-purple-900 text-purple-200',
  'bg-green-900  text-green-200',
  'bg-yellow-900 text-yellow-200',
  'bg-red-900    text-red-200',
  'bg-cyan-900   text-cyan-200',
  'bg-pink-900   text-pink-200',
  'bg-orange-900 text-orange-200',
]