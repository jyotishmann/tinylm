// src/hooks/useDebounce.js
// Generic debounce hook — used by TokenizePage to avoid calling
// POST /api/tokenize on every keystroke.

import { useState, useEffect } from 'react'

/**
 * Returns a debounced copy of `value` that only updates
 * after `delay` ms of inactivity.
 *
 * @param {any} value - Value to debounce
 * @param {number} delay - Debounce delay in milliseconds
 */
export function useDebounce(value, delay = 300) {
  const [debouncedValue, setDebouncedValue] = useState(value)

  useEffect(() => {
    const timer = setTimeout(() => setDebouncedValue(value), delay)
    return () => clearTimeout(timer)  // Cleanup: cancel pending update on re-render
  }, [value, delay])

  return debouncedValue
}