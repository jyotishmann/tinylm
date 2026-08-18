// src/hooks/useModelInfo.js
// Fetches model metadata once on mount.
// Result passed to Header and AttentionPage (layer/head count).

import { useState, useEffect } from 'react'
import { getModelInfo } from '../services/api'

export function useModelInfo() {
  const [modelInfo, setModelInfo] = useState(null)
  const [isLoading, setIsLoading]  = useState(true)
  const [error, setError]          = useState(null)

  useEffect(() => {
    let cancelled = false  // Prevents state update on unmounted component

    getModelInfo()
      .then((data) => { if (!cancelled) setModelInfo(data) })
      .catch((e)   => { if (!cancelled) setError(e.message) })
      .finally(()  => { if (!cancelled) setIsLoading(false) })

    return () => { cancelled = true }
  }, [])  // Empty deps: runs once on mount

  return { modelInfo, isLoading, error }
}