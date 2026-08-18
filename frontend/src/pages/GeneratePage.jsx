// src/pages/GeneratePage.jsx
// Main page: prompt input + controls + output display.
// Supports REST (full response) and WebSocket streaming modes.

import { useState }             from 'react'
import { useGenerate }          from '../hooks/useGenerate'
import { useStreamGenerate }    from '../hooks/useStreamGenerate'
import PromptInput              from '../components/generation/PromptInput'
import GenerationControls       from '../components/generation/GenerationControls'
import OutputDisplay            from '../components/generation/OutputDisplay'
import GenerationStats          from '../components/generation/GenerationStats'
import Button                   from '../components/shared/Button'
import ErrorToast               from '../components/shared/ErrorToast'
import { DEFAULTS }             from '../utils/constants'

export default function GeneratePage() {
  const [prompt, setPrompt]       = useState('')
  const [mode, setMode]           = useState('stream')  // 'rest' | 'stream'
  const [params, setParams]       = useState(DEFAULTS)

  // REST hook
  const rest = useGenerate()

  // Streaming hook
  const stream = useStreamGenerate()

  // Derived state: whichever mode is active
  const isActive  = mode === 'rest' ? rest.isGenerating  : stream.isStreaming
  const outputText = mode === 'rest' ? rest.generatedText : stream.buffer
  const stats     = mode === 'rest' ? rest.stats         : stream.stats
  const error     = mode === 'rest' ? rest.error         : stream.error

  const handleGenerate = () => {
    if (!prompt.trim()) return
    if (mode === 'rest') {
      rest.generate({ prompt, ...params })
    } else {
      stream.startStream({ prompt, ...params })
    }
  }

  const handleStop = () => {
    if (mode === 'stream') stream.stopStream()
  }

  const handleReset = () => {
    rest.reset()
    // For stream: just start a new one (old buffer cleared on startStream)
  }

  return (
    <div className="max-w-5xl mx-auto flex flex-col gap-6">
      <div>
        <h2 className="text-lg font-semibold text-gray-100">Text Generation</h2>
        <p className="text-sm text-gray-500 mt-0.5">
          Complete a Lovecraftian prompt using the trained GPT model.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left column: input + controls */}
        <div className="lg:col-span-1 flex flex-col gap-4">
          <PromptInput
            value    = {prompt}
            onChange = {setPrompt}
            disabled = {isActive}
          />
          <GenerationControls
            params       = {params}
            onChange     = {setParams}
            mode         = {mode}
            onModeChange = {setMode}
            disabled     = {isActive}
          />
          <div className="flex gap-2">
            <Button
              onClick   = {handleGenerate}
              isLoading = {isActive}
              disabled  = {!prompt.trim() || isActive}
              className = "flex-1"
            >
              {isActive ? 'Generating…' : 'Generate'}
            </Button>
            {isActive && mode === 'stream' && (
              <Button variant="danger" onClick={handleStop}>
                Stop
              </Button>
            )}
          </div>
        </div>

        {/* Right column: output */}
        <div className="lg:col-span-2 flex flex-col gap-3">
          <OutputDisplay
            text      = {outputText}
            isLoading = {isActive}
          />
          <GenerationStats stats={stats} />
          <ErrorToast
            message   = {error}
            onDismiss = {handleReset}
          />
        </div>
      </div>
    </div>
  )
}