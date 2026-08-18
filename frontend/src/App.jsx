// src/App.jsx
// Root component: layout shell + routing.
// useModelInfo() is called here so modelInfo is fetched once
// and passed down to Header and AttentionPage via props.

import { HashRouter, Routes, Route, Navigate } from 'react-router-dom'
import Header       from './components/shared/Header'
import Sidebar      from './components/shared/Sidebar'
import GeneratePage from './pages/GeneratePage'
import TokenizePage from './pages/TokenizePage'
import AttentionPage from './pages/AttentionPage'
import { useModelInfo } from './hooks/useModelInfo'

export default function App() {
  // Fetch model info once — passed to Header and AttentionPage
  const { modelInfo, isLoading } = useModelInfo()

  return (
    <HashRouter>
      <div className="flex flex-col h-screen overflow-hidden bg-gray-950 text-gray-100">
        {/* Top bar */}
        <Header modelInfo={modelInfo} isLoading={isLoading} />

        {/* Body: sidebar + page content */}
        <div className="flex flex-1 overflow-hidden">
          <Sidebar />

          <main className="flex-1 overflow-y-auto p-6">
            <Routes>
              <Route path="/"          element={<Navigate to="/generate" replace />} />
              <Route path="/generate"  element={<GeneratePage />} />
              <Route path="/tokenize"  element={<TokenizePage />} />
              <Route path="/attention" element={<AttentionPage modelInfo={modelInfo} />} />
            </Routes>
          </main>
        </div>
      </div>
    </HashRouter>
  )
}