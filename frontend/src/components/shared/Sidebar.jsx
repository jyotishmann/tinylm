// src/components/shared/Sidebar.jsx
// Left navigation — three links: Generate, Tokenize, Attention.

import { NavLink } from 'react-router-dom'

const NAV_ITEMS = [
  { to: '/generate', label: 'Generate',  icon: '⚡' },
  { to: '/tokenize', label: 'Tokenize',  icon: '🔤' },
  { to: '/attention', label: 'Attention', icon: '👁' },
]

export default function Sidebar() {
  return (
    <nav className="w-44 shrink-0 bg-gray-900 border-r border-gray-800
                    flex flex-col py-4 gap-1 px-2">
      {NAV_ITEMS.map(({ to, label, icon }) => (
        <NavLink
          key       = {to}
          to        = {to}
          className = {({ isActive }) =>
            `flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm
             transition-colors
             ${isActive
               ? 'bg-indigo-600 text-white font-medium'
               : 'text-gray-400 hover:text-gray-100 hover:bg-gray-800'
             }`
          }
        >
          <span>{icon}</span>
          <span>{label}</span>
        </NavLink>
      ))}

      {/* Bottom: tiny project info */}
      <div className="mt-auto px-3 py-2 text-xs text-gray-600 leading-snug">
        TinyLM (EMG-01)<br />
        Built from scratch<br />
        ~12.7M parameters
      </div>
    </nav>
  )
}