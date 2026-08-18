// src/components/shared/ErrorToast.jsx
// Displays error messages from hooks in a dismissible banner.

export default function ErrorToast({ message, onDismiss }) {
  if (!message) return null

  return (
    <div className="flex items-start gap-3 p-3 rounded-lg
                    bg-red-950 border border-red-800 text-red-300 text-sm">
      <span className="shrink-0 mt-0.5">⚠</span>
      <p className="flex-1">{message}</p>
      {onDismiss && (
        <button
          onClick   = {onDismiss}
          className = "shrink-0 text-red-400 hover:text-red-200"
        >
          ✕
        </button>
      )}
    </div>
  )
}