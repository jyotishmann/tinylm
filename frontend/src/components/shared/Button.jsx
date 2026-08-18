// src/components/shared/Button.jsx
// Reusable button with primary, secondary, and loading variants.

export default function Button({
  children,
  onClick,
  variant   = 'primary',   // 'primary' | 'secondary' | 'danger'
  disabled  = false,
  isLoading = false,
  className = '',
  ...props
}) {
  const base = 'inline-flex items-center justify-center gap-2 px-4 py-2 ' +
               'rounded-lg text-sm font-medium transition-colors ' +
               'focus:outline-none focus:ring-2 focus:ring-offset-2 ' +
               'focus:ring-offset-gray-950 disabled:opacity-50 disabled:cursor-not-allowed'

  const variants = {
    primary:   'bg-indigo-600 hover:bg-indigo-500 text-white focus:ring-indigo-500',
    secondary: 'bg-gray-800 hover:bg-gray-700 text-gray-200 focus:ring-gray-600 border border-gray-700',
    danger:    'bg-red-900 hover:bg-red-800 text-red-100 focus:ring-red-700',
  }

  return (
    <button
      onClick   = {onClick}
      disabled  = {disabled || isLoading}
      className = {`${base} ${variants[variant]} ${className}`}
      {...props}
    >
      {isLoading && (
        <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
          <path className="opacity-75" fill="currentColor"
                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
        </svg>
      )}
      {children}
    </button>
  )
}