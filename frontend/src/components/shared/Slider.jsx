// src/components/shared/Slider.jsx
// Labelled range input with live value display.
// Used by GenerationControls for temperature, top-k, top-p, max-tokens.

export default function Slider({
  label,
  value,
  onChange,
  min  = 0,
  max  = 1,
  step = 0.01,
  formatValue = (v) => v,  // Optional display formatter
}) {
  return (
    <div className="flex flex-col gap-1">
      <div className="flex justify-between items-baseline">
        <label className="text-xs font-medium text-gray-400 uppercase tracking-wider">
          {label}
        </label>
        <span className="text-xs font-mono text-gray-300">
          {formatValue(value)}
        </span>
      </div>
      <input
        type     = "range"
        min      = {min}
        max      = {max}
        step     = {step}
        value    = {value}
        onChange = {(e) => onChange(Number(e.target.value))}
        className = "w-full h-1.5 rounded-full appearance-none cursor-pointer bg-gray-700 accent-indigo-500"
      />
    </div>
  )
}
