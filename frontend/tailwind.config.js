
// frontend/tailwind.config.js
/** @type {import('tailwindcss').Config} */
export default {
  // Scan all JSX files for class names to include in the build
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      // Custom colour for the Lovecraft brand accent
      colors: {
        brand: '#6366f1',  // indigo-500 — used for focus rings, highlights
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
    },
  },
  plugins: [],
}