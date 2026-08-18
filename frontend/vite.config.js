// frontend/vite.config.js
// Vite configuration with backend proxy.
// The proxy eliminates CORS friction in development and avoids
// hardcoded backend URLs anywhere in the frontend source.

import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // HTTP API routes → FastAPI backend
      '/api': {
        target:      'http://localhost:8000',
        changeOrigin: true,
      },
      // WebSocket streaming → FastAPI backend
      '/ws': {
        target:      'ws://localhost:8000',
        ws:           true,   // enable WebSocket proxying
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
  },
})