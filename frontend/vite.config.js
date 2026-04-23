import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/compare': 'http://localhost:8001',
      '/predict': 'http://localhost:8001',
      '/answer':  'http://localhost:8001',
      '/health':  'http://localhost:8001',
    },
  },
})
