import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import basicSsl from '@vitejs/plugin-basic-ssl' // Import the plugin
const BACKEND_HOST_IP = '192.168.0.54'
// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react(), basicSsl()], // Add the plugin to the plugins array
  server: {
    https: true, // Enable HTTPS
    port: 3000,
    open: true,
    host: true, // Automatically open browser
    proxy: {
      // Proxy WebSocket connections to backend
      '/wss': {
        target: `wss://${BACKEND_HOST_IP}:8000`,
        ws: true,
      },
      // Proxy REST API calls to backend
      '/api': {
        target: `https://${BACKEND_HOST_IP}:8000`,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})
