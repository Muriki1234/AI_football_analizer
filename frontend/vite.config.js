import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  // 生产 build 时 strip 掉 console.* 和 debugger —— 之前 console.warn/error 全
  // ship 到客户端，给逆向工程开门，也增加 bundle 体积。保留 console.error 反而
  // 容易让"内部状态全暴露"。如果需要错误上报上 Sentry / GlitchTip。
  esbuild: {
    drop: ['console', 'debugger'],
  },
  build: {
    sourcemap: false,   // 显式关闭，防未来配置变更把 source 泄漏到 prod
  },
  server: {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        secure: false,
      },
      '/artifacts': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        secure: false,
      },
    },
  },
})
