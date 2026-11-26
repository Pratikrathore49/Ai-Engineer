// src/app/main.tsx
// Entry point — mounts the React tree into #root.

import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { AppProvider } from './providers/AppProvider'
import App from './App'
import '../styles/globals.css'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <AppProvider>
      <App />
    </AppProvider>
  </StrictMode>,
)
