// src/app/App.tsx
// Application root — sets up BrowserRouter context and global UI (Toaster).
// All providers live in AppProvider; all routes live in AppRouter.

import { BrowserRouter } from 'react-router-dom'
import { Toaster } from 'sonner'
import { AppRouter } from './router'

function App() {
  return (
    <>
      <Toaster position="top-right" richColors />
      <BrowserRouter>
        <AppRouter />
      </BrowserRouter>
    </>
  )
}

export default App
