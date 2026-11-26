// src/app/providers/AppProvider.tsx
// Composes all app-level providers in one place.
// Add ThemeProvider, AuthProvider, etc. here as the app grows.

import type { ReactNode } from 'react'
import { QueryProvider } from './QueryProvider'

interface AppProviderProps {
  children: ReactNode
}

export function AppProvider({ children }: AppProviderProps) {
  return (
    <QueryProvider>
      {children}
    </QueryProvider>
  )
}
