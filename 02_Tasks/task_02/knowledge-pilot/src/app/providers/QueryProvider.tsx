// src/app/providers/QueryProvider.tsx
// WHY QueryClient here?
//   QueryClient holds the cache for all server state.
//   Creating it at the top level means it lives for the entire
//   app lifetime — cache persists as the user navigates between pages.

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import type { ReactNode } from 'react'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      staleTime: 30_000, // 30 seconds — data is "fresh" for 30s before refetch
    },
  },
})

interface QueryProviderProps {
  children: ReactNode
}

export function QueryProvider({ children }: QueryProviderProps) {
  return (
    <QueryClientProvider client={queryClient}>
      {children}
    </QueryClientProvider>
  )
}
