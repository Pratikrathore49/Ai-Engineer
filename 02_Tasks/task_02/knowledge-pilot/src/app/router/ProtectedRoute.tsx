// src/app/router/ProtectedRoute.tsx
// Placeholder for auth-guarded routes.
// Wrap any <Route> element with this when auth is implemented.

import type { ReactNode } from 'react'

interface ProtectedRouteProps {
  children: ReactNode
}

export function ProtectedRoute({ children }: ProtectedRouteProps) {
  // TODO: check auth state and redirect to /login if unauthenticated
  return <>{children}</>
}
