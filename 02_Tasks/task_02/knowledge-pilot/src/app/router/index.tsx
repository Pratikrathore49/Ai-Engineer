// src/app/router/index.tsx
// Defines all routes and wraps every page in the Layout.
// WHY separate from App.tsx?
//   App.tsx only sets up BrowserRouter context.
//   This file handles route definitions.
//   Clean separation — if you add auth guards later,
//   you add them here without touching App.tsx.

import { Routes, Route } from 'react-router-dom'
import { Layout } from '@shared/components/layout/Layout'
import { Dashboard } from '@modules/dashboard'
import { DocumentList, DocumentDetail } from '@modules/documents'
import { SearchPage } from '@modules/search'

export function AppRouter() {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        {/* All pages render inside Layout's <Outlet /> */}
        <Route index element={<Dashboard />} />
        <Route path="documents" element={<DocumentList />} />
        <Route path="documents/:id" element={<DocumentDetail />} />
        <Route path="search" element={<SearchPage />} />
      </Route>
    </Routes>
  )
}
