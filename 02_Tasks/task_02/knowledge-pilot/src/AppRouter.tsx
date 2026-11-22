// AppRouter.tsx
// Defines all routes and wraps every page in the Layout.
// WHY separate from App.tsx?
//   App.tsx only sets up BrowserRouter context.
//   AppRouter.tsx handles route definitions.
//   Clean separation — if you add auth guards later,
//   you add them here without touching App.tsx.

import { Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import DocumentList from './pages/DocumentList'
import DocumentDetail from './pages/DocumentDetail'
import SearchPage from './pages/SearchPage'

function AppRouter() {
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

export default AppRouter
