// Layout.tsx
// The 3-column shell that wraps every page.
//
// WHY <Outlet />?
//   React Router's Outlet renders whichever child route matched.
//   So when the user goes to /documents, the Layout renders with
//   DocumentList inside the Outlet — Sidebar and ChatPanel stay fixed.
//
// Column widths:
//   Sidebar   → fixed 240px, never shrinks
//   Main area → flex-1 (takes all remaining space)
//   ChatPanel → fixed 320px, never shrinks

import { Outlet } from 'react-router-dom'
import Sidebar from './Sidebar'
import ChatPanel from './ChatPanel'

function Layout() {
  return (
    <div className="flex h-screen bg-[#0f172a] text-white overflow-hidden">

      {/* LEFT — Sidebar navigation, fixed width */}
      <Sidebar />

      {/* CENTER — Main content area, scrollable */}
      <main className="flex-1 overflow-y-auto p-6">
        <Outlet />
      </main>

      {/* RIGHT — Chat panel, fixed width */}
      <ChatPanel />

    </div>
  )
}

export default Layout
