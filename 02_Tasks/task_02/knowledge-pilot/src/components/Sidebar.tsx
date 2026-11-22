// Sidebar.tsx
// Left navigation — logo, nav links, bottom settings link.
//
// WHY NavLink instead of Link?
//   NavLink automatically adds an "active" class when the route matches.
//   We use that to highlight the current page in the nav.

import { NavLink } from 'react-router-dom'

// Nav items — label + path + icon (emoji for now, swap for icons later)
const NAV_ITEMS = [
  { label: 'Dashboard',     path: '/',          icon: '⬛' },
  { label: 'My Documents',  path: '/documents', icon: '📄' },
  { label: 'Search',        path: '/search',    icon: '🔍' },
]

function Sidebar() {
  return (
    <aside className="
      w-60 min-w-[240px] h-screen
      bg-[#1e293b] border-r border-[#334155]
      flex flex-col
      overflow-hidden
    ">

      {/* Logo */}
      <div className="px-5 py-6 border-b border-[#334155]">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center text-sm font-bold">
            K
          </div>
          <div>
            <p className="text-sm font-semibold text-white leading-tight">KnowledgePilot</p>
            <p className="text-xs text-slate-400">AI</p>
          </div>
        </div>
      </div>

      {/* Navigation links */}
      <nav className="flex-1 px-3 py-4 space-y-1">
        {NAV_ITEMS.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            end={item.path === '/'}
            // end prop on "/" means it only matches exactly "/"
            // without it, "/" would match every route and always be active
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-colors duration-150
              ${isActive
                ? 'bg-blue-600 text-white font-medium'
                : 'text-slate-400 hover:bg-[#334155] hover:text-white'
              }`
            }
          >
            <span className="text-base">{item.icon}</span>
            <span>{item.label}</span>
          </NavLink>
        ))}
      </nav>

      {/* Bottom — Settings */}
      <div className="px-3 py-4 border-t border-[#334155]">
        <NavLink
          to="/settings"
          className={({ isActive }) =>
            `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-colors duration-150
            ${isActive
              ? 'bg-blue-600 text-white font-medium'
              : 'text-slate-400 hover:bg-[#334155] hover:text-white'
            }`
          }
        >
          <span className="text-base">⚙️</span>
          <span>Settings</span>
        </NavLink>
      </div>

    </aside>
  )
}

export default Sidebar
