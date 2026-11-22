// pages/Dashboard.tsx
// Placeholder — will show stats and recent activity in a later task.

function Dashboard() {
  return (
    <div>
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-white">Dashboard</h1>
        <p className="text-slate-400 text-sm mt-1">
          Your document intelligence overview
        </p>
      </div>

      {/* Stat cards — placeholder */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        {[
          { label: 'Documents',    value: '—', icon: '📄' },
          { label: 'Chunks',       value: '—', icon: '🧩' },
          { label: 'Searches',     value: '—', icon: '🔍' },
        ].map((stat) => (
          <div
            key={stat.label}
            className="bg-[#1e293b] border border-[#334155] rounded-xl p-4"
          >
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs text-slate-400 font-medium">{stat.label}</span>
              <span className="text-lg">{stat.icon}</span>
            </div>
            <p className="text-2xl font-bold text-white">{stat.value}</p>
          </div>
        ))}
      </div>

      {/* Recent activity — placeholder */}
      <div className="bg-[#1e293b] border border-[#334155] rounded-xl p-5">
        <h2 className="text-sm font-semibold text-white mb-4">Recent Documents</h2>
        <div className="flex flex-col items-center justify-center py-8 text-center">
          <span className="text-3xl mb-2">📭</span>
          <p className="text-sm text-slate-400">No documents yet</p>
          <p className="text-xs text-slate-500 mt-1">
            Upload your first document to get started
          </p>
        </div>
      </div>
    </div>
  )
}

export default Dashboard
