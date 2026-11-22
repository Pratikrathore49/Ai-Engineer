// pages/SearchPage.tsx
// Placeholder — will become the semantic search UI in Task 22.

function SearchPage() {
  return (
    <div>
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-white">Search</h1>
        <p className="text-slate-400 text-sm mt-1">
          Semantic search across all your documents
        </p>
      </div>

      {/* Search bar — static placeholder */}
      <div className="flex items-center gap-3 bg-[#1e293b] border border-[#334155] rounded-xl px-4 py-3 mb-6">
        <span className="text-slate-500">🔍</span>
        <input
          type="text"
          placeholder="Ask anything about your documents..."
          disabled
          className="
            flex-1 bg-transparent text-sm text-slate-300
            placeholder-slate-500 outline-none
            disabled:cursor-not-allowed
          "
        />
      </div>

      <div className="bg-[#1e293b] border border-[#334155] rounded-xl p-5">
        <div className="flex flex-col items-center justify-center py-12 text-center">
          <span className="text-4xl mb-3">🔎</span>
          <p className="text-sm font-medium text-slate-300">Search UI coming soon</p>
          <p className="text-xs text-slate-500 mt-1">
            Full semantic search in the next task
          </p>
        </div>
      </div>
    </div>
  )
}

export default SearchPage
