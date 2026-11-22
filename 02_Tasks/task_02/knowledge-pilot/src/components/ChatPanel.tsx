// ChatPanel.tsx
// Right-side panel — will become the RAG chat interface in Task 22.
// For now: shell with header + empty message area + input bar.
//
// WHY build the shell now?
//   The layout needs the 3 columns to exist.
//   It's easier to build the full chat UI in Task 22 when the
//   shell is already positioned and sized correctly.

function ChatPanel() {
  return (
    <aside className="
      w-80 min-w-[320px] h-screen
      bg-[#1e293b] border-l border-[#334155]
      flex flex-col
      overflow-hidden
    ">

      {/* Header */}
      <div className="px-4 py-4 border-b border-[#334155]">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 bg-green-400 rounded-full" />
          <h2 className="text-sm font-semibold text-white">Document Assistant</h2>
        </div>
        <p className="text-xs text-slate-400 mt-0.5">
          Ask questions about your documents
        </p>
      </div>

      {/* Messages area — empty for now */}
      <div className="flex-1 overflow-y-auto px-4 py-4">
        {/* Empty state */}
        <div className="h-full flex flex-col items-center justify-center text-center">
          <div className="w-12 h-12 bg-[#334155] rounded-full flex items-center justify-center mb-3">
            <span className="text-xl">💬</span>
          </div>
          <p className="text-sm text-slate-400 font-medium">No messages yet</p>
          <p className="text-xs text-slate-500 mt-1">
            Upload a document and ask a question
          </p>
        </div>
      </div>

      {/* Input bar — static for now */}
      <div className="px-4 py-4 border-t border-[#334155]">
        <div className="flex items-center gap-2 bg-[#0f172a] rounded-lg px-3 py-2.5 border border-[#334155]">
          <input
            type="text"
            placeholder="Ask a question..."
            disabled
            className="
              flex-1 bg-transparent text-sm text-slate-300
              placeholder-slate-500 outline-none
              disabled:cursor-not-allowed
            "
          />
          <button
            disabled
            className="
              w-7 h-7 bg-blue-600 rounded-md flex items-center justify-center
              disabled:opacity-40 disabled:cursor-not-allowed
              text-xs
            "
          >
            ↑
          </button>
        </div>
        <p className="text-xs text-slate-600 mt-2 text-center">
          Connect a document to start chatting
        </p>
      </div>

    </aside>
  )
}

export default ChatPanel
