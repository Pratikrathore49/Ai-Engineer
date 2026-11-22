// pages/DocumentList.tsx
// Placeholder — will list all uploaded documents in Task 22.

function DocumentList() {
  return (
    <div>
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-white">My Documents</h1>
        <p className="text-slate-400 text-sm mt-1">
          Upload and manage your documents
        </p>
      </div>

      <div className="bg-[#1e293b] border border-[#334155] rounded-xl p-5">
        <div className="flex flex-col items-center justify-center py-12 text-center">
          <span className="text-4xl mb-3">📂</span>
          <p className="text-sm font-medium text-slate-300">No documents uploaded</p>
          <p className="text-xs text-slate-500 mt-1">
            Document upload UI coming in the next task
          </p>
        </div>
      </div>
    </div>
  )
}

export default DocumentList
