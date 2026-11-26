// src/modules/documents/pages/DocumentDetail.tsx
// Placeholder — will show document info + chunks in a later task.

import { useParams } from 'react-router-dom'

function DocumentDetail() {
  const { id } = useParams<{ id: string }>()

  return (
    <div>
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-white">Document Detail</h1>
        <p className="text-slate-400 text-sm mt-1 font-mono">
          ID: {id}
        </p>
      </div>

      <div className="bg-[#1e293b] border border-[#334155] rounded-xl p-5">
        <div className="flex flex-col items-center justify-center py-12 text-center">
          <span className="text-4xl mb-3">📄</span>
          <p className="text-sm font-medium text-slate-300">Document viewer</p>
          <p className="text-xs text-slate-500 mt-1">
            Coming in the next task
          </p>
        </div>
      </div>
    </div>
  )
}

export default DocumentDetail
