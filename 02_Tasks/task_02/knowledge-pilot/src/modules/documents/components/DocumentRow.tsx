// src/modules/documents/components/DocumentRow.tsx
// Renders a single row in the document list.
// Delegates polling to useDocumentPolling, formatting to formatBytes.

import { Layers } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { useDocumentPolling } from '../hooks/useDocumentPolling'
import type { Document } from '../types'
import { formatBytes } from '@shared/utils/format'
import { FileIcon } from './FileIcon'
import { StatusBadge } from './StatusBadge'

interface DocumentRowProps {
  doc: Document
}

export function DocumentRow({ doc }: DocumentRowProps) {
  const navigate = useNavigate()
  useDocumentPolling(doc)

  return (
    <div className="flex items-center gap-4 px-4 py-3 hover:bg-[#334155]/30 transition-colors rounded-lg group">
      <FileIcon fileType={doc.file_type} />

      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium text-white truncate">{doc.filename}</p>
        <p className="text-xs text-slate-500 mt-0.5">{formatBytes(doc.file_size_bytes)}</p>
      </div>

      <StatusBadge status={doc.status} />

      {doc.status === 'ready' && doc.chunk_count !== undefined && (
        <span className="flex items-center gap-1 text-xs text-slate-400">
          <Layers size={12} />
          {doc.chunk_count} chunks
        </span>
      )}

      {doc.status === 'ready' && (
        <button
          onClick={() => navigate(`/documents/${doc.document_id}`)}
          className="
            text-xs text-blue-400 hover:text-blue-300
            opacity-0 group-hover:opacity-100
            transition-opacity px-2 py-1
            border border-blue-400/30 rounded-md
          "
        >
          View
        </button>
      )}
    </div>
  )
}
