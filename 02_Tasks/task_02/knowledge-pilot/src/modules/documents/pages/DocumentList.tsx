// src/modules/documents/pages/DocumentList.tsx
// Page-level orchestrator — only composes pieces together.
// No formatting logic, no mutation logic, no polling logic lives here.

import { useQuery } from '@tanstack/react-query'
import { Loader2 } from 'lucide-react'
import { DocumentEmptyState } from '../components/DocumentEmptyState'
import { DocumentRow } from '../components/DocumentRow'
import { UploadButton } from '../components/UploadButton'
import { useDocumentUpload } from '../hooks/useDocumentUpload'
import { listDocuments } from '../services/documentService'

function DocumentList() {
  const { data: documents = [], isLoading } = useQuery({
    queryKey: ['documents'],
    queryFn: listDocuments,
    refetchInterval: 5000,
  })

  const uploadMutation = useDocumentUpload()

  return (
    <div>
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-white">My Documents</h1>
          <p className="text-slate-400 text-sm mt-1">
            Upload and manage your knowledge base
          </p>
        </div>

        <UploadButton
          isPending={uploadMutation.isPending}
          onFileSelect={(file) => uploadMutation.mutate(file)}
        />
      </div>

      {/* Document list */}
      <div className="bg-[#1e293b] border border-[#334155] rounded-xl overflow-hidden">
        {isLoading && (
          <div className="flex items-center justify-center py-12">
            <Loader2 size={24} className="animate-spin text-blue-400" />
          </div>
        )}

        {!isLoading && documents.length === 0 && <DocumentEmptyState />}

        {!isLoading && documents.length > 0 && (
          <div className="divide-y divide-[#334155]/50 px-2 py-2">
            {documents.map((doc) => (
              <DocumentRow key={doc.document_id} doc={doc} />
            ))}
          </div>
        )}
      </div>

      <p className="text-xs text-slate-600 mt-3 text-center">
        Supported: PDF, TXT, MD · Max size: 10MB
      </p>
    </div>
  )
}

export default DocumentList
