// src/pages/DocumentList.tsx
// Document list page with upload button and status polling.

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
    AlertCircle,
    CheckCircle2,
    FileText,
    Layers,
    Loader2,
    Upload,
} from 'lucide-react'
import { useRef } from 'react'
import { toast } from 'sonner'
import { useNavigate } from 'react-router-dom'
import { getDocument, listDocuments, uploadDocument } from '@lib/documents'
import type { Document } from '../types'

// ── Status badge ───────────────────────────────────────────────────


function StatusBadge({ status }: { status: Document['status'] }) {
    if (status === 'ready') {
        return (
            <span className="flex items-center gap-1 text-xs font-medium text-emerald-400">
                <CheckCircle2 size={12} />
                ready
            </span>
        )
    }
    if (status === 'processing') {
        return (
            <span className="flex items-center gap-1 text-xs font-medium text-amber-400">
                <Loader2 size={12} className="animate-spin" />
                processing
            </span>
        )
    }
    return (
        <span className="flex items-center gap-1 text-xs font-medium text-red-400">
            <AlertCircle size={12} />
            failed
        </span>
    )
}

// ── File icon by type ──────────────────────────────────────────────

function FileIcon({ fileType }: { fileType: string }) {
    return (
        <div className="w-8 h-8 bg-blue-600/20 rounded-lg flex items-center justify-center">
            <FileText size={16} className="text-blue-400" />
        </div>
    )
}

// ── Format bytes ───────────────────────────────────────────────────

function formatBytes(bytes: number): string {
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

// ── Single document row ────────────────────────────────────────────

function DocumentRow({ doc }: { doc: Document }) {
    const navigate = useNavigate()
    const queryClient = useQueryClient()

    // WHY useQuery with enabled: doc.status === 'processing'?
    //   We only want to poll when the document is still processing.
    //   Once it becomes ready or failed, polling stops automatically.
    //   refetchInterval: 2000 polls every 2 seconds while enabled.
    useQuery({
        queryKey: ['document', doc.document_id],
        queryFn: () => getDocument(doc.document_id),
        enabled: doc.status === 'processing',
        refetchInterval: 2000,
        // When we get fresh data, update the documents list in the cache
        // so the row re-renders with the new status without a full refetch.
        select: (freshDoc) => {
            if (freshDoc.status !== 'processing') {
                queryClient.invalidateQueries({ queryKey: ['documents'] })
            }
            return freshDoc
        },
    })

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

// ── Main page ──────────────────────────────────────────────────────

function DocumentList() {
    const fileInputRef = useRef<HTMLInputElement>(null)
    const queryClient = useQueryClient()

    // Fetch document list
    const { data: documents = [], isLoading } = useQuery({
        queryKey: ['documents'],
        queryFn: listDocuments,
        refetchInterval: 5000,
        // WHY refetchInterval here too?
        //   This is the background refresh for the whole list.
        //   The per-row polling (above) handles individual documents.
        //   Both work together — per-row is faster (2s), list is slower (5s).
    })

    // Upload mutation
    const uploadMutation = useMutation({
        mutationFn: uploadDocument,
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['documents'] })
            toast.success('Document uploaded successfully!')
        },
        onError: (error: unknown) => {
            const detail =
                (error as any)?.response?.data?.detail ??
                (error as any)?.message ??
                'Upload failed. Make sure the file is a PDF, TXT, or MD under 10MB.'

            toast.error(typeof detail === 'string' ? detail : JSON.stringify(detail))
        },
    })

    function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
        const file = e.target.files?.[0]
        if (!file) return
        uploadMutation.mutate(file)
        // Reset input so the same file can be uploaded again if needed
        e.target.value = ''
    }

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

                {/* Upload button — triggers hidden file input */}
                <button
                    onClick={() => fileInputRef.current?.click()}
                    disabled={uploadMutation.isPending}
                    className="
            flex items-center gap-2 px-4 py-2
            bg-blue-600 hover:bg-blue-700
            disabled:opacity-50 disabled:cursor-not-allowed
            text-white text-sm font-medium rounded-lg
            transition-colors
          "
                >
                    {uploadMutation.isPending ? (
                        <Loader2 size={16} className="animate-spin" />
                    ) : (
                        <Upload size={16} />
                    )}
                    {uploadMutation.isPending ? 'Uploading...' : 'Upload'}
                </button>

                {/* Hidden file input — only accepts our file types */}
                <input
                    ref={fileInputRef}
                    type="file"
                    accept=".pdf,.txt,.md"
                    className="hidden"
                    onChange={handleFileChange}
                />
            </div>

            {/* Document list */}
            <div className="bg-[#1e293b] border border-[#334155] rounded-xl overflow-hidden">

                {isLoading && (
                    <div className="flex items-center justify-center py-12">
                        <Loader2 size={24} className="animate-spin text-blue-400" />
                    </div>
                )}

                {!isLoading && documents.length === 0 && (
                    <div className="flex flex-col items-center justify-center py-16 text-center">
                        <div className="w-14 h-14 bg-[#334155] rounded-full flex items-center justify-center mb-4">
                            <FileText size={24} className="text-slate-500" />
                        </div>
                        <p className="text-sm font-medium text-slate-300">No documents yet</p>
                        <p className="text-xs text-slate-500 mt-1">
                            Click Upload to add your first document
                        </p>
                    </div>
                )}

                {!isLoading && documents.length > 0 && (
                    <div className="divide-y divide-[#334155]/50 px-2 py-2">
                        {documents.map((doc) => (
                            <DocumentRow key={doc.document_id} doc={doc} />
                        ))}
                    </div>
                )}

            </div>

            {/* File type info */}
            <p className="text-xs text-slate-600 mt-3 text-center">
                Supported: PDF, TXT, MD · Max size: 10MB
            </p>
        </div>
    )
}

export default DocumentList
