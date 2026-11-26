// src/modules/documents/index.ts
// Public API for the documents module.

export { default as DocumentList } from './pages/DocumentList'
export { default as DocumentDetail } from './pages/DocumentDetail'
export { DocumentRow } from './components/DocumentRow'
export { DocumentEmptyState } from './components/DocumentEmptyState'
export { StatusBadge } from './components/StatusBadge'
export { FileIcon } from './components/FileIcon'
export { UploadButton } from './components/UploadButton'
export { useDocumentUpload } from './hooks/useDocumentUpload'
export { useDocumentPolling } from './hooks/useDocumentPolling'
export { listDocuments, uploadDocument, getDocument } from './services/documentService'
export type { Document, DocumentUploadResponse } from './types'
