// src/modules/documents/types/index.ts

export interface Document {
  document_id: string
  filename: string
  file_type: string
  file_size_bytes: number
  status: 'processing' | 'ready' | 'failed'
  created_at: string
  chunk_count?: number
}

export interface DocumentUploadResponse {
  document_id: string
  filename: string
  file_type: string
  file_size_bytes: number
  status: string
}
