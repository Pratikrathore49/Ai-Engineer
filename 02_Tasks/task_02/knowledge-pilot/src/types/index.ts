// src/types/index.ts
// All shared TypeScript types in one place.
// Every component and API function imports from here.

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

export interface SourceCitation {
  filename: string
  chunk_index: number
  similarity_score: number
}

export interface RAGResponse {
  answer: string
  conversation_id: string
  sources: SourceCitation[]
  chunks_used: number
}

export interface RAGRequest {
  question: string
  document_id?: string | null
  conversation_id?: string | null
  top_k?: number
}

export interface SearchRequest {
  query: string
  document_id?: string | null
  top_k?: number
}

export interface RetrievedChunk {
  chunk_id: string
  content: string
  filename: string
  chunk_index: number
  similarity_score: number
}

export interface SearchResponse {
  query: string
  results: RetrievedChunk[]
  total_results: number
}

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  sources?: SourceCitation[]
  timestamp: Date
}
