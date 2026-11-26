// src/modules/documents/services/documentService.ts
// All document-related API calls.
// Every function uses apiClient — no fetch, no raw axios.

import type { Document, DocumentUploadResponse } from '../types'
import { apiClient } from '@infrastructure/api/client'

export async function uploadDocument(file: File): Promise<DocumentUploadResponse> {
  // WHY FormData?
  //   File uploads use multipart/form-data encoding, not JSON.
  //   axios sets the Content-Type header automatically when you pass FormData.
  const form = new FormData()
  form.append('file', file)

  const response = await apiClient.post<DocumentUploadResponse>('/documents/upload', form)
  return response.data
}

export async function listDocuments(): Promise<Document[]> {
  const response = await apiClient.get<Document[]>('/documents')
  return response.data
}

export async function getDocument(documentId: string): Promise<Document> {
  const response = await apiClient.get<Document>(`/documents/${documentId}`)
  return response.data
}
