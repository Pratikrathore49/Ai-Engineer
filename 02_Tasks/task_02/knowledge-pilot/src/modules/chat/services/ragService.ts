// src/modules/chat/services/ragService.ts
// RAG pipeline API call.

import { apiClient } from '@infrastructure/api/client'
import type { RAGRequest, RAGResponse } from '../types'

export async function askQuestion(request: RAGRequest): Promise<RAGResponse> {
  const response = await apiClient.post<RAGResponse>('/rag/ask', request)
  return response.data
}
