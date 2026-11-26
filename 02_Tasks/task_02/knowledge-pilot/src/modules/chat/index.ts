// src/modules/chat/index.ts
// Public API for the chat module.

export { ChatPanel } from './components/ChatPanel'
export { askQuestion } from './services/ragService'
export type { RAGRequest, RAGResponse, Message, SourceCitation } from './types'
