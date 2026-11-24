// src/components/ChatPanel.tsx
// Working RAG chat panel connected to POST /rag/ask.

import { useMutation } from '@tanstack/react-query'
import { Bot, FileText, Loader2, MessageSquare, Send, User } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { askQuestion } from '@lib/rag'
import type { Message, SourceCitation } from '../types'

// ── Utility ────────────────────────────────────────────────────────

function generateId(): string {
  return Math.random().toString(36).slice(2, 9)
}

// ── Source citation pill ───────────────────────────────────────────

function SourcePill({ source }: { source: SourceCitation }) {
  return (
    <span className="inline-flex items-center gap-1 px-2 py-0.5 bg-[#334155] rounded text-xs text-slate-400">
      <FileText size={10} className="text-blue-400" />
      {source.filename}, chunk {source.chunk_index}
      <span className="text-slate-500 ml-0.5">
        {(source.similarity_score * 100).toFixed(0)}%
      </span>
    </span>
  )
}

// ── Single message bubble ──────────────────────────────────────────

function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === 'user'

  return (
    <div className={`flex gap-2.5 ${isUser ? 'flex-row-reverse' : 'flex-row'}`}>

      {/* Avatar */}
      <div className={`
        w-7 h-7 rounded-full flex items-center justify-center shrink-0 mt-0.5
        ${isUser ? 'bg-blue-600' : 'bg-[#334155]'}
      `}>
        {isUser
          ? <User size={14} className="text-white" />
          : <Bot size={14} className="text-slate-300" />
        }
      </div>

      {/* Bubble */}
      <div className={`max-w-[85%] ${isUser ? 'items-end' : 'items-start'} flex flex-col gap-1.5`}>
        <div className={`
          px-3 py-2 rounded-xl text-sm leading-relaxed
          ${isUser
            ? 'bg-blue-600 text-white rounded-tr-sm'
            : 'bg-[#334155] text-slate-200 rounded-tl-sm'
          }
        `}>
          {message.content}
        </div>

        {/* Sources — only for assistant messages */}
        {message.sources && message.sources.length > 0 && (
          <div className="flex flex-wrap gap-1">
            {message.sources.map((source, i) => (
              <SourcePill key={i} source={source} />
            ))}
          </div>
        )}
      </div>

    </div>
  )
}

// ── Typing indicator ───────────────────────────────────────────────

function TypingIndicator() {
  return (
    <div className="flex gap-2.5">
      <div className="w-7 h-7 rounded-full bg-[#334155] flex items-center justify-center shrink-0">
        <Bot size={14} className="text-slate-300" />
      </div>
      <div className="bg-[#334155] px-3 py-2.5 rounded-xl rounded-tl-sm">
        <div className="flex gap-1 items-center">
          <span className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce [animation-delay:0ms]" />
          <span className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce [animation-delay:150ms]" />
          <span className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce [animation-delay:300ms]" />
        </div>
      </div>
    </div>
  )
}

// ── Main chat panel ────────────────────────────────────────────────

function ChatPanel() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [conversationId, setConversationId] = useState<string | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  // Scroll to bottom whenever messages change
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const ragMutation = useMutation({
    mutationFn: askQuestion,
    onSuccess: (data) => {
      // Save conversation_id from first response so follow-up
      // questions are linked to the same conversation in Redis.
      if (!conversationId) {
        setConversationId(data.conversation_id)
      }

      setMessages((prev) => [
        ...prev,
        {
          id: generateId(),
          role: 'assistant',
          content: data.answer,
          sources: data.sources,
          timestamp: new Date(),
        },
      ])
    },
    onError: () => {
      setMessages((prev) => [
        ...prev,
        {
          id: generateId(),
          role: 'assistant',
          content: 'Something went wrong. Please try again.',
          timestamp: new Date(),
        },
      ])
    },
  })

  function handleSend() {
    const question = input.trim()
    if (!question || ragMutation.isPending) return

    // Optimistic UI — show user message immediately before API responds
    setMessages((prev) => [
      ...prev,
      {
        id: generateId(),
        role: 'user',
        content: question,
        timestamp: new Date(),
      },
    ])
    setInput('')

    ragMutation.mutate({
      question,
      conversation_id: conversationId,
      top_k: 5,
    })
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const hasMessages = messages.length > 0

  return (
    <aside className="w-80 min-w-[320px] h-screen bg-[#1e293b] border-l border-[#334155] flex flex-col overflow-hidden">

      {/* Header */}
      <div className="px-4 py-4 border-b border-[#334155]">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 bg-emerald-400 rounded-full" />
          <h2 className="text-sm font-semibold text-white">Document Assistant</h2>
        </div>
        <p className="text-xs text-slate-400 mt-0.5">
          Ask questions about your documents
        </p>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4">

        {/* Empty state */}
        {!hasMessages && (
          <div className="h-full flex flex-col items-center justify-center text-center">
            <div className="w-12 h-12 bg-[#334155] rounded-full flex items-center justify-center mb-3">
              <MessageSquare size={20} className="text-slate-500" />
            </div>
            <p className="text-sm text-slate-400 font-medium">No messages yet</p>
            <p className="text-xs text-slate-500 mt-1">
              Upload a document and ask a question
            </p>
          </div>
        )}

        {/* Message list */}
        {messages.map((message) => (
          <MessageBubble key={message.id} message={message} />
        ))}

        {/* Typing indicator while waiting for response */}
        {ragMutation.isPending && <TypingIndicator />}

        {/* Invisible div at the bottom — scroll target */}
        <div ref={bottomRef} />

      </div>

      {/* Input */}
      <div className="px-4 py-4 border-t border-[#334155]">
        <div className={`
          flex items-center gap-2 bg-[#0f172a] rounded-lg px-3 py-2.5
          border transition-colors
          ${input.length > 0 ? 'border-blue-500/50' : 'border-[#334155]'}
        `}>
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question..."
            disabled={ragMutation.isPending}
            className="
              flex-1 bg-transparent text-sm text-slate-300
              placeholder-slate-500 outline-none
              disabled:cursor-not-allowed
            "
          />
          <button
            onClick={handleSend}
            disabled={!input.trim() || ragMutation.isPending}
            className="
              w-7 h-7 bg-blue-600 rounded-md flex items-center justify-center
              hover:bg-blue-700 transition-colors
              disabled:opacity-40 disabled:cursor-not-allowed
            "
          >
            {ragMutation.isPending
              ? <Loader2 size={14} className="text-white animate-spin" />
              : <Send size={14} className="text-white" />
            }
          </button>
        </div>

        {conversationId && (
          <p className="text-xs text-slate-600 mt-2 text-center truncate">
            Conversation active
          </p>
        )}
      </div>

    </aside>
  )
}

export default ChatPanel
