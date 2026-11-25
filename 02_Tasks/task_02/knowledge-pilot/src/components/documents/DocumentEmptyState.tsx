// src/components/documents/DocumentEmptyState.tsx
// Shown when the document list is empty.

import { FileText } from 'lucide-react'

export function DocumentEmptyState() {
    return (
        <div className="flex flex-col items-center justify-center py-16 text-center">
            <div className="w-14 h-14 bg-[#334155] rounded-full flex items-center justify-center mb-4">
                <FileText size={24} className="text-slate-500" />
            </div>
            <p className="text-sm font-medium text-slate-300">No documents yet</p>
            <p className="text-xs text-slate-500 mt-1">
                Click Upload to add your first document
            </p>
        </div>
    )
}
