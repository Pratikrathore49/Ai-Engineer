// src/components/documents/StatusBadge.tsx
// Renders a coloured pill for processing / ready / failed status.

import { AlertCircle, CheckCircle2, Loader2 } from 'lucide-react'
import type { Document } from '@types/index'

interface StatusBadgeProps {
    status: Document['status']
}

export function StatusBadge({ status }: StatusBadgeProps) {
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
