// src/hooks/useDocumentPolling.ts
// Polls a single document's status every 2s while it is still processing.
// When status changes, it invalidates the documents list so the row re-renders.

import { useQuery, useQueryClient } from '@tanstack/react-query'
import { getDocument } from '@lib/documents'
import type { Document } from '@types/index'

export function useDocumentPolling(doc: Document) {
    const queryClient = useQueryClient()

    useQuery({
        queryKey: ['document', doc.document_id],
        queryFn: () => getDocument(doc.document_id),
        enabled: doc.status === 'processing',
        refetchInterval: 2000,
        select: (freshDoc) => {
            if (freshDoc.status !== 'processing') {
                queryClient.invalidateQueries({ queryKey: ['documents'] })
            }
            return freshDoc
        },
    })
}
