// src/hooks/useDocumentUpload.ts
// Owns the upload mutation + toast side-effects.
// Nothing here knows about UI layout or the document list.

import { useMutation, useQueryClient } from '@tanstack/react-query'
import { toast } from 'sonner'
import { uploadDocument } from '@lib/documents'
import { extractErrorDetail } from '@utils/error'

export function useDocumentUpload() {
    const queryClient = useQueryClient()

    const mutation = useMutation({
        mutationFn: uploadDocument,
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['documents'] })
            toast.success('Document uploaded successfully!')
        },
        onError: (error: unknown) => {
            toast.error(
                extractErrorDetail(
                    error,
                    'Upload failed. Make sure the file is a PDF, TXT, or MD under 10MB.'
                )
            )
        },
    })

    return mutation
}
