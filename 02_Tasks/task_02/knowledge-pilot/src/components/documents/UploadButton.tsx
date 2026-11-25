// src/components/documents/UploadButton.tsx
// Upload button + hidden file input.
// Owns the file-picking interaction; delegates the actual upload to the caller via onFileSelect.

import { Loader2, Upload } from 'lucide-react'
import { useRef } from 'react'

interface UploadButtonProps {
    isPending: boolean
    onFileSelect: (file: File) => void
}

export function UploadButton({ isPending, onFileSelect }: UploadButtonProps) {
    const fileInputRef = useRef<HTMLInputElement>(null)

    function handleChange(e: React.ChangeEvent<HTMLInputElement>) {
        const file = e.target.files?.[0]
        if (!file) return
        onFileSelect(file)
        e.target.value = ''
    }

    return (
        <>
            <button
                onClick={() => fileInputRef.current?.click()}
                disabled={isPending}
                className="
                    flex items-center gap-2 px-4 py-2
                    bg-blue-600 hover:bg-blue-700
                    disabled:opacity-50 disabled:cursor-not-allowed
                    text-white text-sm font-medium rounded-lg
                    transition-colors
                "
            >
                {isPending ? (
                    <Loader2 size={16} className="animate-spin" />
                ) : (
                    <Upload size={16} />
                )}
                {isPending ? 'Uploading...' : 'Upload'}
            </button>

            <input
                ref={fileInputRef}
                type="file"
                accept=".pdf,.txt,.md"
                className="hidden"
                onChange={handleChange}
            />
        </>
    )
}
