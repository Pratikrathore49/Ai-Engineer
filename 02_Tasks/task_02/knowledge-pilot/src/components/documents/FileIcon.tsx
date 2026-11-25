// src/components/documents/FileIcon.tsx
// Small coloured square icon representing a document file.

import { FileText } from 'lucide-react'

interface FileIconProps {
    fileType: string
}

export function FileIcon({ fileType: _fileType }: FileIconProps) {
    return (
        <div className="w-8 h-8 bg-blue-600/20 rounded-lg flex items-center justify-center">
            <FileText size={16} className="text-blue-400" />
        </div>
    )
}
