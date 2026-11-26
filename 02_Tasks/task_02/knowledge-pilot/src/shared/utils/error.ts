// src/shared/utils/error.ts
// Extracts a human-readable error message from an axios/fetch error.
// FastAPI returns { detail: "..." } for every HTTPException.

export function extractErrorDetail(
  error: unknown,
  fallback = 'Something went wrong. Please try again.'
): string {
  const detail =
    (error as any)?.response?.data?.detail ??
    (error as any)?.message ??
    fallback

  return typeof detail === 'string' ? detail : JSON.stringify(detail)
}
