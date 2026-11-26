// src/infrastructure/api/client.ts
// Single axios instance used by every API function.
// API key and base URL come from .env — never hardcoded.

import axios from 'axios'

export const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL,
  headers: {
    'X-API-Key': import.meta.env.VITE_API_KEY,
  },
})
