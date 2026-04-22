/**
 * Frontend configuration. Vite injects VITE_* env vars at build time.
 *
 * API_BASE_URL  — absolute URL of the FastAPI server. Leave blank in dev so
 *                 Vite's proxy (vite.config.js) forwards /api/* to :8000.
 * API_KEY       — bearer token the server expects as X-API-Key. Blank = dev.
 */

export const API_BASE_URL =
    (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '');

export const API_KEY = import.meta.env.VITE_API_KEY || '';

export const absUrl = (path) => {
    if (!path) return path;
    if (/^https?:\/\//i.test(path)) return path;
    const p = path.startsWith('/') ? path : `/${path}`;
    return API_BASE_URL ? `${API_BASE_URL}${p}` : p;
};

export const authHeaders = () =>
    API_KEY ? { 'X-API-Key': API_KEY } : {};
