const STORAGE_KEY = 'pitchlogic.recentSessions';
const MAX_ITEMS = 5;
let currentUserId = null;

export const setRecentSessionUser = (userId) => {
    if (currentUserId && currentUserId !== userId) {
        try { localStorage.removeItem(`${STORAGE_KEY}.${currentUserId}`); } catch { /* storage unavailable */ }
    }
    currentUserId = userId || null;
    // The old unscoped cache can include another person's video URLs.
    try { localStorage.removeItem(STORAGE_KEY); } catch { /* storage unavailable */ }
};

const userStorageKey = () => currentUserId ? `${STORAGE_KEY}.${currentUserId}` : null;

const read = () => {
    try {
        const key = userStorageKey();
        if (!key) return [];
        const raw = localStorage.getItem(key);
        const items = raw ? JSON.parse(raw) : [];
        return Array.isArray(items) ? items.slice(0, MAX_ITEMS) : [];
    } catch {
        return [];
    }
};

const write = (items) => {
    try {
        const key = userStorageKey();
        if (key) localStorage.setItem(key, JSON.stringify(items.slice(0, MAX_ITEMS)));
    } catch {
        // localStorage full / disabled — silent ignore
    }
};

export const getRecentSessions = () => read();

export const addRecentSession = (session) => {
    const next = [
        { ...session, addedAt: Date.now() },
        ...read().filter((s) => s.id !== session.id),
    ];
    write(next);
};

export const removeRecentSession = (id) => {
    write(read().filter((s) => s.id !== id));
};
