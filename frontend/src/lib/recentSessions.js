const STORAGE_KEY = 'pitchlogic.recentSessions';
const MAX_ITEMS = 5;

const read = () => {
    try {
        const raw = localStorage.getItem(STORAGE_KEY);
        return raw ? JSON.parse(raw) : [];
    } catch {
        return [];
    }
};

const write = (items) => {
    try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(items.slice(0, MAX_ITEMS)));
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
