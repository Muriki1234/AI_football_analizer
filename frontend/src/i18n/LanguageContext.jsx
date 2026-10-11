import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { translations } from './translations';

const LanguageContext = createContext(null);

const STORAGE_KEY = 'pitchlogic_app_lang';

export function LanguageProvider({ children }) {
    const [lang, setLangState] = useState(() => {
        try {
            const saved = localStorage.getItem(STORAGE_KEY);
            if (saved === 'zh' || saved === 'en') return saved;
        } catch {}
        return 'zh';
    });

    const setLang = useCallback((newLang) => {
        if (newLang !== 'zh' && newLang !== 'en') return;
        setLangState(newLang);
        try {
            localStorage.setItem(STORAGE_KEY, newLang);
            document.documentElement.lang = newLang;
        } catch {}
    }, []);

    const toggleLang = useCallback(() => {
        setLang(lang === 'zh' ? 'en' : 'zh');
    }, [lang, setLang]);

    useEffect(() => {
        try {
            document.documentElement.lang = lang;
        } catch {}
    }, [lang]);

    /**
     * Translates a dot-notated key. E.g. t('welcome.title', { name: 'Player' })
     */
    const t = useCallback((keyPath, params = {}) => {
        if (!keyPath) return '';
        const parts = keyPath.split('.');
        
        let curr = translations[lang];
        for (const part of parts) {
            if (curr && typeof curr === 'object' && part in curr) {
                curr = curr[part];
            } else {
                curr = null;
                break;
            }
        }

        // Fallback to English if missing in Chinese (or vice-versa)
        if (curr == null) {
            let fallback = translations[lang === 'zh' ? 'en' : 'zh'];
            for (const part of parts) {
                if (fallback && typeof fallback === 'object' && part in fallback) {
                    fallback = fallback[part];
                } else {
                    fallback = null;
                    break;
                }
            }
            curr = fallback != null ? fallback : keyPath;
        }

        if (typeof curr !== 'string') return keyPath;

        // Replace {param} placeholders
        let res = curr;
        if (params && typeof params === 'object') {
            for (const [k, v] of Object.entries(params)) {
                res = res.replace(new RegExp(`\\{${k}\\}`, 'g'), String(v ?? ''));
            }
        }
        return res;
    }, [lang]);

    const value = {
        lang,
        setLang,
        toggleLang,
        t,
        isZh: lang === 'zh',
        isEn: lang === 'en',
    };

    return (
        <LanguageContext.Provider value={value}>
            {children}
        </LanguageContext.Provider>
    );
}

export function useLanguage() {
    const ctx = useContext(LanguageContext);
    if (!ctx) {
        throw new Error('useLanguage must be used within a LanguageProvider');
    }
    return ctx;
}
