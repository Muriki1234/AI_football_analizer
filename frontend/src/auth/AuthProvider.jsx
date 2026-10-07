import { useCallback, useEffect, useState } from 'react';
import { supabase } from '../lib/supabase';
import { isCustomer } from '../lib/auth';
import { setRecentSessionUser } from '../lib/recentSessions';
import { AuthContext } from './AuthContext';

export default function AuthProvider({ children }) {
    const [state, setState] = useState({ session: null, loading: true, error: null });
    const [recovery, setRecovery] = useState(false);

    useEffect(() => {
        let mounted = true;
        let revision = 0;
        const apply = (session, error = null) => {
            if (!mounted) return;
            const customerSession = isCustomer(session?.user) ? session : null;
            setRecentSessionUser(customerSession?.user.id);
            setState({ session: customerSession, loading: false, error });
        };
        // Keep this callback synchronous: awaiting Auth calls inside it can deadlock the SDK.
        const { data: { subscription } } = supabase.auth.onAuthStateChange((event, session) => {
            if (event === 'INITIAL_SESSION') return;
            revision += 1;
            if (event === 'PASSWORD_RECOVERY') setRecovery(true);
            if (event === 'SIGNED_OUT') setRecovery(false);
            apply(session);
        });
        const initialRevision = revision;
        (async () => {
            try {
                const { data, error } = await supabase.auth.getSession();
                if (error) throw error;
                let session = data.session;
                if (session) {
                    const verified = await supabase.auth.getUser();
                    if (verified.error) throw verified.error;
                    session = { ...session, user: verified.data.user };
                }
                if (revision === initialRevision) apply(session);
            } catch {
                if (revision === initialRevision) apply(null, 'Unable to restore your sign-in. Please sign in again.');
            }
        })();
        return () => { mounted = false; subscription.unsubscribe(); };
    }, []);

    const signOut = useCallback(async () => {
        const { error } = await supabase.auth.signOut({ scope: 'local' });
        if (error) throw error;
        setRecentSessionUser(null);
        setState({ session: null, loading: false, error: null });
        setRecovery(false);
    }, []);

    return <AuthContext.Provider value={{ ...state, user: state.session?.user ?? null, recovery, finishRecovery: () => setRecovery(false), signOut }}>
        {children}
    </AuthContext.Provider>;
}
