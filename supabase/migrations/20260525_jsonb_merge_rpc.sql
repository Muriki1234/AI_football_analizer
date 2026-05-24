-- =============================================================================
-- Atomic JSONB merge for sessions.extra — kills the SELECT-then-UPDATE N+1
-- in db.py:update_status().
--
-- Before:
--   db.py reads sessions.extra, merges in Python, writes back. Two round-trips
--   per update; race-prone if two workers update the same session at once.
--
-- After:
--   ONE round-trip. Atomic on the Postgres side. Use jsonb || jsonb concat,
--   which merges keys with the new value winning. NULL-safe via COALESCE.
-- =============================================================================

CREATE OR REPLACE FUNCTION public.merge_session_extra(
    p_session_id uuid,
    p_extra      jsonb,
    p_updates    jsonb DEFAULT '{}'::jsonb
)
RETURNS void
LANGUAGE plpgsql
SECURITY DEFINER   -- so service-role calls bypass RLS like the existing path
SET search_path = public
AS $$
BEGIN
    UPDATE public.sessions
    SET
        extra      = COALESCE(extra, '{}'::jsonb) || p_extra,
        status     = COALESCE(p_updates ->> 'status',   status),
        progress   = COALESCE((p_updates ->> 'progress')::int, progress),
        stage      = COALESCE(p_updates ->> 'stage',    stage),
        error      = COALESCE(p_updates ->> 'error',    error),
        updated_at = NOW()
    WHERE id = p_session_id;
END;
$$;

GRANT EXECUTE ON FUNCTION public.merge_session_extra(uuid, jsonb, jsonb) TO service_role;
GRANT EXECUTE ON FUNCTION public.merge_session_extra(uuid, jsonb, jsonb) TO authenticated;
