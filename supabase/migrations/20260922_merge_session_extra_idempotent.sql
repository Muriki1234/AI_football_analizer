-- =============================================================================
-- Migration: 20260922_merge_session_extra_idempotent.sql
-- Description: Idempotent deployment of public.merge_session_extra() RPC.
-- Resolves PostgREST PGRST202 (HTTP 404) schema cache error on RunPod workers.
-- Kills the sequential GET-then-PATCH roundtrips during analysis streaming.
-- =============================================================================

CREATE OR REPLACE FUNCTION public.merge_session_extra(
    p_session_id text,
    p_extra      jsonb,
    p_updates    jsonb DEFAULT '{}'::jsonb
)
RETURNS void
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
    UPDATE public.sessions
    SET
        extra      = COALESCE(extra, '{}'::jsonb) || COALESCE(p_extra, '{}'::jsonb),
        status     = COALESCE(p_updates ->> 'status',   status),
        progress   = COALESCE((p_updates ->> 'progress')::int, progress),
        stage      = COALESCE(p_updates ->> 'stage',    stage),
        error      = COALESCE(p_updates ->> 'error',    error),
        updated_at = NOW()
    WHERE id = p_session_id;
END;
$$;

-- Grant execution permissions
GRANT EXECUTE ON FUNCTION public.merge_session_extra(text, jsonb, jsonb) TO anon;
GRANT EXECUTE ON FUNCTION public.merge_session_extra(text, jsonb, jsonb) TO authenticated;
GRANT EXECUTE ON FUNCTION public.merge_session_extra(text, jsonb, jsonb) TO service_role;

-- Force PostgREST schema cache reload so the RPC becomes immediately callable
NOTIFY pgrst, 'reload schema';
