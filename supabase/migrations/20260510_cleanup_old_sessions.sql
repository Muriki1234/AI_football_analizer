-- ============================================================================
-- Cleanup job: delete sessions older than 7 days, plus their storage objects.
-- Run once in Supabase SQL Editor. Safe to re-run (uses unschedule + schedule).
-- ============================================================================

-- 1. Required extensions ------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS pg_cron;
CREATE EXTENSION IF NOT EXISTS pg_net;  -- not used here, but commonly needed

-- 2. Make sure tasks rows die with their session ------------------------------
-- If the FK already exists with ON DELETE CASCADE, this is a no-op.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.referential_constraints
        WHERE constraint_name = 'tasks_session_id_fkey'
          AND delete_rule = 'CASCADE'
    ) THEN
        BEGIN
            ALTER TABLE public.tasks
                DROP CONSTRAINT IF EXISTS tasks_session_id_fkey;
        EXCEPTION WHEN undefined_object THEN NULL;
        END;
        ALTER TABLE public.tasks
            ADD CONSTRAINT tasks_session_id_fkey
            FOREIGN KEY (session_id)
            REFERENCES public.sessions (id)
            ON DELETE CASCADE;
    END IF;
END $$;

-- 3. The actual cleanup function ---------------------------------------------
CREATE OR REPLACE FUNCTION public.cleanup_old_sessions()
RETURNS void
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
    cutoff timestamptz := NOW() - INTERVAL '7 days';
    obj record;
BEGIN
    -- Delete storage objects whose owning session is about to be deleted.
    -- We do this BEFORE the row delete so we can still resolve session_id -> path.
    FOR obj IN
        SELECT o.name
        FROM storage.objects o
        JOIN public.sessions s
          ON o.bucket_id = 'videos'
         AND split_part(o.name, '/', 1) = s.id::text
        WHERE s.created_at < cutoff
    LOOP
        DELETE FROM storage.objects
            WHERE bucket_id = 'videos' AND name = obj.name;
    END LOOP;

    -- Delete the sessions themselves (CASCADE wipes tasks).
    DELETE FROM public.sessions WHERE created_at < cutoff;

    -- Sweep orphan tasks just in case CASCADE didn't catch them.
    DELETE FROM public.tasks
        WHERE session_id NOT IN (SELECT id FROM public.sessions);
END;
$$;

-- 4. Schedule it daily at 04:00 UTC -------------------------------------------
-- Unschedule any prior version so this migration is idempotent.
DO $$
DECLARE
    job_id bigint;
BEGIN
    SELECT jobid INTO job_id
        FROM cron.job
        WHERE jobname = 'cleanup-old-sessions';
    IF job_id IS NOT NULL THEN
        PERFORM cron.unschedule(job_id);
    END IF;
END $$;

SELECT cron.schedule(
    'cleanup-old-sessions',
    '0 4 * * *',
    $$ SELECT public.cleanup_old_sessions(); $$
);

-- 5. Manual test --------------------------------------------------------------
-- Run this once to confirm it works on the current data:
--     SELECT public.cleanup_old_sessions();
--
-- Inspect the schedule:
--     SELECT jobid, jobname, schedule, command FROM cron.job;
--
-- Watch recent runs:
--     SELECT * FROM cron.job_run_details ORDER BY start_time DESC LIMIT 10;
