-- ============================================================================
-- Enable Realtime broadcasting for sessions + tasks tables.
--
-- For Supabase Realtime to push UPDATE events with full row data we need:
--   1. The table is in the supabase_realtime publication
--   2. REPLICA IDENTITY FULL (so UPDATE payloads include every column,
--      not just the primary key)
--
-- Without #2 the frontend gets a NOTIFY ping but no `progress` / `status`
-- fields, which is why we'd been falling back to 2s polling.
-- ============================================================================

-- 1. REPLICA IDENTITY FULL ----------------------------------------------------
-- Costs slightly more WAL bandwidth but is required for full-row UPDATE
-- payloads. Negligible at our row size.
ALTER TABLE public.sessions REPLICA IDENTITY FULL;
ALTER TABLE public.tasks    REPLICA IDENTITY FULL;

-- 2. Add tables to the supabase_realtime publication --------------------------
-- ALTER PUBLICATION ... ADD TABLE errors if already a member, so we guard.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_publication_tables
        WHERE pubname = 'supabase_realtime'
          AND schemaname = 'public'
          AND tablename = 'sessions'
    ) THEN
        ALTER PUBLICATION supabase_realtime ADD TABLE public.sessions;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_publication_tables
        WHERE pubname = 'supabase_realtime'
          AND schemaname = 'public'
          AND tablename = 'tasks'
    ) THEN
        ALTER PUBLICATION supabase_realtime ADD TABLE public.tasks;
    END IF;
END $$;

-- 3. Verify -------------------------------------------------------------------
-- Run this to confirm:
--     SELECT schemaname, tablename FROM pg_publication_tables
--      WHERE pubname = 'supabase_realtime';
--
--     SELECT relname, relreplident FROM pg_class
--      WHERE relname IN ('sessions', 'tasks');
--   relreplident should be 'f' (FULL), not 'd' (DEFAULT) or 'n' (NOTHING).
