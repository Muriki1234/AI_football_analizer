-- Enable Row Level Security on sessions + tasks.
--
-- BACKGROUND
--   Until this migration, both tables had no RLS. Combined with anonymous
--   sign-ins, ANY authenticated visitor could `select *` and read every
--   session row by guessing a 12-hex session_id (48 bits, enumerable),
--   and `update`/`insert` arbitrary rows. The merge_session_extra RPC
--   was also granted to `authenticated`, letting any user mutate any
--   session's stage / extra JSONB.
--
-- POLICY DESIGN
--   - `sessions.user_id` already exists (populated by uploadVideo on the
--     frontend with auth.uid()). We trust it as the ownership column.
--   - `tasks.session_id` FKs to sessions(id); ownership delegates via
--     a sub-select against sessions.user_id.
--   - Server-side workers (RunPod handler, Pod FastAPI) authenticate with
--     the SERVICE ROLE key which bypasses RLS — no policy needed for them.
--
-- ROLLBACK
--   If something breaks, you can quickly disable RLS via
--     alter table public.sessions disable row level security;
--     alter table public.tasks    disable row level security;
--   The policies themselves keep existing but stop enforcing.

-- 1. sessions ---------------------------------------------------------------
alter table public.sessions enable row level security;

-- Drop any old policies (idempotent — first run will say "does not exist")
drop policy if exists "sessions_select_own" on public.sessions;
drop policy if exists "sessions_insert_own" on public.sessions;
drop policy if exists "sessions_update_own" on public.sessions;
drop policy if exists "sessions_delete_own" on public.sessions;

create policy "sessions_select_own"
    on public.sessions for select
    to authenticated
    using (user_id = auth.uid());

create policy "sessions_insert_own"
    on public.sessions for insert
    to authenticated
    with check (user_id = auth.uid());

create policy "sessions_update_own"
    on public.sessions for update
    to authenticated
    using (user_id = auth.uid())
    with check (user_id = auth.uid());

create policy "sessions_delete_own"
    on public.sessions for delete
    to authenticated
    using (user_id = auth.uid());

-- 2. tasks ------------------------------------------------------------------
alter table public.tasks enable row level security;

drop policy if exists "tasks_select_own" on public.tasks;
drop policy if exists "tasks_modify_own" on public.tasks;

create policy "tasks_select_own"
    on public.tasks for select
    to authenticated
    using (
        exists (
            select 1 from public.sessions s
            where s.id = tasks.session_id and s.user_id = auth.uid()
        )
    );

-- Clients normally don't write tasks directly (the worker does via
-- service_role), but anonymous frontend code does occasionally write
-- task rows for client-only features. We still scope writes to the
-- session owner.
create policy "tasks_modify_own"
    on public.tasks for all
    to authenticated
    using (
        exists (
            select 1 from public.sessions s
            where s.id = tasks.session_id and s.user_id = auth.uid()
        )
    )
    with check (
        exists (
            select 1 from public.sessions s
            where s.id = tasks.session_id and s.user_id = auth.uid()
        )
    );

-- 3. merge_session_extra RPC — tighten access (if deployed) -------------------
-- 之前 20260525_jsonb_merge_rpc.sql 把 EXECUTE 授给了 authenticated，相当于
-- 任何登录用户都能改任意 session 的元数据。撤销该权限，只留 service_role。
--
-- 注：这个 RPC 可能在某些 Supabase 项目里还没部署 (server/storage/db.py 有
-- 自动 fallback 走 SELECT-then-UPDATE 老路)。所以用动态查找 + 条件 revoke，
-- 不存在就跳过，整个 migration 仍能成功。
do $$
declare
    f record;
begin
    for f in
        select n.nspname || '.' || p.proname || '(' || pg_get_function_identity_arguments(p.oid) || ')' as sig
        from pg_proc p
        join pg_namespace n on n.oid = p.pronamespace
        where n.nspname = 'public'
          and p.proname = 'merge_session_extra'
    loop
        execute format('revoke execute on function %s from authenticated', f.sig);
        execute format('revoke execute on function %s from anon', f.sig);
        raise notice 'revoked merge_session_extra access from authenticated/anon: %', f.sig;
    end loop;
end$$;
-- service_role retains access via its bypass-RLS default; no grant needed.
