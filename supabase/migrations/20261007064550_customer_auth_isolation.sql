-- Verified customers only. No user/session data is deleted by this migration.
-- A logout removes auth.sessions even though its access JWT has not yet expired.
create or replace function public.customer_session_active()
returns boolean language sql stable security definer set search_path = ''
as $$
    select exists (
        select 1 from auth.sessions s join auth.users u on u.id = s.user_id
        where s.user_id = (select auth.uid())
          and s.id::text = (select auth.jwt() ->> 'session_id')
          and (s.not_after is null or s.not_after > now())
          and not coalesce(u.is_anonymous, false)
          and (u.email_confirmed_at is not null or u.phone_confirmed_at is not null)
    );
$$;
revoke all on function public.customer_session_active() from public, anon;
grant execute on function public.customer_session_active() to authenticated, service_role;

alter table public.sessions enable row level security;
alter table public.tasks enable row level security;
-- Permissive policies combine with OR: historic allow-all rules must be removed.
do $$
declare p record;
begin
    for p in select tablename, policyname from pg_policies
        where schemaname = 'public' and tablename in ('sessions', 'tasks')
    loop
        execute format('drop policy %I on public.%I', p.policyname, p.tablename);
    end loop;
end $$;
create policy sessions_select_own on public.sessions for select to authenticated
using (user_id = (select auth.uid()) and (select public.customer_session_active()));
create policy sessions_insert_own on public.sessions for insert to authenticated
with check (user_id = (select auth.uid()) and (select public.customer_session_active()));
create policy sessions_update_own on public.sessions for update to authenticated
using (user_id = (select auth.uid()) and (select public.customer_session_active()))
with check (user_id = (select auth.uid()) and (select public.customer_session_active()));
create policy sessions_delete_own on public.sessions for delete to authenticated
using (user_id = (select auth.uid()) and (select public.customer_session_active()));
create policy tasks_own on public.tasks for all to authenticated
using ((select public.customer_session_active()) and exists (
    select 1 from public.sessions s where s.id = tasks.session_id and s.user_id = (select auth.uid())
))
with check ((select public.customer_session_active()) and exists (
    select 1 from public.sessions s where s.id = tasks.session_id and s.user_id = (select auth.uid())
));
-- Only workers may call privileged update/cleanup functions. PUBLIC otherwise
-- grants access even after explicit authenticated/anon grants are revoked.
do $$
declare f record;
begin
    for f in select p.oid::regprocedure as signature
        from pg_proc p join pg_namespace n on n.oid = p.pronamespace
        where n.nspname = 'public' and p.proname in ('merge_session_extra', 'cleanup_old_sessions', 'rls_auto_enable')
    loop
        execute format('revoke execute on function %s from public, anon, authenticated', f.signature);
        execute format('grant execute on function %s to service_role', f.signature);
    end loop;
end $$;
-- R2 writes use Vercel. Legacy Supabase video reads remain scoped to owners.
drop policy if exists "Authenticated users can read videos" on storage.objects;
drop policy if exists "Authenticated users can upload videos" on storage.objects;
drop policy if exists "users_read_own_videos" on storage.objects;
drop policy if exists "users_upload_own_videos" on storage.objects;
create policy users_read_own_videos on storage.objects for select to authenticated
using (bucket_id = 'videos' and (select public.customer_session_active()) and exists (
    select 1 from public.sessions s
    where s.id::text = split_part(storage.objects.name, '/', 1)
      and s.user_id = (select auth.uid())
));
