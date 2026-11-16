-- OSIRIS workspace v3: multi-user writes and an admin subspace. NOT APPLIED until the owner approves it.
--
-- Target: Supabase project "osiris" (ref eirdsaeutzkzepibnfxf). Depends on the two earlier migrations.
--
-- What changes:
--   1. A signed-in user can create, change and delete their OWN projects and the tasks inside them.
--      owner_id defaults to the caller and can never be set or changed to someone else.
--   2. Dispositions stay backend-only for writes (OSIRIS records them); owners can read theirs.
--   3. Admins (rows in osiris_admins) can READ every project, task and disposition. Only the backend or the
--      database owner can add or remove an admin; no user can make themselves one.
--   4. The append-only trigger function gets a fixed search_path (Supabase security advisor warning).
--
-- Rollback:
--   drop policy osiris_projects_admin_select on public.osiris_projects;
--   drop policy osiris_tasks_admin_select on public.osiris_tasks;
--   drop policy osiris_dispositions_admin_select on public.osiris_dispositions;
--   drop policy osiris_projects_owner_insert on public.osiris_projects;
--   drop policy osiris_projects_owner_update on public.osiris_projects;
--   drop policy osiris_projects_owner_delete on public.osiris_projects;
--   drop policy osiris_tasks_owner_insert on public.osiris_tasks;
--   drop policy osiris_tasks_owner_update on public.osiris_tasks;
--   drop policy osiris_tasks_owner_delete on public.osiris_tasks;
--   revoke insert, update, delete on public.osiris_projects, public.osiris_tasks from authenticated;
--   alter table public.osiris_projects alter column owner_id drop default;
--   drop function public.osiris_is_admin();
--   drop table public.osiris_admins;

alter function public.osiris_dispositions_append_only() set search_path = '';

create table public.osiris_admins (
  user_id    uuid primary key references auth.users(id) on delete cascade,
  created_at timestamptz not null default now()
);
alter table public.osiris_admins enable row level security;          -- no policies: backend only
revoke all on public.osiris_admins from anon, authenticated;

-- True when the caller is an admin. SECURITY DEFINER so it can read osiris_admins, which callers cannot.
create function public.osiris_is_admin() returns boolean
  language sql stable security definer set search_path = ''
  as $$ select exists (select 1 from public.osiris_admins a where a.user_id = (select auth.uid())) $$;
revoke all on function public.osiris_is_admin() from public, anon;
grant execute on function public.osiris_is_admin() to authenticated;

alter table public.osiris_projects alter column owner_id set default auth.uid();
grant insert, update, delete on public.osiris_projects, public.osiris_tasks to authenticated;

create policy osiris_projects_owner_insert on public.osiris_projects for insert to authenticated
  with check (owner_id is not null and owner_id = (select auth.uid()));
create policy osiris_projects_owner_update on public.osiris_projects for update to authenticated
  using (owner_id = (select auth.uid())) with check (owner_id = (select auth.uid()));
create policy osiris_projects_owner_delete on public.osiris_projects for delete to authenticated
  using (owner_id = (select auth.uid()));

create policy osiris_tasks_owner_insert on public.osiris_tasks for insert to authenticated
  with check (exists (select 1 from public.osiris_projects p where p.id = project_id and p.owner_id = (select auth.uid())));
create policy osiris_tasks_owner_update on public.osiris_tasks for update to authenticated
  using (exists (select 1 from public.osiris_projects p where p.id = project_id and p.owner_id = (select auth.uid())))
  with check (exists (select 1 from public.osiris_projects p where p.id = project_id and p.owner_id = (select auth.uid())));
create policy osiris_tasks_owner_delete on public.osiris_tasks for delete to authenticated
  using (exists (select 1 from public.osiris_projects p where p.id = project_id and p.owner_id = (select auth.uid())));

create policy osiris_projects_admin_select on public.osiris_projects for select to authenticated using (public.osiris_is_admin());
create policy osiris_tasks_admin_select on public.osiris_tasks for select to authenticated using (public.osiris_is_admin());
create policy osiris_dispositions_admin_select on public.osiris_dispositions for select to authenticated using (public.osiris_is_admin());
