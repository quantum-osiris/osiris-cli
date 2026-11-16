-- OSIRIS workspace v3 (delta): admin read-only subspace and owner project rename. NOT APPLIED until the owner approves it.
--
-- Target: Supabase project "osiris" (ref eirdsaeutzkzepibnfxf).
-- Depends on 20261116140000_osiris_hosted_workspace.sql (recorded on the hosted project as 20261116154042), which
-- already provides: owner_id default auth.uid(), owner insert policies on projects and tasks, owner task-status
-- update, column-level insert grants, the append-only disposition triggers with a fixed search_path, the proposal
-- queue and osiris_platform_developers. This migration adds only what that one does not.
--
-- What changes:
--   1. A signed-in owner can rename their own project (name, description). owner_id, id and archived stay
--      unwritable for application roles (column grants), so ownership cannot be changed or forged.
--   2. Admins (rows in osiris_admins) can READ every project, task, disposition and proposal request.
--      Admins get no write path. Only the database owner can add or remove an admin; no user can make
--      themselves one (no grants and no policies on osiris_admins).
--   No deletes are granted: dispositions are append-only and reference tasks, so projects are archived, not deleted.
--
-- Rollback:
--   drop policy osiris_proposal_requests_admin_select on public.osiris_proposal_requests;
--   drop policy osiris_dispositions_admin_select on public.osiris_dispositions;
--   drop policy osiris_tasks_admin_select on public.osiris_tasks;
--   drop policy osiris_projects_admin_select on public.osiris_projects;
--   drop policy osiris_projects_owner_update on public.osiris_projects;
--   revoke update (name, description) on public.osiris_projects from authenticated;
--   drop function public.osiris_is_admin();
--   drop table public.osiris_admins;
--   (20261116170000 depends on osiris_is_admin(); roll that back first if it was applied.)

create table public.osiris_admins (
  user_id    uuid primary key references auth.users(id) on delete cascade,
  created_at timestamptz not null default now()
);
alter table public.osiris_admins enable row level security;          -- no policies: database owner only
revoke all on public.osiris_admins from public, anon, authenticated, service_role;

-- True when the caller is an admin. SECURITY DEFINER so it can read osiris_admins, which callers cannot.
create function public.osiris_is_admin() returns boolean
  language sql stable security definer set search_path = ''
  as $$ select exists (select 1 from public.osiris_admins a where a.user_id = (select auth.uid())) $$;
revoke all on function public.osiris_is_admin() from public, anon;
grant execute on function public.osiris_is_admin() to authenticated;

grant update (name, description) on public.osiris_projects to authenticated;
create policy osiris_projects_owner_update on public.osiris_projects for update to authenticated
  using (owner_id = (select auth.uid())) with check (owner_id = (select auth.uid()));

create policy osiris_projects_admin_select on public.osiris_projects
  for select to authenticated using ((select public.osiris_is_admin()));
create policy osiris_tasks_admin_select on public.osiris_tasks
  for select to authenticated using ((select public.osiris_is_admin()));
create policy osiris_dispositions_admin_select on public.osiris_dispositions
  for select to authenticated using ((select public.osiris_is_admin()));
create policy osiris_proposal_requests_admin_select on public.osiris_proposal_requests
  for select to authenticated using ((select public.osiris_is_admin()));
