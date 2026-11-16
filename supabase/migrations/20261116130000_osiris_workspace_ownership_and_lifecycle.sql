-- OSIRIS workspace v2: explicit ownership boundary, RLS read policies for authenticated owners,
-- and bounded objective excerpt for UI retrieval.
--
-- Target: Supabase project "osiris" (ref eirdsaeutzkzepibnfxf).
-- Depends on: 20261116120000_osiris_workspace.sql
--
-- Security boundaries:
--   1. Developer MCP access: schema inspection/migrations only; never used at application runtime.
--   2. OSIRIS backend (service_role): sole writer via PostgREST (`OSIRIS_SUPABASE_SERVICE_KEY`).
--   3. End-user auth (`authenticated` JWT): read-only access strictly scoped to rows whose project
--      has `owner_id = auth.uid()`. Rows with `owner_id is null` (local operator backend mode) remain
--      invisible to `authenticated` and `anon`. `anon` retains zero privileges.
--
-- Rollback:
--   drop policy if exists osiris_dispositions_owner_select on public.osiris_dispositions;
--   drop policy if exists osiris_tasks_owner_select on public.osiris_tasks;
--   drop policy if exists osiris_projects_owner_select on public.osiris_projects;
--   revoke select on public.osiris_projects, public.osiris_tasks, public.osiris_dispositions from authenticated;
--   alter table public.osiris_dispositions drop column if exists objective_excerpt;
--   drop index if exists public.osiris_projects_owner_idx;
--   alter table public.osiris_projects drop column if exists owner_id;

alter table public.osiris_projects
  add column if not exists owner_id uuid references auth.users(id) on delete restrict;

create index if not exists osiris_projects_owner_idx
  on public.osiris_projects (owner_id)
  where owner_id is not null;

alter table public.osiris_dispositions
  add column if not exists objective_excerpt text not null default ''
    check (char_length(objective_excerpt) <= 280);

-- Keep anon completely locked out; allow authenticated end users to SELECT only their own rows.
revoke all on public.osiris_projects, public.osiris_tasks, public.osiris_dispositions from anon;
grant select on public.osiris_projects, public.osiris_tasks, public.osiris_dispositions to authenticated;

create policy osiris_projects_owner_select
  on public.osiris_projects
  for select
  to authenticated
  using (owner_id is not null and owner_id = auth.uid());

create policy osiris_tasks_owner_select
  on public.osiris_tasks
  for select
  to authenticated
  using (
    exists (
      select 1
      from public.osiris_projects p
      where p.id = osiris_tasks.project_id
        and p.owner_id is not null
        and p.owner_id = auth.uid()
    )
  );

create policy osiris_dispositions_owner_select
  on public.osiris_dispositions
  for select
  to authenticated
  using (
    exists (
      select 1
      from public.osiris_projects p
      where p.id = osiris_dispositions.project_id
        and p.owner_id is not null
        and p.owner_id = auth.uid()
    )
  );
