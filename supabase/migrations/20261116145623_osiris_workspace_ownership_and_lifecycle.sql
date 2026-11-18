-- OSIRIS workspace v2: explicit ownership boundary, RLS read policies for authenticated owners,
-- and bounded objective excerpt for UI retrieval.
-- Depends on: osiris_workspace

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
