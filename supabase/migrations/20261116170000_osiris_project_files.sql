-- OSIRIS workspace v4: project files for the DNA-Lang editor. NOT APPLIED until the owner approves it.
--
-- Target: Supabase project "osiris" (ref eirdsaeutzkzepibnfxf).
-- Depends on: 20261116160000_osiris_multi_user_and_admin.sql (owner writes on projects, osiris_is_admin()).
--
-- What this adds:
--   A project-scoped virtual file tree. A file belongs to a project; who may touch it follows from who
--   owns the project. There is no owner column on a file, so a caller has no owner value to forge.
--     owner of the project   select, insert, update, delete
--     admin (osiris_admins)  select only
--     anon                   nothing
--   revision, content_sha256, created_at and updated_at are set by a trigger and cannot be chosen by the
--   caller. A file cannot be moved to another project. Saving is "update where revision = the revision I
--   read": a stale save matches no row, so it cannot overwrite a newer revision.
--   Deleting is a hard delete by the owner; deleting the project removes its files (cascade). Earlier
--   revisions are not kept: this table holds the current text only.
--   Limits: 65536 bytes per file, 200 characters per path, 200 files per project.
--
-- Rollback (the table is new; nothing else depends on it):
--   drop table public.osiris_files;
--   drop function public.osiris_files_guard();

create table public.osiris_files (
  id             uuid primary key default gen_random_uuid(),
  project_id     uuid not null references public.osiris_projects(id) on delete cascade,
  path           text not null
                   check (char_length(path) between 1 and 200)
                   check (path ~ '^[A-Za-z0-9_][A-Za-z0-9_.-]*(/[A-Za-z0-9_][A-Za-z0-9_.-]*)*$')
                   check (position('..' in path) = 0),
  content        text not null default '' check (octet_length(content) <= 65536),
  content_sha256 text not null default '' check (content_sha256 ~ '^[0-9a-f]{64}$'),
  revision       integer not null default 1 check (revision >= 1),
  created_at     timestamptz not null default now(),
  updated_at     timestamptz not null default now(),
  unique (project_id, path)
);

-- Server-owned columns. SECURITY INVOKER: the file count below is taken with the caller's own rights.
create function public.osiris_files_guard() returns trigger
  language plpgsql set search_path = ''
as $$
begin
  if tg_op = 'INSERT' then
    if (select count(*) from public.osiris_files f where f.project_id = new.project_id) >= 200 then
      raise exception 'a project may hold at most 200 files' using errcode = '23514';
    end if;
    new.revision := 1;
    new.created_at := now();
  else
    if new.id <> old.id or new.project_id <> old.project_id then
      raise exception 'a file cannot change identity or project' using errcode = '23514';
    end if;
    new.revision := old.revision + 1;
    new.created_at := old.created_at;
  end if;
  new.updated_at := now();
  new.content_sha256 := pg_catalog.encode(pg_catalog.sha256(pg_catalog.convert_to(new.content, 'UTF8')), 'hex');
  return new;
end $$;
revoke all on function public.osiris_files_guard() from public, anon, authenticated;

create trigger osiris_files_guard before insert or update on public.osiris_files
  for each row execute function public.osiris_files_guard();

alter table public.osiris_files enable row level security;
revoke all on public.osiris_files from anon, authenticated;
grant select, insert, update, delete on public.osiris_files to authenticated;

create policy osiris_files_owner_select on public.osiris_files for select to authenticated
  using (exists (select 1 from public.osiris_projects p where p.id = project_id and p.owner_id = (select auth.uid())));
create policy osiris_files_owner_insert on public.osiris_files for insert to authenticated
  with check (exists (select 1 from public.osiris_projects p where p.id = project_id and p.owner_id = (select auth.uid())));
create policy osiris_files_owner_update on public.osiris_files for update to authenticated
  using (exists (select 1 from public.osiris_projects p where p.id = project_id and p.owner_id = (select auth.uid())))
  with check (exists (select 1 from public.osiris_projects p where p.id = project_id and p.owner_id = (select auth.uid())));
create policy osiris_files_owner_delete on public.osiris_files for delete to authenticated
  using (exists (select 1 from public.osiris_projects p where p.id = project_id and p.owner_id = (select auth.uid())));
create policy osiris_files_admin_select on public.osiris_files for select to authenticated
  using (public.osiris_is_admin());
