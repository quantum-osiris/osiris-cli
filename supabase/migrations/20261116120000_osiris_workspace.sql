-- OSIRIS workspace: projects, tasks and the recorded disposition of each OSIRIS council task.
--
-- Target: Supabase project "osiris" (ref eirdsaeutzkzepibnfxf), empty when this was written.
-- Access: row level security is on and there are no policies, and the API roles are revoked, so
-- neither the anon key nor a signed-in user can read or write these tables. Only the OSIRIS
-- backend (service role) can. End-user access is a later, separate migration.
-- Rollback: drop the three tables (they are new; nothing else depends on them):
--   drop table public.osiris_dispositions, public.osiris_tasks, public.osiris_projects;

create table public.osiris_projects (
  id          uuid primary key default gen_random_uuid(),
  name        text not null check (char_length(name) between 1 and 200),
  description text not null default '' check (char_length(description) <= 4000),
  archived    boolean not null default false,
  created_at  timestamptz not null default now(),
  updated_at  timestamptz not null default now()
);

create table public.osiris_tasks (
  id          uuid primary key default gen_random_uuid(),
  project_id  uuid not null references public.osiris_projects(id) on delete cascade,
  title       text not null check (char_length(title) between 1 and 400),
  status      text not null default 'todo' check (status in ('todo', 'doing', 'done')),
  created_at  timestamptz not null default now(),
  updated_at  timestamptz not null default now()
);
create index osiris_tasks_project_idx on public.osiris_tasks (project_id);

-- One row per council run. It stores digests and references, not the model text. `executed` can
-- only be false: a disposition is a workflow outcome, never an execution record.
create table public.osiris_dispositions (
  id               uuid primary key default gen_random_uuid(),
  project_id       uuid not null references public.osiris_projects(id) on delete cascade,
  task_id          uuid references public.osiris_tasks(id) on delete set null,
  council_task_id  text not null unique,
  objective_sha256 text not null check (objective_sha256 ~ '^[0-9a-f]{64}$'),
  decision         text not null check (decision in ('PENDING_HUMAN_APPROVAL', 'REVISION_REQUESTED',
                     'HELD_BY_REVIEW', 'QUARANTINED', 'REJECTED', 'FAILED')),
  reason           text not null,
  aiden_verdict    text check (aiden_verdict in ('PROCEED', 'REVISE', 'STOP')),
  allow_cloud      boolean not null,
  receipt_kind     text not null,
  receipt_hash     text not null check (receipt_hash ~ '^[0-9a-f]{64}$'),
  routes           jsonb not null default '[]'::jsonb,
  executed         boolean not null default false check (executed = false),
  created_at       timestamptz not null default now()
);
create index osiris_dispositions_project_idx on public.osiris_dispositions (project_id, created_at desc);

alter table public.osiris_projects     enable row level security;
alter table public.osiris_tasks        enable row level security;
alter table public.osiris_dispositions enable row level security;

revoke all on public.osiris_projects, public.osiris_tasks, public.osiris_dispositions from anon, authenticated;
