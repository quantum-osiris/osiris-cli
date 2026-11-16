-- Hosted Mission Control uses each user's Auth JWT. The local council worker
-- retains the service role and claims work through two restricted functions.

alter table public.osiris_projects alter column owner_id set default auth.uid();

revoke all on public.osiris_projects, public.osiris_tasks, public.osiris_dispositions from public;
grant insert (id, name, description) on public.osiris_projects to authenticated;
grant insert (id, project_id, title) on public.osiris_tasks to authenticated;
grant update (status) on public.osiris_tasks to authenticated;
grant select, insert, update on public.osiris_projects, public.osiris_tasks to service_role;
grant select, insert on public.osiris_dispositions to service_role;
revoke update, delete, truncate on public.osiris_dispositions from service_role;

create policy osiris_projects_owner_insert on public.osiris_projects
  for insert to authenticated with check (owner_id = (select auth.uid()));

create policy osiris_tasks_owner_insert on public.osiris_tasks
  for insert to authenticated with check (exists (
    select 1 from public.osiris_projects p
    where p.id = osiris_tasks.project_id and p.owner_id = (select auth.uid()) and not p.archived
  ));

create policy osiris_tasks_owner_update on public.osiris_tasks
  for update to authenticated using (exists (
    select 1 from public.osiris_projects p
    where p.id = osiris_tasks.project_id and p.owner_id = (select auth.uid()) and not p.archived
  )) with check (exists (
    select 1 from public.osiris_projects p
    where p.id = osiris_tasks.project_id and p.owner_id = (select auth.uid()) and not p.archived
  ));

create function public.osiris_touch_task() returns trigger
  language plpgsql set search_path = '' as $$
begin
  new.updated_at := clock_timestamp();
  return new;
end;
$$;
revoke all on function public.osiris_touch_task() from public, anon, authenticated;
create trigger osiris_touch_task before update on public.osiris_tasks
  for each row execute function public.osiris_touch_task();

-- The task reference and project must agree, including for local operator writes.
alter table public.osiris_tasks add constraint osiris_tasks_project_task_unique unique (project_id, id);
alter table public.osiris_dispositions
  drop constraint osiris_dispositions_task_id_fkey,
  add constraint osiris_dispositions_project_task_fkey
    foreign key (project_id, task_id) references public.osiris_tasks (project_id, id);

create or replace function public.osiris_dispositions_append_only() returns trigger
  language plpgsql set search_path = '' as $$
begin
  raise exception 'OSIRIS dispositions are append-only' using errcode = '55000';
end;
$$;
revoke all on function public.osiris_dispositions_append_only() from public, anon, authenticated;
drop trigger if exists osiris_dispositions_no_update on public.osiris_dispositions;
create trigger osiris_dispositions_prevent_changes before update or delete on public.osiris_dispositions
  for each row execute function public.osiris_dispositions_append_only();
create trigger osiris_dispositions_prevent_truncate before truncate on public.osiris_dispositions
  for each statement execute function public.osiris_dispositions_append_only();

create table public.osiris_proposal_requests (
  id uuid primary key default gen_random_uuid(),
  project_id uuid not null,
  task_id uuid not null,
  objective text not null check (char_length(objective) between 1 and 4000 and objective ~ '[^[:space:]]'),
  status text not null default 'queued' check (status in ('queued', 'processing', 'saved')),
  lease_id uuid,
  lease_expires_at timestamptz,
  created_at timestamptz not null default now(),
  constraint osiris_proposal_request_task_fkey foreign key (project_id, task_id)
    references public.osiris_tasks (project_id, id),
  constraint osiris_proposal_request_lease check (
    (status = 'processing' and lease_id is not null and lease_expires_at is not null)
    or (status <> 'processing' and lease_id is null and lease_expires_at is null)
  ),
  unique (project_id, task_id, id)
);
create index osiris_proposal_requests_queue_idx on public.osiris_proposal_requests (created_at, id)
  where status <> 'saved';
create index osiris_proposal_requests_project_idx on public.osiris_proposal_requests (project_id, created_at desc);
create unique index osiris_proposal_requests_active_task_idx on public.osiris_proposal_requests (project_id, task_id)
  where status <> 'saved';

alter table public.osiris_dispositions add column request_id uuid unique,
  add constraint osiris_dispositions_proposal_request_fkey foreign key (project_id, task_id, request_id)
    references public.osiris_proposal_requests (project_id, task_id, id),
  add constraint osiris_dispositions_proposal_request_bound check (
    request_id is null or (task_id is not null and allow_cloud = false)
  );

alter table public.osiris_proposal_requests enable row level security;
revoke all on public.osiris_proposal_requests from public, anon, authenticated, service_role;
grant select (id, project_id, task_id, objective, status, created_at),
  insert (id, project_id, task_id, objective) on public.osiris_proposal_requests to authenticated;

create policy osiris_proposal_requests_owner_select on public.osiris_proposal_requests
  for select to authenticated using (exists (
    select 1 from public.osiris_projects p
    where p.id = osiris_proposal_requests.project_id and p.owner_id = (select auth.uid())
  ));

create policy osiris_proposal_requests_owner_insert on public.osiris_proposal_requests
  for insert to authenticated with check (exists (
    select 1 from public.osiris_projects p
    where p.id = osiris_proposal_requests.project_id and p.owner_id = (select auth.uid()) and not p.archived
  ));

create schema osiris_workspace_private;
revoke all on schema osiris_workspace_private from public, anon, authenticated, service_role;
grant usage on schema osiris_workspace_private to authenticated, service_role;

create function osiris_workspace_private.claim_proposal_request(p_request_id uuid)
  returns table (id uuid, lease_id uuid, objective text)
  language sql security definer set search_path = '' as $$
  with candidate as (
    select r.id from public.osiris_proposal_requests r
    join public.osiris_projects p on p.id = r.project_id
    where not p.archived and (p_request_id is null or r.id = p_request_id)
      and (r.status = 'queued' or (r.status = 'processing' and r.lease_expires_at <= clock_timestamp()))
    order by r.created_at, r.id
    limit 1 for update of r skip locked
  )
  update public.osiris_proposal_requests r
  set status = 'processing', lease_id = gen_random_uuid(), lease_expires_at = clock_timestamp() + interval '30 minutes'
  from candidate where r.id = candidate.id
  returning r.id, r.lease_id, r.objective;
$$;

create function osiris_workspace_private.finish_proposal_request(p_request_id uuid, p_lease_id uuid, p_disposition jsonb)
  returns setof public.osiris_dispositions
  language plpgsql security definer set search_path = '' as $$
declare
  request public.osiris_proposal_requests;
  saved public.osiris_dispositions;
begin
  select * into request from public.osiris_proposal_requests where id = p_request_id for update;
  if not found or request.status <> 'processing' or p_lease_id is null
     or request.lease_id <> p_lease_id or request.lease_expires_at <= clock_timestamp() then
    raise exception 'OSIRIS proposal lease is unavailable' using errcode = '55000';
  end if;
  if p_disposition -> 'allow_cloud' is distinct from 'false'::jsonb
     or p_disposition -> 'executed' is distinct from 'false'::jsonb then
    raise exception 'OSIRIS proposal must be local and proposal-only' using errcode = '22023';
  end if;
  insert into public.osiris_dispositions (
    project_id, task_id, request_id, council_task_id, objective_sha256, objective_excerpt, decision,
    reason, aiden_verdict, allow_cloud, receipt_kind, receipt_hash, routes, executed
  ) values (
    request.project_id, request.task_id, request.id, p_disposition ->> 'council_task_id',
    p_disposition ->> 'objective_sha256', p_disposition ->> 'objective_excerpt',
    p_disposition ->> 'decision', p_disposition ->> 'reason', p_disposition ->> 'aiden_verdict',
    false, p_disposition ->> 'receipt_kind', p_disposition ->> 'receipt_hash', p_disposition -> 'routes', false
  ) returning * into saved;
  update public.osiris_proposal_requests
  set status = 'saved', lease_id = null, lease_expires_at = null where id = request.id;
  return next saved;
end;
$$;

revoke all on function osiris_workspace_private.claim_proposal_request(uuid),
  osiris_workspace_private.finish_proposal_request(uuid, uuid, jsonb) from public, anon, authenticated;
grant execute on function osiris_workspace_private.claim_proposal_request(uuid),
  osiris_workspace_private.finish_proposal_request(uuid, uuid, jsonb) to service_role;

create function public.osiris_claim_proposal_request(p_request_id uuid default null)
  returns table (id uuid, lease_id uuid, objective text)
  language sql security invoker set search_path = '' as $$
  select * from osiris_workspace_private.claim_proposal_request(p_request_id);
$$;

create function public.osiris_finish_proposal_request(p_request_id uuid, p_lease_id uuid, p_disposition jsonb)
  returns setof public.osiris_dispositions
  language sql security invoker set search_path = '' as $$
  select * from osiris_workspace_private.finish_proposal_request(p_request_id, p_lease_id, p_disposition);
$$;

revoke all on function public.osiris_claim_proposal_request(uuid),
  public.osiris_finish_proposal_request(uuid, uuid, jsonb) from public, anon, authenticated;
grant execute on function public.osiris_claim_proposal_request(uuid),
  public.osiris_finish_proposal_request(uuid, uuid, jsonb) to service_role;

-- A database owner registers the developer's verified Auth user ID here.
create table public.osiris_platform_developers (
  user_id uuid primary key references auth.users(id) on delete cascade
);
alter table public.osiris_platform_developers enable row level security;
revoke all on public.osiris_platform_developers from public, anon, authenticated, service_role;
grant select on public.osiris_platform_developers to authenticated;
create policy osiris_platform_developers_self_select on public.osiris_platform_developers
  for select to authenticated using (user_id = (select auth.uid()));

create function osiris_workspace_private.developer_status()
  returns table (queued bigint, processing bigint, retryable bigint, saved bigint)
  language plpgsql security definer set search_path = '' as $$
begin
  if not exists (select 1 from public.osiris_platform_developers d where d.user_id = auth.uid()) then
    raise insufficient_privilege using message = 'Developer access required';
  end if;
  return query select
    count(*) filter (where r.status = 'queued'),
    count(*) filter (where r.status = 'processing' and r.lease_expires_at > statement_timestamp()),
    count(*) filter (where r.status = 'processing' and r.lease_expires_at <= statement_timestamp()),
    count(*) filter (where r.status = 'saved')
  from public.osiris_proposal_requests r;
end;
$$;
revoke all on function osiris_workspace_private.developer_status() from public, anon, service_role;
grant execute on function osiris_workspace_private.developer_status() to authenticated;

create function public.osiris_developer_status()
  returns table (queued bigint, processing bigint, retryable bigint, saved bigint)
  language sql security invoker set search_path = '' as $$
  select * from osiris_workspace_private.developer_status();
$$;
revoke all on function public.osiris_developer_status() from public, anon, service_role;
grant execute on function public.osiris_developer_status() to authenticated;
