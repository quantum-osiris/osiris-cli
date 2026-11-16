// Both migrations, in order, in a fresh in-process Postgres (pglite) with Supabase's three API roles emulated.
import { PGlite } from "@electric-sql/pglite"
import { readFileSync } from "node:fs"
const db = new PGlite(), out = { positive: {}, negative: {}, notes: {} }
await db.exec(`create schema auth; create table auth.users(id uuid primary key);
  create function auth.uid() returns uuid language sql stable as $$ select nullif(current_setting('request.jwt.claim.sub', true), '')::uuid $$;
  create role anon nologin; create role authenticated nologin; create role service_role nologin bypassrls;
  grant usage on schema public, auth to anon, authenticated, service_role;
  alter default privileges in schema public grant all on tables to anon, authenticated, service_role;`)   // Supabase's defaults
await db.exec(readFileSync(process.argv[2], "utf8")); await db.exec(readFileSync(process.argv[3], "utf8"))
const A = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", B = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb", h = "a".repeat(64)
await db.exec(`insert into auth.users values ('${A}'),('${B}')`)
const as = async (role, uid, fn) => { await db.exec(`set role ${role}; select set_config('request.jwt.claim.sub', '${uid}', false)`); try { return await fn() } finally { await db.exec("reset role") } }
const q = async (sql, args = []) => { try { const r = await db.query(sql, args); return { ok: true, rows: r.rows, n: r.rows.length || r.affectedRows || 0 } } catch (e) { return { ok: false, err: e.message.slice(0, 70) } } }
const DISP = "insert into osiris_dispositions(project_id,task_id,council_task_id,objective_sha256,objective_excerpt,decision,reason,allow_cloud,receipt_kind,receipt_hash) values ($1,$2,$3,$4,'Plan a Bell baseline',$5,'r',false,'agents.task',$4) returning id"
// ---- positive: the backend (service role) acts for user A, then A reads through RLS ----
let pa, ta
await as("service_role", "", async () => {
  const p = await q("insert into osiris_projects(name, owner_id) values ('A project', $1) returning id", [A]); pa = p.rows?.[0]?.id; out.positive.backend_creates_project_for_A = p.ok
  const t = await q("insert into osiris_tasks(project_id,title) values ($1,'Run simulator') returning id", [pa]); ta = t.rows?.[0]?.id; out.positive.backend_creates_task = t.ok
  out.positive.backend_moves_task = (await q("update osiris_tasks set status='doing' where id=$1", [ta])).n === 1
  out.positive.osiris_appends_disposition = (await q(DISP, [pa, ta, "task-1", h, "REVISION_REQUESTED"])).ok
  out.positive.backend_reads_it_back = (await q("select decision from osiris_dispositions where project_id=$1", [pa])).rows?.[0]?.decision
})
out.positive.A_reads_own_project_task_disposition = await as("authenticated", A, async () => [(await q("select name from osiris_projects")).rows, (await q("select title,status from osiris_tasks")).rows, (await q("select decision,objective_excerpt,executed from osiris_dispositions")).rows])
// ---- negative ----
let pb; await as("service_role", "", async () => { pb = (await q("insert into osiris_projects(name, owner_id) values ('B project', $1) returning id", [B])).rows[0].id; await q(DISP, [pb, null, "task-2", h, "FAILED"]) })
const N = out.negative
N.B_sees_As_rows = await as("authenticated", B, async () => (await q("select 1 from osiris_projects where id=$1", [pa])).n + (await q("select 1 from osiris_tasks where project_id=$1", [pa])).n + (await q("select 1 from osiris_dispositions where project_id=$1", [pa])).n)
for (const [name, sql, args] of [
  ["B_updates_As_project", "update osiris_projects set name='x' where id=$1", [pa]], ["B_deletes_As_project", "delete from osiris_projects where id=$1", [pa]],
  ["B_takes_ownership_of_As_project", "update osiris_projects set owner_id=$2 where id=$1", [pa, B]], ["B_inserts_project_owned_by_A", "insert into osiris_projects(name,owner_id) values ('x',$1)", [A]],
  ["B_inserts_own_project_directly", "insert into osiris_projects(name,owner_id) values ('x',$1)", [B]], ["B_writes_task_into_As_project", "insert into osiris_tasks(project_id,title) values ($1,'x')", [pa]],
  ["B_forges_disposition", DISP, [pb, null, "forged", h, "PENDING_HUMAN_APPROVAL"]], ["B_edits_own_disposition", "update osiris_dispositions set decision='PENDING_HUMAN_APPROVAL' where project_id=$1", [pb]],
  ["B_deletes_own_disposition", "delete from osiris_dispositions where project_id=$1", [pb]]]) { const r = await as("authenticated", B, () => q(sql, args)); N[name] = r.ok ? `ALLOWED (${r.n} rows)` : "refused" }
N.anon_reads = await as("anon", "", async () => (await q("select 1 from osiris_projects")).ok ? "ALLOWED" : "refused")
N.anon_writes = await as("anon", "", async () => (await q("insert into osiris_projects(name) values ('x')")).ok ? "ALLOWED" : "refused")
await as("service_role", "", async () => {
  N.backend_edits_disposition = (await q("update osiris_dispositions set decision='PENDING_HUMAN_APPROVAL' where council_task_id='task-1'")).ok ? "ALLOWED" : "refused"
  N.backend_retries_same_council_task = (await q(DISP, [pa, ta, "task-1", h, "FAILED"])).ok ? "ALLOWED (duplicate)" : "refused (unique council_task_id)"
  N.disposition_executed_true = (await q("insert into osiris_dispositions(project_id,council_task_id,objective_sha256,decision,reason,allow_cloud,receipt_kind,receipt_hash,executed) values ($1,'t9',$2,'FAILED','r',false,'k',$2,true)", [pa, h])).ok ? "ALLOWED" : "refused"
  N.disposition_decision_APPROVED = (await q(DISP, [pa, ta, "t10", h, "APPROVED"])).ok ? "ALLOWED" : "refused"
  N.disposition_decision_EXECUTED = (await q(DISP, [pa, ta, "t11", h, "EXECUTED"])).ok ? "ALLOWED" : "refused"
  out.notes.backend_deletes_disposition = (await q("delete from osiris_dispositions where council_task_id='task-2'")).ok ? "ALLOWED (append-only blocks UPDATE only)" : "refused"
  out.notes.backend_reassigns_owner = (await q("update osiris_projects set owner_id=$2 where id=$1", [pb, A])).ok ? "ALLOWED (backend is trusted to set owners)" : "refused"
  out.notes.owner_null_project_visible_to_users = (await q("insert into osiris_projects(name) values ('local operator project') returning id")).ok
})
out.notes.A_sees_ownerless_projects = await as("authenticated", A, async () => (await q("select 1 from osiris_projects where owner_id is null")).n)
out.notes.db_owner_can_disable_trigger = (await q("alter table osiris_dispositions disable trigger osiris_dispositions_no_update")).ok
console.log(JSON.stringify(out, null, 1))
