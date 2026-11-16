import { PGlite } from "@electric-sql/pglite"
import { readFileSync, readdirSync } from "node:fs"
const dir = process.argv[2], db = new PGlite(), out = { positive: {}, negative: {}, admin: {} }
await db.exec(`create schema auth; create table auth.users(id uuid primary key);
  create function auth.uid() returns uuid language sql stable as $$ select nullif(current_setting('request.jwt.claim.sub', true), '')::uuid $$;
  create role anon nologin; create role authenticated nologin; create role service_role nologin bypassrls;
  grant usage on schema public, auth to anon, authenticated, service_role; grant execute on function auth.uid() to anon, authenticated, service_role;
  alter default privileges in schema public grant all on tables to anon, authenticated, service_role;`)
for (const f of readdirSync(dir).sort()) await db.exec(readFileSync(dir + "/" + f, "utf8"))
const A = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", B = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb", D = "dddddddd-dddd-4ddd-8ddd-dddddddddddd", h = "a".repeat(64)
await db.exec(`insert into auth.users values ('${A}'),('${B}'),('${D}')`)
const as = async (role, uid, fn) => { await db.exec(`set role ${role}; select set_config('request.jwt.claim.sub', '${uid}', false)`); try { return await fn() } finally { await db.exec("reset role") } }
const q = async (sql, args = []) => { try { const r = await db.query(sql, args); return { ok: true, rows: r.rows, n: r.rows.length || r.affectedRows || 0 } } catch (e) { return { ok: false } } }
const R = (r) => r.ok && r.n > 0 ? `ALLOWED (${r.n})` : r.ok ? "no rows" : "refused"
const DISP = "insert into osiris_dispositions(project_id,council_task_id,objective_sha256,decision,reason,allow_cloud,receipt_kind,receipt_hash) values ($1,$2,$3,'REVISION_REQUESTED','r',false,'agents.task',$3)"
let pa, ta, pb
await as("authenticated", A, async () => {
  const p = await q("insert into osiris_projects(name) values ('A project') returning id, owner_id"); pa = p.rows?.[0]?.id
  out.positive.A_creates_project_owned_by_A = p.ok && p.rows[0].owner_id === A
  const t = await q("insert into osiris_tasks(project_id,title) values ($1,'Run simulator') returning id", [pa]); ta = t.rows?.[0]?.id; out.positive.A_creates_task = t.ok
  out.positive.A_moves_task = R(await q("update osiris_tasks set status='done' where id=$1", [ta]))
  out.positive.A_renames_project = R(await q("update osiris_projects set name='A renamed' where id=$1", [pa]))
})
await as("service_role", "", async () => { out.positive.osiris_appends_disposition = (await q(DISP, [pa, "task-1", h])).ok })
out.positive.A_reads_disposition = await as("authenticated", A, async () => (await q("select decision from osiris_dispositions")).rows)
await as("authenticated", B, async () => { pb = (await q("insert into osiris_projects(name) values ('B project') returning id")).rows[0].id })
const N = out.negative
await as("authenticated", B, async () => {
  N.B_reads_As_rows = (await q("select 1 from osiris_projects where id=$1", [pa])).n + (await q("select 1 from osiris_tasks where project_id=$1", [pa])).n + (await q("select 1 from osiris_dispositions where project_id=$1", [pa])).n
  N.B_updates_As_project = R(await q("update osiris_projects set name='x' where id=$1", [pa]))
  N.B_deletes_As_project = R(await q("delete from osiris_projects where id=$1", [pa]))
  N.B_takes_As_project = R(await q("update osiris_projects set owner_id=$2 where id=$1", [pa, B]))
  N.B_gives_own_project_to_A = R(await q("update osiris_projects set owner_id=$2 where id=$1", [pb, A]))
  N.B_creates_project_owned_by_A = R(await q("insert into osiris_projects(name,owner_id) values ('x',$1)", [A]))
  N.B_creates_ownerless_project = R(await q("insert into osiris_projects(name,owner_id) values ('x',null)"))
  N.B_adds_task_to_As_project = R(await q("insert into osiris_tasks(project_id,title) values ($1,'x')", [pa]))
  N.B_moves_own_task_into_As_project = R(await (async () => { const t = (await q("insert into osiris_tasks(project_id,title) values ($1,'mine') returning id", [pb])).rows[0].id; return q("update osiris_tasks set project_id=$1 where id=$2", [pa, t]) })())
  N.B_edits_As_task = R(await q("update osiris_tasks set title='x' where id=$1", [ta]))
  N.B_writes_disposition = R(await q(DISP, [pb, "forged", h]))
  N.B_makes_self_admin = R(await q("insert into osiris_admins(user_id) values ($1)", [B]))
  N.B_reads_admin_list = R(await q("select * from osiris_admins"))
  N.B_is_admin = (await q("select public.osiris_is_admin() v")).rows?.[0]?.v
})
await as("authenticated", A, async () => { N.A_edits_own_disposition = R(await q("update osiris_dispositions set decision='PENDING_HUMAN_APPROVAL'")); N.A_deletes_own_disposition = R(await q("delete from osiris_dispositions")) })
await as("anon", "", async () => { N.anon_reads = R(await q("select 1 from osiris_projects")); N.anon_writes = R(await q("insert into osiris_projects(name) values ('x')")); N.anon_calls_is_admin = R(await q("select public.osiris_is_admin()")) })
await as("service_role", "", () => q("insert into osiris_admins(user_id) values ($1)", [D]))
await as("authenticated", D, async () => {
  out.admin.D_is_admin = (await q("select public.osiris_is_admin() v")).rows?.[0]?.v
  out.admin.D_sees_all = [(await q("select 1 from osiris_projects")).n, (await q("select 1 from osiris_tasks")).n, (await q("select 1 from osiris_dispositions")).n]
  out.admin.D_edits_As_project = R(await q("update osiris_projects set name='x' where id=$1", [pa]))
  out.admin.D_deletes_As_project = R(await q("delete from osiris_projects where id=$1", [pa]))
  out.admin.D_edits_disposition = R(await q("update osiris_dispositions set decision='PENDING_HUMAN_APPROVAL'"))
})
console.log(JSON.stringify(out, null, 1))
