// LOCAL DEVELOPMENT STAND-IN for the two Supabase services the OSIRIS editor uses.
//
//   node server.mjs --port 54321 --migrations ../migrations --state <dir> [--pglite <node_modules dir>]
//
// What is real: PostgreSQL (PGlite, in process) with every file in supabase/migrations applied in
// order, so the tables, triggers, grants and row level security policies are the ones written for
// the hosted project. Each REST request runs as the role in its token with auth.uid() set from it.
//
// What is a stand-in: the HTTP layer. It implements only the part of Supabase Auth (email one-time
// code) and PostgREST (single-table select / insert / update / delete with `eq` filters) that
// osiris_cli/editor_store.py calls. It sends NO email: the one-time code is written to
// <state>/outbox.log for the developer to read. It binds to 127.0.0.1 only.
//
// Passing a test against this server shows the SQL behaves as written. It does not show that the
// hosted project is configured, that email is delivered, or that hosted redirects work.
import { createServer } from "node:http"
import { createHmac, randomBytes, randomInt, randomUUID, timingSafeEqual, createHash } from "node:crypto"
import { appendFileSync, existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs"
import { createRequire } from "node:module"
import { join, resolve } from "node:path"
import { pathToFileURL } from "node:url"

const arg = (name, fallback) => { const i = process.argv.indexOf(`--${name}`); return i > 0 ? process.argv[i + 1] : fallback }
const PORT = Number(arg("port", "54321"))
const MIGRATIONS = resolve(arg("migrations", new URL("../migrations", import.meta.url).pathname))
const STATE = resolve(arg("state", ".local-stack"))
const PGLITE_DIR = arg("pglite", process.env.OSIRIS_PGLITE_NODE_MODULES)
const MEMORY = process.argv.includes("--memory")

const require = createRequire(PGLITE_DIR ? join(resolve(PGLITE_DIR), "x.js") : import.meta.url)
const { PGlite } = await import(pathToFileURL(require.resolve("@electric-sql/pglite")).href)

mkdirSync(STATE, { recursive: true, mode: 0o700 })
const secretFile = join(STATE, "jwt.secret"), anonFile = join(STATE, "anon.key"), outbox = join(STATE, "outbox.log")
if (!existsSync(secretFile)) writeFileSync(secretFile, randomBytes(32).toString("hex"), { mode: 0o600 })
const SECRET = readFileSync(secretFile, "utf8").trim()

const b64 = (v) => Buffer.from(v).toString("base64url")
const sign = (claims) => {
  const head = b64(JSON.stringify({ alg: "HS256", typ: "JWT" })), body = b64(JSON.stringify(claims))
  return `${head}.${body}.${createHmac("sha256", SECRET).update(`${head}.${body}`).digest("base64url")}`
}
const verify = (token) => {
  const parts = typeof token === "string" ? token.split(".") : []
  if (parts.length !== 3) return null
  const expected = createHmac("sha256", SECRET).update(`${parts[0]}.${parts[1]}`).digest()
  const given = Buffer.from(parts[2], "base64url")
  if (given.length !== expected.length || !timingSafeEqual(given, expected)) return null
  let claims
  try { claims = JSON.parse(Buffer.from(parts[1], "base64url").toString("utf8")) } catch { return null }
  if (typeof claims.exp === "number" && claims.exp < Date.now() / 1000) return null
  return claims
}
const ANON = sign({ role: "anon", iss: "osiris-local-stack" })
writeFileSync(anonFile, ANON, { mode: 0o600 })

const db = new PGlite(MEMORY ? undefined : join(STATE, "pgdata"))
await db.exec(`
  create schema if not exists auth;
  create table if not exists auth.users (id uuid primary key, email text unique);
  create or replace function auth.uid() returns uuid language sql stable
    as $$ select nullif(current_setting('request.jwt.claim.sub', true), '')::uuid $$;
  do $$ begin
    if not exists (select 1 from pg_roles where rolname = 'anon') then create role anon nologin; end if;
    if not exists (select 1 from pg_roles where rolname = 'authenticated') then create role authenticated nologin; end if;
    if not exists (select 1 from pg_roles where rolname = 'service_role') then create role service_role nologin bypassrls; end if;
  end $$;
  grant usage on schema public, auth to anon, authenticated, service_role;
  grant execute on function auth.uid() to anon, authenticated, service_role;
  alter default privileges in schema public grant all on tables to anon, authenticated, service_role;
  create table if not exists auth.local_stack_migrations (name text primary key, sha256 text not null, applied_at timestamptz default now());
  create table if not exists auth.local_codes (email text primary key, code_sha256 text not null, expires_at timestamptz not null, attempts int not null default 0);
  create table if not exists auth.local_sessions (id uuid primary key, user_id uuid not null, refresh_sha256 text not null unique);
`)
const applied = []
for (const name of readdirSync(MIGRATIONS).filter((f) => f.endsWith(".sql")).sort()) {
  const sql = readFileSync(join(MIGRATIONS, name), "utf8"), sha = createHash("sha256").update(sql).digest("hex")
  const seen = await db.query("select sha256 from auth.local_stack_migrations where name = $1", [name])
  if (seen.rows.length) {
    if (seen.rows[0].sha256 !== sha) throw new Error(`${name} changed after it was applied here; delete ${STATE}/pgdata to rebuild`)
    continue
  }
  await db.transaction(async (tx) => { await tx.exec(sql); await tx.query("insert into auth.local_stack_migrations(name, sha256) values ($1, $2)", [name, sha]) })
  applied.push(name)
}

const sha = (text) => createHash("sha256").update(text).digest("hex")
const IDENT = /^[a-z_][a-z0-9_]{0,62}$/
const TABLE = /^osiris_[a-z_]{1,50}$/
class HttpError extends Error { constructor(status, code, message) { super(message); this.status = status; this.code = code } }
const sqlError = (e) => {
  const c = e?.code ?? ""
  if (c === "23505" || c === "23503") return new HttpError(409, c, "conflict")
  if (c === "42501") return new HttpError(403, c, "permission denied")
  if (c.startsWith("23") || c === "P0001" || c.startsWith("22")) return new HttpError(400, c, "rejected by a constraint")
  if (c === "42P01" || c === "42703") return new HttpError(404, c, "no such table or column")
  return new HttpError(500, c || "XX000", "database error")
}

/** One statement as the caller's role, with auth.uid() taken from the token. Serialised: PGlite is one connection. */
let queue = Promise.resolve()
function asCaller(claims, sql, params) {
  const run = async () => {
    const role = claims.role === "authenticated" ? "authenticated" : "anon"
    try {
      return await db.transaction(async (tx) => {
        await tx.query("select set_config('request.jwt.claim.sub', $1, true)", [role === "authenticated" ? String(claims.sub) : ""])
        await tx.exec(`set local role ${role}`)
        return (await tx.query(sql, params)).rows
      })
    } catch (e) { throw e instanceof HttpError ? e : sqlError(e) }
  }
  const result = queue.then(run, run)
  queue = result.catch(() => {})
  return result
}

function columns(select) {
  if (!select || select === "*") return "*"
  const cols = select.split(",")
  if (!cols.every((c) => IDENT.test(c))) throw new HttpError(400, "PGRST100", "unsupported select")
  return cols.map((c) => `"${c}"`).join(", ")
}

async function rest(method, table, query, claims, body) {
  if (!TABLE.test(table)) throw new HttpError(404, "PGRST205", "no such table")
  const where = [], params = []
  let select = null, order = "", limit = ""
  for (const [key, value] of query) {
    if (key === "select") select = value
    else if (key === "order") {
      const [col, dir = "asc"] = value.split(".")
      if (!IDENT.test(col) || !["asc", "desc"].includes(dir)) throw new HttpError(400, "PGRST100", "unsupported order")
      order = ` order by "${col}" ${dir}`
    } else if (key === "limit") {
      if (!/^\d{1,4}$/.test(value)) throw new HttpError(400, "PGRST100", "unsupported limit")
      limit = ` limit ${Number(value)}`
    } else {
      if (!IDENT.test(key) || !value.startsWith("eq.")) throw new HttpError(400, "PGRST100", "only eq filters are implemented")
      params.push(value.slice(3)); where.push(`"${key}"::text = $${params.length}`)
    }
  }
  const cols = columns(select), cond = where.length ? ` where ${where.join(" and ")}` : ""
  if (method === "GET") return asCaller(claims, `select ${cols} from public."${table}"${cond}${order}${limit}`, params)
  if (method === "DELETE") {
    if (!where.length) throw new HttpError(400, "21000", "DELETE requires a filter")
    return asCaller(claims, `delete from public."${table}"${cond} returning ${cols}`, params)
  }
  if (!body || typeof body !== "object" || Array.isArray(body)) throw new HttpError(400, "PGRST102", "expected one JSON object")
  const keys = Object.keys(body)
  if (!keys.length || !keys.every((k) => IDENT.test(k))) throw new HttpError(400, "PGRST102", "unsupported columns")
  const value = (v) => (v !== null && typeof v === "object" ? JSON.stringify(v) : v)
  if (method === "POST") {
    const start = params.length
    keys.forEach((k) => params.push(value(body[k])))
    return asCaller(claims, `insert into public."${table}" (${keys.map((k) => `"${k}"`).join(", ")}) values (${keys.map((_, i) => `$${start + i + 1}`).join(", ")}) returning ${cols}`, params)
  }
  if (method === "PATCH") {
    if (!where.length) throw new HttpError(400, "21000", "UPDATE requires a filter")
    const sets = keys.map((k) => { params.push(value(body[k])); return `"${k}" = $${params.length}` })
    return asCaller(claims, `update public."${table}" set ${sets.join(", ")}${cond} returning ${cols}`, params)
  }
  throw new HttpError(405, "PGRST117", "unsupported method")
}

const EMAIL = /^[^@\s]{1,64}@[^@\s]{1,255}$/
async function session(userId, email) {
  const id = randomUUID(), refresh = randomBytes(24).toString("base64url")
  await db.query("insert into auth.local_sessions(id, user_id, refresh_sha256) values ($1, $2, $3)", [id, userId, sha(refresh)])
  const expires_in = 3600
  const access_token = sign({ role: "authenticated", sub: userId, email, session_id: id, iss: "osiris-local-stack", exp: Math.floor(Date.now() / 1000) + expires_in })
  return { access_token, token_type: "bearer", expires_in, refresh_token: refresh, user: { id: userId, email } }
}

async function auth(method, path, query, body, bearer) {
  if (method === "POST" && path === "/otp") {
    const email = String(body?.email ?? "").toLowerCase()
    if (!EMAIL.test(email)) throw new HttpError(400, "validation_failed", "email required")
    await db.query("insert into auth.users(id, email) values ($1, $2) on conflict (email) do nothing", [randomUUID(), email])
    const code = String(randomInt(0, 1_000_000)).padStart(6, "0")
    await db.query(`insert into auth.local_codes(email, code_sha256, expires_at, attempts) values ($1, $2, now() + interval '10 minutes', 0)
                    on conflict (email) do update set code_sha256 = excluded.code_sha256, expires_at = excluded.expires_at, attempts = 0`, [email, sha(code)])
    appendFileSync(outbox, `${new Date().toISOString()} LOCAL STAND-IN, NO EMAIL SENT  to=${email}  code=${code}\n`, { mode: 0o600 })
    return {}
  }
  if (method === "POST" && path === "/verify") {
    const email = String(body?.email ?? "").toLowerCase(), token = String(body?.token ?? "")
    const row = (await db.query("update auth.local_codes set attempts = attempts + 1 where email = $1 returning code_sha256, expires_at > now() as live, attempts", [email])).rows[0]
    if (!row || !row.live || row.attempts > 5 || row.code_sha256 !== sha(token)) throw new HttpError(403, "otp_expired", "code is invalid or has expired")
    await db.query("delete from auth.local_codes where email = $1", [email])
    const user = (await db.query("select id, email from auth.users where email = $1", [email])).rows[0]
    return session(user.id, user.email)
  }
  if (method === "POST" && path === "/token" && query.get("grant_type") === "refresh_token") {
    const old = (await db.query("delete from auth.local_sessions where refresh_sha256 = $1 returning user_id", [sha(String(body?.refresh_token ?? ""))])).rows[0]
    if (!old) throw new HttpError(400, "refresh_token_not_found", "invalid refresh token")
    const user = (await db.query("select id, email from auth.users where id = $1", [old.user_id])).rows[0]
    return session(user.id, user.email)
  }
  const claims = verify(bearer)
  const live = claims?.role === "authenticated" && (await db.query("select 1 from auth.local_sessions where id = $1", [claims.session_id])).rows.length === 1
  if (method === "GET" && path === "/user") {
    if (!live) throw new HttpError(401, "bad_jwt", "no session")
    return { id: claims.sub, email: claims.email, role: "authenticated" }
  }
  if (method === "POST" && path === "/logout") {
    if (!live) throw new HttpError(401, "bad_jwt", "no session")
    await db.query("delete from auth.local_sessions where id = $1", [claims.session_id])
    return {}
  }
  throw new HttpError(404, "not_found", "not implemented in the local stand-in")
}

const server = createServer(async (req, res) => {
  const send = (status, payload) => { const data = JSON.stringify(payload); res.writeHead(status, { "Content-Type": "application/json", "Content-Length": Buffer.byteLength(data), "Cache-Control": "no-store" }); res.end(data) }
  try {
    const host = (req.headers.host ?? "").toLowerCase()
    if (host !== `127.0.0.1:${PORT}` && host !== `localhost:${PORT}`) throw new HttpError(421, "host", "loopback only")
    const url = new URL(req.url, `http://127.0.0.1:${PORT}`)
    let raw = ""
    for await (const chunk of req) { raw += chunk; if (raw.length > 300_000) throw new HttpError(413, "too_large", "body too large") }
    let body = null
    if (raw) { try { body = JSON.parse(raw) } catch { throw new HttpError(400, "bad_json", "invalid JSON") } }
    const key = verify(req.headers.apikey)
    if (!key || key.iss !== "osiris-local-stack") throw new HttpError(401, "no_api_key", "apikey required")
    const bearer = (req.headers.authorization ?? "").replace(/^Bearer /, "")
    if (url.pathname.startsWith("/auth/v1/")) return send(200, await auth(req.method, url.pathname.slice(8), url.searchParams, body, bearer))
    if (url.pathname.startsWith("/rest/v1/")) {
      const claims = verify(bearer)
      if (!claims) throw new HttpError(401, "PGRST301", "JWT invalid or expired")
      if (claims.role === "service_role") throw new HttpError(403, "local", "the local stand-in issues no service role")
      const rows = await rest(req.method, url.pathname.slice(9), [...url.searchParams], claims, body)
      return send(req.method === "POST" ? 201 : 200, rows)
    }
    throw new HttpError(404, "not_found", "not found")
  } catch (e) {
    const err = e instanceof HttpError ? e : new HttpError(500, "internal", "internal error")
    if (!(e instanceof HttpError)) console.error("[local-stack] internal:", e?.message)
    send(err.status, { code: err.code, message: err.message })
  }
})
server.listen(PORT, "127.0.0.1", () => {
  console.log(`[local-stack] LOCAL STAND-IN on http://127.0.0.1:${PORT} (not the hosted Supabase project)`)
  console.log(`[local-stack] migrations applied now: ${applied.length ? applied.join(", ") : "none (already applied)"}; database: ${MEMORY ? "memory" : join(STATE, "pgdata")}`)
  console.log(`[local-stack] publishable key file: ${anonFile}; sign-in codes: ${outbox} (no email is sent)`)
})
