# MCP Specification 2026-07-28 — Impact Analysis

> **Date:** 2026-07-30
> **Subject:** Model Context Protocol specification revision 2026-07-28 (finalized Jul 28, 2026)
> **Sources:** [Official changelog](https://modelcontextprotocol.io/specification/2026-07-28/changelog), [Blog post](https://blog.modelcontextprotocol.io/posts/2026-07-28), [Deprecated features](https://modelcontextprotocol.io/specification/2026-07-28/deprecated), [FastMCP v4 docs](https://gofastmcp.com/getting-started/whats-new), [Python SDK v2.0.0](https://github.com/modelcontextprotocol/python-sdk/releases)

---

## What Changed

### ⚡ Architecture: Stateless Protocol Core (Breaking)

The most fundamental change: **MCP is now fully stateless at the protocol layer**.

- Removed `initialize` / `notifications/initialized` handshake (SEP-2575)
- Removed `Mcp-Session-Id` header (SEP-2567)
- Removed `ping` — no longer a core method
- Every request carries protocol version, client identity, and capabilities in `_meta`
- New **`server/discover`** RPC — servers MUST implement it for up-front capability discovery
- **STDIO transport**: not changed architecturally (still subprocess stdin/stdout), but the wire negotiation changes to per-request metadata

**Impact on this project:** Low on the surface — FastMCP handles negotiation internally. However, the `ctx.sample()` API (which this project uses) is **removed** in FastMCP 4.0 because the stateless protocol no longer supports server-initiated requests back to the client.

### 🔄 Multi Round-Trip Requests (MRTR)

Replaces the old server-initiated request pattern. When a tool needs user input mid-call, the server returns `resultType: "input_required"` with `InputRequired` requests, and the client retries the original call with answers attached.

- Deprecates the old push-based `elicitation/create`, `sampling/createMessage`, `roots/list`
- All results now carry a required `resultType` field (`"complete"` or `"input_required"`)
- Absent `resultType` on legacy servers is treated as `"complete"` (backward compat)

**Impact on this project:** Not directly relevant currently — this project doesn't do server-initiated requests. But the MRTR pattern is how FastMCP v4 replaces `ctx.sample()` for servers that need the client's LLM.

### 📡 Transport Layer

| Change | Detail |
|--------|--------|
| HTTP GET stream | Removed — no more standalone SSE endpoint |
| `Mcp-Session-Id` | Removed entirely |
| SSE resumability | Removed (`Last-Event-ID`, SSE event IDs) |
| `Mcp-Method`, `Mcp-Name` headers | **Required** on Streamable HTTP POST requests (SEP-2243) |
| Custom headers | `x-mcp-header` annotation mirrors parameter values into HTTP headers |
| `subscriptions/listen` | Replaces old HTTP GET + `resources/subscribe`/`unsubscribe` |

**Impact on this project:** None. This project uses **stdio transport exclusively** (not Streamable HTTP). No changes needed to transport code.

### ❌ Deprecated Features (12-month offramp)

| Feature | Replaced by | Earliest removal |
|---------|------------|-----------------|
| **Sampling** | Direct LLM provider API calls | 2027-07-28 |
| **Roots** | Tool params, resource URIs, or server config | 2027-07-28 |
| **Logging** protocol | stderr (stdio) or OpenTelemetry | 2027-07-28 |
| **Dynamic Client Registration (DCR)** | Client ID Metadata Documents | 2027-07-28 |
| HTTP+SSE transport | Streamable HTTP | Q2-Q3 2026 |
| `includeContext: "thisServer"/"allServers"` | Use `"none"` | Follows Sampling removal |

**Impact on this project: HIGH.** This project uses **Sampling** (`ctx.sample()`) in `search_tools.py` for AI-powered answer synthesis. Sampling is deprecated and will be removed no earlier than 2027-07-28. **FastMCP 4.0 removes `ctx.sample()` entirely**, even on handshake-era connections — servers must migrate to direct LLM API calls or the MRTR pattern.

### 🔢 Error Code Renumbering

| Code | Error | Old code |
|------|-------|---------|
| -32020 | HeaderMismatch | -32001 |
| -32021 | MissingRequiredClientCapability | -32003 |
| -32022 | UnsupportedProtocolVersion | -32004 |
| -32602 | Resource not found (now Invalid Params) | -32002 |

**Impact on this project:** Low. No custom error-code matching in the codebase. SDK handles transparently.

### 📝 Schema Changes

- Default dialect: **JSON Schema 2020-12** (was draft-07)
- `inputSchema` and `outputSchema` allow any JSON Schema 2020-12 keywords
- New `outputSchema` field on `Tool` definitions
- New `structuredContent` return value on `CallToolResult`
- SDK v2 renamed all fields: **camelCase → snake_case** (`inputSchema` → `input_schema`, `mimeType` → `mime_type`, `isError` → `is_error`)

### 🧩 Extensions Framework

MCP now supports official extensions (reverse-DNS identifiers):
- **`io.modelcontextprotocol/tasks`** — async task execution with polling
- **`io.modelcontextprotocol/ui`** (MCP Apps) — interactive UI elements
- **OAuth Client Credentials** — machine-to-machine auth
- **Enterprise-Managed Authorization** — centralized access control

### 🔐 Authorization Hardening

- `iss` parameter validation per RFC 9207 (clients MUST validate)
- `application_type` required during DCR
- Client credentials bound to issuer
- DCR deprecated in favor of Client ID Metadata Documents

---

## Impact on `mcp-plesk-dev-docs`

### Current State

| Aspect | Current value |
|--------|-------------|
| MCP Framework | **FastMCP 3.2.4** |
| Python MCP SDK | **mcp 1.27.0** |
| Transport | **stdio only** |
| Sampling used? | **Yes** — `ctx.sample()` in `search_tools.py` (gated by `plesk_enable_sampling`) |
| Sessions/roots/OAuth | **None** |
| Protocol version | Handled by SDK internally |
| File impacted | `search_tools.py:22,49-57,96` — `SamplingMessage`, `TextContent`, `ctx.sample()` |

### Impact Matrix

| Change | Severity | Action required? | Urgency |
|--------|----------|-----------------|---------|
| Sampling deprecated | 🔴 **High** | Migrate `ctx.sample()` to direct LLM API or MRTR pattern | By FastMCP v4 upgrade |
| `ctx.sample()` removed in FastMCP 4 | 🔴 **High** | Blocking upgrade to FastMCP 4 until sampling use is replaced | When ready to upgrade |
| `mcp.types` imports change | 🟡 Medium | `SamplingMessage`/`TextContent` paths may change in SDK v2 | On SDK v2 upgrade |
| snake_case field names | 🟡 Medium | `inputSchema` → `input_schema`, `isError` → `is_error` | On SDK v2 upgrade |
| Stateless core | 🟢 Low | No sessions used; transparent via FastMCP | No action |
| `server/discover` | 🟢 Low | FastMCP handles internally | No action |
| Cache hints | 🟢 Low | Not using list endpoints directly | Nice-to-have |
| Error code renumbering | 🟢 Low | No custom error-code matching | No action |
| Authorization changes | 🟢 Low | No OAuth configured | No action |
| Extensions framework | 🟢 Low | Not using extensions | No action |

---

## Migration Roadmap

### Short-term (Now — Q4 2026)

1. **Pin the current stack**: Keep `fastmcp>=3.2.4,<4` and `mcp>=1.27,<2` to avoid accidental v2/v4 upgrades until migration is planned
2. **Plan sampling replacement**: Evaluate options:
   - **Option A** — Direct API calls: Replace `ctx.sample()` with direct calls to the LLM provider (OpenAI, Anthropic, OpenRouter) from `search_tools.py`. This removes the dependency on the client's model and gives the server full control over model selection.
   - **Option B** — MRTR pattern (FastMCP v4): Return `InputRequiredResult` with a sampling request, rely on the client to respond. Requires FastMCP v4 and SDK v2.
   - **Recommendation: Option A** — The server already has `OPENROUTER_API_KEY` configured in `.env` for the enrichment script. Using it for sampling replaces the deprecated feature cleanly and works on any FastMCP version.

### Medium-term (Q1–Q2 2027)

3. **Upgrade to FastMCP 4.0** and Python MCP SDK 2.x once stable:
   - Update `pyproject.toml`: `fastmcp>=4.0.0`
   - Expected changes:
     - No `ctx.sample()` — must already be migrated (see step 2)
     - Snake_case field names in any manual model construction
     - `mcp_types` package replaces some imports from `mcp.types`
     - `UserSession` pattern for any application state
   - Test compatibility with all existing tools

4. **Optionally adopt new capabilities**:
   - Add `ttl_ms`/`cache_scope` hints to FastMCP server config for list caching
   - Evaluate Tasks extension for long-running indexing operations
   - Add `server/discover` integration if needed for clients

### Long-term (Before 2027-07-28)

5. Verify all deprecated features are migrated before the removal window closes

---

## Key References

| Resource | URL |
|----------|-----|
| Official changelog | https://modelcontextprotocol.io/specification/2026-07-28/changelog |
| Deprecated features registry | https://modelcontextprotocol.io/specification/2026-07-28/deprecated |
| Official blog post | https://blog.modelcontextprotocol.io/posts/2026-07-28 |
| Spec draft | https://modelcontextprotocol.io/specification/2026-07-28 |
| FastMCP v4 what's new | https://gofastmcp.com/getting-started/whats-new |
| FastMCP v4 upgrade guide | https://gofastmcp.com/getting-started/upgrading/from-fastmcp-3 |
| Python SDK v2 releases | https://github.com/modelcontextprotocol/python-sdk/releases |
