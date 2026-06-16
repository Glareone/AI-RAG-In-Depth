# MCP → GitHub Integration for Claude Desktop

> How to connect Claude Desktop to GitHub via MCP using Docker and a Personal Access Token.
> Enables Claude to read/write repos, create/update files, and commit directly from conversation.

---

## 1. Prerequisites

- Claude Desktop installed
- Docker Desktop installed and running
- GitHub account with a repo to connect to

---

## 2. Create a GitHub Personal Access Token (PAT)

Go to: `https://github.com/settings/personal-access-tokens/new`

```
Token name:        claude-mcp          (or any descriptive name)
Expiration:        90 days             (renew periodically)
Repository access: All repositories    (or select specific repos)

Permissions required:
  Contents:   Read and Write           ← create/update/delete files
  Metadata:   Read-only                ← required baseline
```

Copy the token immediately — GitHub shows it only once.

- For private repos: needs `repo` scope
- For public repos only: `public_repo` scope is sufficient
- Principle of least privilege: scope to specific repos if you know which ones

---

## 3. Configure Claude Desktop

Open the config file:

```
macOS:    ~/Library/Application Support/Claude/claude_desktop_config.json
Windows:  %APPDATA%\Claude\claude_desktop_config.json
```

Add the `github` entry alongside any existing MCP servers:

```json
{
  "mcpServers": {
    "obsidian": {
      "...your existing obsidian config..."
    },
    "github": {
      "command": "docker",
      "args": [
        "run", "-i", "--rm",
        "--name", "github-mcp-claude",
        "-e", "GITHUB_PERSONAL_ACCESS_TOKEN",
        "ghcr.io/github/github-mcp-server"
      ],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_your_token_here"
      }
    }
  }
}
```

Key flags:
- `--name github-mcp-claude` — names the container, identifiable in Docker Desktop
- `--rm` — ephemeral container, removed when Claude closes the connection
- `-e GITHUB_PERSONAL_ACCESS_TOKEN` — passes PAT as environment variable

---

## 4. First Run

Restart Claude Desktop after saving the config.

Docker pulls `ghcr.io/github/github-mcp-server` automatically on first run.
The container appears in Docker Desktop as `github-mcp-claude` while Claude is active.

Expected container logs:

```
time=... level=INFO msg="starting server" version=v1.3.0
time=... level=INFO msg="token scopes fetched for filtering" scopes="[repo]"
GitHub MCP Server running on stdio
time=... level=INFO msg="server session connected"
```

---

## 5. Verify It Works

Test with a read before doing any writes:

```
"list the contents of github.com/youruser/yourrepo"
"read the file README.md from github.com/youruser/yourrepo"
```

Reads working = writes will work.

---

## 6. What Claude Can Do Via GitHub MCP

```
read operations:
  get file contents
  list directory contents
  list branches, commits, releases, tags
  read issues and pull requests

write operations:
  create or update files (direct commit to any branch)
  delete files
  create branches
  create issues and pull requests
  add comments to issues and PRs
```

---

## 7. Known Limitations

### Payload size limit (~6-8KB per write)
The Docker stdio transport has a size limit per write operation.
Large files (>6-8KB content) cause the request to hang and time out
silently — no error message, just a 4-minute wait then failure.

**Workaround:** split large files into sequential chunk commits.
Each commit replaces the full file, so pass the full accumulated content
on each write (not just the new section). Read the SHA after each commit
and use it for the next write.

Reads are unaffected — large files read without issue.

### Branch must exist
Verify the default branch name before writing (`main` vs `master`).
Claude cannot create a branch implicitly on a write.

### No PR required
Claude commits directly to the target branch without a PR.
Intentional for note-publishing workflows.
Ask explicitly if you want PR-based workflow instead.

---

## 8. Typical Workflow — Publishing Notes to GitHub

```
1. Write and refine notes in Obsidian (via Obsidian MCP)

2. When ready to publish:
   "push [note name] to github.com/youruser/yourrepo,
    path: folder/filename.md, commit directly to main"

3. Claude reads from Obsidian → writes to GitHub in one conversation turn

4. For notes >6KB: Claude splits into sequential commits automatically
   (pass full file content each time, not just the new section)
```

Both MCPs (Obsidian + GitHub) can be active simultaneously.
Claude reads from one and writes to the other in the same turn.

---

## 9. Claude Code CLI (Alternative)

For Claude Code CLI, no Docker needed — use the remote HTTP endpoint:

```bash
claude mcp add-json github '{"type":"http","url":"https://api.githubcopilot.com/mcp","headers":{"Authorization":"Bearer YOUR_PAT_HERE"}}'
```

Verify registration:

```bash
claude mcp list
```

Both Claude Desktop (Docker) and Claude Code CLI (remote HTTP) can use
the same PAT — no separate token needed.

---

## 10. PAT Renewal

PATs expire. When the token expires:

1. Go to `https://github.com/settings/tokens`
2. Regenerate the token
3. Update `GITHUB_PERSONAL_ACCESS_TOKEN` in `claude_desktop_config.json`
4. Restart Claude Desktop

Set a calendar reminder before expiry to avoid interruption.

---

## 11. Official Reference

Installation guide (GitHub official):
`https://github.com/github/github-mcp-server/blob/main/docs/installation-guides/install-claude.md`

> Note: the npm package `@modelcontextprotocol/server-github` is **deprecated as of April 2025**.
> Use the Docker image `ghcr.io/github/github-mcp-server` instead.
