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
