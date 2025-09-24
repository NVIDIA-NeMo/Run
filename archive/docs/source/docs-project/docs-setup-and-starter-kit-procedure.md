### Procedure: Set up a clean docs workspace, create a staging branch, and run the docs locally

#### Prerequisites

- **GitHub CLI** installed (`gh`)
- **VS Code** installed
- **Git** and **Python** available on your system
- Access to the `NVIDIA-NeMo/Run` GitHub repository

#### 1) Clone the repository with GitHub CLI

1. Open a terminal (PowerShell on Windows).
2. Run:

```powershell
gh repo clone NVIDIA-NeMo/Run
```

3. If prompted to authenticate, follow the steps in section 2. If the repo cloned to an unexpected path, move it where you want to keep it (e.g., `C:\Users\<you>\Documents\github\Run` or `C:\Users\<you>\github\Run`).

#### 2) Authenticate GitHub CLI (first-time only)

1. Run:

```powershell
gh auth login
```

2. When prompted:
   - Choose: GitHub.com
   - Protocol: HTTPS
   - Authenticate Git with your GitHub credentials: Yes
   - Login method: Log in with a web browser
3. Copy the one-time code shown in the terminal.
4. Open the provided device login URL, paste the code, and authorize GitHub CLI.
5. Confirm access (you may be asked for your GitHub password depending on your settings).

#### 3) Open the project in VS Code

1. In the terminal, change to the repo directory, then open VS Code:

```powershell
cd Run
code .
```

2. Ensure you opened the local folder (not a remote cloud workspace).

#### 4) Create a staging branch from `main`

- VS Code UI: Click the branch name in the lower-left status bar, create a new branch from `main`, name it: `jgerhold/docs-refactor-staging`.
- Or via terminal:

```powershell
git fetch origin
git switch main
git pull
git switch -c jgerhold/docs-refactor-staging
```

#### 5) Archive current docs and create a clean `docs` sandbox

1. Create an `archive` directory at the repository root.
2. Move the existing `docs` folder into `archive`.
3. Create a new, empty `docs` folder at the repository root.

Example (PowerShell):

```powershell
New-Item -ItemType Directory -Path archive -ErrorAction SilentlyContinue | Out-Null
git mv docs archive\
New-Item -ItemType Directory -Path docs | Out-Null
```

#### 6) Commit and publish the staging branch

1. In VS Code, open Source Control, enter a message like: "Archiving old docs".
2. Commit, then Publish Branch (VS Code will create the remote branch on GitHub).
3. Or via terminal:

```powershell
git add -A
git commit -m "archive: move old docs to archive and create clean docs sandbox"
git push -u origin jgerhold/docs-refactor-staging
```

#### 7) Bring in the Docs Starter Kit

1. Download the starter kit from your internal source (e.g., `https://gitlab-master.nvidia.com/llane/docs-example-project-setup`).
2. From the starter kit, copy the entire `docs` directory into the top-level of the `Run` repo. If drag-and-drop doesn’t work in VS Code, manually copy/paste the files into the new top-level `docs` folder you created.
3. Also copy the starter kit’s `Makefile` and `requirements.txt` into the repository root (top level).

#### 8) Set up the documentation environment

1. In a terminal at the repo root, run:

```powershell
make docs-env
```

2. When complete, activate the virtual environment (PowerShell):

```powershell
.venv-docs\Scripts\Activate.ps1
```

   Your prompt should show `(.venv-docs)`.

#### 9) Run the docs locally with live reload

```powershell
make docs-live
```

The terminal will display a local URL for the docs (live-reload server). Open it in your browser.

#### Notes

- If `make` is not recognized on Windows, ensure you are using the repo-provided `Makefile` and that your shell is PowerShell. Alternatively, run the equivalent Python/Command Prompt scripts if provided by the starter kit.
- If you moved the cloned repository after cloning, verify your terminal working directory before running commands.
