# Push this project to GitHub

Your project is committed locally. Use one of the options below to put it on GitHub.

## Option A: GitHub website (no CLI)

1. Go to [github.com/new](https://github.com/new).
2. Create a new repository:
   - **Repository name:** e.g. `waimea-forecast` or `waimea-bay-waves`
   - **Visibility:** Public or Private
   - Do **not** check "Add a README" or "Add .gitignore" (you already have them).
3. After creating the repo, run these in your project folder (replace `YOUR_USERNAME` and `REPO_NAME` with your GitHub username and repo name):

   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git
   git push -u origin main
   ```

## Option B: GitHub CLI (`gh`)

If you have [GitHub CLI](https://cli.github.com/) installed and logged in:

```bash
cd c:\Users\19135\Documents\waimea-forecast-1
gh repo create waimea-forecast --private --source=. --remote=origin --push
```

Use `--public` instead of `--private` if you want a public repo. Change `waimea-forecast` to any repo name you prefer.

## After pushing

- Your code will be at: `https://github.com/YOUR_USERNAME/REPO_NAME`
- To push future changes: `git add .` → `git commit -m "message"` → `git push`
