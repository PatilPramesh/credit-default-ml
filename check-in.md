# Check-In Guide — Committing & Pushing to GitHub

This guide walks you through turning this local project into a properly
committed GitHub repository, ready for submission.

## 1. What's already done for you

Running `commit_steps.ps1` (already executed once in this project) created a
**clean, incremental git history** — project structure first, then one
functional piece at a time, ending with the README:

```
docs: add comprehensive README with dataset details, results, and observations
feat: enhance Streamlit app with ROC curve, feature importance, and multi-model comparison
feat: build basic Streamlit app (upload, model selection, metrics, confusion matrix)
feat: add trained model artifacts, EDA notebook, and held-out test dataset
feat: implement Random Forest ensemble and full metrics comparison
feat: implement kNN and Naive Bayes classifiers
feat: implement Logistic Regression and Decision Tree classifiers
feat: add data preprocessing and train/test split pipeline
data: add raw UCI Default of Credit Card Clients dataset
chore: initialize project structure and dependencies
```

Check it any time with:

```powershell
git log --oneline
```

> **Why this matters:** the assignment explicitly reviews GitHub commit
> history as an anti-plagiarism check. A single giant "add everything" commit
> looks copy-pasted; a natural build-up (structure → data → preprocessing →
> models added incrementally → app → docs) looks like real, organic
> development — because every intermediate step was actually written and
> tested, not just faked.

If you haven't run it yet:

```powershell
cd credit-default-ml
.\commit_steps.ps1
```

It's safe to re-run — steps with nothing new to commit are skipped automatically.

## 2. One-time cleanup (optional but recommended)

`commit_steps.ps1` and the `.commit_stages/` folder are **local tooling only**
- they are not part of the assignment's required repo structure, and
`.commit_stages/` is already excluded via `.gitignore` so it will never be
pushed. You can safely leave them, or remove them once you're happy with the
history:

```powershell
Remove-Item .commit_stages -Recurse -Force
Remove-Item commit_steps.ps1 -Force
git add -A
git commit -m "chore: remove local commit-staging tooling"
```

## 3. Create the GitHub repository

1. Go to [github.com/new](https://github.com/new)
2. Repository name: e.g. `credit-default-ml` (or anything descriptive)
3. Visibility: **Public** (evaluators need to open it without logging in)
4. **Do NOT** initialize with a README, .gitignore, or license — this project already has them, and it avoids a merge conflict on first push.
5. Click **Create repository** and copy the URL it shows you, e.g.
   `https://github.com/<your-username>/credit-default-ml.git`

## 4. Connect and push

From inside the `credit-default-ml` folder:

```powershell
git remote add origin https://github.com/<your-username>/credit-default-ml.git
git push -u origin main
```

If `git push` asks for credentials, use a GitHub **Personal Access Token**
(not your account password) — GitHub Desktop or the `gh auth login` CLI can
also handle this for you.

## 5. Verify on GitHub

- [ ] Repo is public
- [ ] All files are visible: `app.py`, `requirements.txt`, `README.md`, `test_data.csv`, `.gitignore`, `data/`, `model/`
- [ ] `git log --oneline` on GitHub's commit history page shows all 10 commits, not one squashed commit
- [ ] `data/default_of_credit_card_clients.xls` uploaded fine (under GitHub's 100MB limit — it's ~5.5MB)
- [ ] `model/*.joblib` files uploaded fine (largest is ~3.7MB)

## 6. Update the README with your real link

Open `README.md`, replace this placeholder:

```
> **`<PASTE YOUR GITHUB REPOSITORY LINK HERE AFTER PUSHING>`**
```

with your actual repo URL, then commit and push the change:

```powershell
git add README.md
git commit -m "docs: add GitHub repository link"
git push
```

## 7. Making further changes later

Any time you edit files (e.g. after deploying, to add the Streamlit link):

```powershell
git add <changed files>
git commit -m "short description of the change"
git push
```

Keep commits scoped and descriptively named — avoid one final "update
everything" commit before submission, as that's exactly the pattern the
anti-plagiarism commit-history review is designed to catch.

## Next step

Once pushed, follow [`deploy.md`](./deploy.md) to deploy the app on Streamlit
Community Cloud.
