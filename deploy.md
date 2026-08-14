# Deploy Guide — Streamlit Community Cloud

This guide deploys `app.py` from your GitHub repo to Streamlit Community
Cloud (free tier), so you get a live, clickable link for submission.

**Prerequisite:** complete [`check-in.md`](./check-in.md) first — your repo
must already be pushed to GitHub before you can deploy it.

> **About "one-click deployment" / auto-deploy on every commit:**
> Streamlit Community Cloud already does this for you, with zero extra
> setup. The steps below are a **one-time** action to connect your repo.
> After that, **every `git push` to `main` automatically triggers a
> rebuild and redeploy** of the live app — you don't need any custom CI/CD,
> GitHub webhook, or third-party service for this. Section 9 below adds an
> optional safety-net (tests that run before/alongside that auto-deploy)
> and a keep-alive job so the free-tier app doesn't fall asleep.

## 1. Sign in

1. Go to [share.streamlit.io](https://share.streamlit.io) (also reachable via [streamlit.io/cloud](https://streamlit.io/cloud))
2. Click **Sign in** → **Continue with GitHub**
3. Authorize Streamlit to access your repositories (you can restrict it to just this repo if prompted)

## 2. Create the app

1. Click **Create app** / **New app**
2. Choose **"Deploy a public app from GitHub"**
3. Fill in:
   - **Repository:** `<your-username>/credit-default-ml`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Pick an app URL (e.g. `your-name-credit-default.streamlit.app`)

## 3. IMPORTANT — Set the Python version before deploying

Click **Advanced settings** and set the **Python version to 3.11 or 3.12**.

> **Why this matters:** Streamlit Community Cloud does **not** read a
> `runtime.txt` file to pick the Python version (a common source of
> confusion — many deployments fail because people assume it does). If you
> skip this step, Cloud may default to a very new Python version for which
> some packages (`numpy`, `scikit-learn`, `xlrd`) don't yet have prebuilt
> wheels, causing the build to fail with errors like
> `ERROR: Could not find a version that satisfies the requirement...`.
> `requirements.txt` in this project already uses flexible `>=` version
> pins specifically so pip can resolve compatible versions once you've
> picked a sane Python version here.

## 4. Deploy

1. Click **Deploy**
2. Watch the build logs (this takes 1–3 minutes). It will:
   - Clone your repo
   - Create a virtual environment with the Python version you chose
   - Run `pip install -r requirements.txt`
   - Launch `streamlit run app.py`
3. When it finishes, your app opens automatically at your chosen URL

## 5. Test the live app

Once deployed, verify every required feature works on the **live** URL (not
just locally):

- [ ] App loads without errors ("Oh no." error screen = something's wrong — check logs)
- [ ] Sidebar → **Download sample test_data.csv** button works
- [ ] Upload that same `test_data.csv` back in
- [ ] Model dropdown switches between all 5 models
- [ ] Evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC) display
- [ ] Confusion matrix and classification report render
- [ ] ROC curve and feature importance charts render
- [ ] "Compare All 5 Models" table and bar chart render

## 6. Troubleshooting common failures

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'X'` | Missing package in `requirements.txt` | Add it, commit, push (auto-redeploys) |
| `ERROR: Could not find a version that satisfies the requirement numpy...` | Python version mismatch | Set Python 3.11/3.12 in **Advanced settings** → **Reboot app** |
| App stuck on "Please wait..." / never loads | Build failed silently | Open **Manage app** → check build logs for the real error |
| `FileNotFoundError` for a `.joblib` or `.csv` file | File wasn't pushed to GitHub (check `.gitignore` didn't accidentally exclude it) | `git status`, `git add`, `git push` |
| App works locally but not on Cloud | Local Python/package versions differ from Cloud's | Confirm the Python version selected matches what you tested locally (3.11) |
| Deployment very slow / times out | Free tier has limited CPU/RAM | Keep model files small (already optimized: Random Forest ~3.7MB) |

If you change the Python version or `requirements.txt` after a failed
deploy, use **Manage app → Reboot app** to force a clean rebuild rather than
waiting for auto-redeploy.

## 7. Update the app after code changes

Streamlit Cloud auto-redeploys on every push to `main`:

```powershell
git add <changed files>
git commit -m "description of the change"
git push
```

Watch the app dashboard — it rebuilds automatically within a minute or two.

## 8. Grab your link for submission

Copy the live app URL (e.g. `https://your-name-credit-default.streamlit.app`)
and:

1. Paste it into `README.md` replacing:
   ```
   > **`<PASTE YOUR DEPLOYED STREAMLIT APP LINK HERE>`**
   ```
2. Commit and push:
   ```powershell
   git add README.md
   git commit -m "docs: add live Streamlit app link"
   git push
   ```
3. Use this same link in your submission PDF (Section 2, item 2 — "Live Streamlit App Link").

## 9. Optional: CI safety net + keep-alive automation (Day 11)

Two GitHub Actions workflows are staged in `daily-commits/day-11/` (see
[`daily-commits/HOW_TO_USE.md`](./daily-commits/HOW_TO_USE.md) — run
`.\daily-commits\commit_day.ps1 -Day 11` then `git push` to add them for
real):

### `.github/workflows/ci.yml` — pre-deploy safety net

Runs automatically on every push/PR to `main`. It installs
`requirements.txt`, checks `app.py` has no syntax errors, and runs
`tests/test_models.py`, which loads all 5 saved models against
`test_data.csv` and asserts they still predict correctly (accuracy/AUC
above a sane floor). If a corrupted model file or a breaking dependency
change ever gets pushed, you get a red ❌ on GitHub **before** relying on
the live app, instead of finding out from a crashed Streamlit page.

View results anytime under your repo's **Actions** tab on GitHub.

### `.github/workflows/keep_alive.yml` — prevent the app from sleeping

Streamlit Community Cloud's free tier puts inactive apps to sleep (shows a
"this app has gone to sleep" page with a manual wake-up button) — a bad
look if an evaluator opens your submission link days later. This workflow
pings your live URL every 3 days automatically to keep it warm.

**One-time setup after you've deployed:**
1. GitHub repo → **Settings** → **Secrets and variables** → **Actions** → **Variables** tab
2. **New repository variable**
   - Name: `STREAMLIT_APP_URL`
   - Value: your live app URL (e.g. `https://your-name-credit-default.streamlit.app`)
3. Done — it now runs on schedule automatically. You can also trigger it manually anytime from the **Actions** tab → **Keep Streamlit App Awake** → **Run workflow**.

## Final submission checklist

- [ ] GitHub repo link works and is public
- [ ] Streamlit app link opens and loads without errors
- [ ] All required app features work on the **live** deployment (not just locally)
- [ ] README.md has both links filled in (no `<PASTE ...>` placeholders left)
- [ ] BITS Virtual Lab screenshot captured (see main `README.md`)
- [ ] Submission PDF assembled in order: GitHub link → Streamlit link → screenshot → README content
