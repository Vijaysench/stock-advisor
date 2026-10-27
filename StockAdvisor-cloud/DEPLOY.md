# Stock Advisor — phone/cloud deploy (free)

Hosted on Streamlit Community Cloud. Steps (one-time, ~5 min):

## 1. Put the code on GitHub
Create a new repo (e.g. `stock-advisor`) at https://github.com/new and add ALL these files
(keep the `.streamlit/config.toml` path):
app.py, data.py, engine.py, history.py, report.py, requirements.txt, .streamlit/config.toml

## 2. Deploy on Streamlit Cloud
1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **Create app → Deploy a public app from GitHub**.
3. Repository: your repo · Branch: `main` · Main file path: `app.py`.
4. Click **Deploy**. First build takes 2-4 min.
5. You get a link like `https://<name>.streamlit.app` — open it on your phone.

## 3. (Phone) Add to home screen
- Android Chrome: menu ⋮ → **Add to Home screen**.
- iPhone Safari: Share → **Add to Home Screen**.
Now it opens like a normal app.

Notes:
- No API keys needed (free Yahoo data). Cloud host can reach Yahoo, so the office-network block is gone.
- Day-to-day tracking saves on the server; on free hosting it can reset if the app sleeps for days.
