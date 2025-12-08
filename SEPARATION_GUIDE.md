# How to Separate TradeLens into Two Repositories

This guide walks you through splitting the monorepo into separate frontend and backend repositories for independent deployment.

---

## Step 1: Create Two New GitHub Repositories

Create two repos on GitHub:
1. `tradelens-backend` (or `tradelens-api`)
2. `tradelens-frontend`

---

## Step 2: Set Up Backend Repository

### 2.1 Copy Backend Files

```bash
# Create a new directory for the backend repo
mkdir ~/tradelens-backend
cd ~/tradelens-backend

# Initialize git
git init

# Copy backend files (excluding venv)
cp -r "/Users/andrewli/Desktop/personal projects/TL/TL v. C2/tradelens/backend/"* .
rm -rf venv __pycache__

# Verify files
ls -la
# Should see: main.py, requirements.txt, Dockerfile, render.yaml, etc.
```

### 2.2 Create .env.local for Development

```bash
cat > .env << 'EOF'
FRONTEND_URL=http://localhost:3000
ANTHROPIC_API_KEY=your_key_here
EOF
```

### 2.3 Push to GitHub

```bash
git add .
git commit -m "Initial backend setup"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/tradelens-backend.git
git push -u origin main
```

---

## Step 3: Set Up Frontend Repository

### 3.1 Copy Frontend Files

```bash
# Create a new directory for the frontend repo
mkdir ~/tradelens-frontend
cd ~/tradelens-frontend

# Initialize git
git init

# Copy frontend files (excluding node_modules)
cp -r "/Users/andrewli/Desktop/personal projects/TL/TL v. C2/tradelens/frontend/"* .
rm -rf node_modules .next

# Verify files
ls -la
# Should see: package.json, src/, public/, etc.
```

### 3.2 Create .env.local for Development

```bash
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local
```

### 3.3 Push to GitHub

```bash
git add .
git commit -m "Initial frontend setup"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/tradelens-frontend.git
git push -u origin main
```

---

## Step 4: Deploy Backend to Render.com (Free Tier)

### 4.1 Connect to Render

1. Go to [render.com](https://render.com) and sign up/login
2. Click "New +" → "Web Service"
3. Connect your GitHub account
4. Select your `tradelens-backend` repository

### 4.2 Configure Build Settings

| Setting | Value |
|---------|-------|
| Name | `tradelens-api` |
| Runtime | Python 3 |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `uvicorn main:app --host 0.0.0.0 --port $PORT` |

### 4.3 Add Environment Variables

In the Render dashboard, add:

| Key | Value |
|-----|-------|
| `FRONTEND_URL` | (leave blank for now, add after frontend deploy) |
| `ANTHROPIC_API_KEY` | Your Anthropic API key |
| `HUGGINGFACE_TOKEN` | Your HuggingFace token (optional) |

### 4.4 Deploy

Click "Create Web Service" and wait for deployment.

**Note your backend URL** - it will look like: `https://tradelens-api.onrender.com`

---

## Step 5: Deploy Frontend to Vercel (Free Tier)

### 5.1 Connect to Vercel

1. Go to [vercel.com](https://vercel.com) and sign up/login
2. Click "Add New Project"
3. Import your `tradelens-frontend` repository

### 5.2 Configure Environment Variables

Before deploying, add:

| Key | Value |
|-----|-------|
| `NEXT_PUBLIC_API_URL` | `https://tradelens-api.onrender.com` (your Render URL) |

### 5.3 Deploy

Click "Deploy" and wait for the build.

**Note your frontend URL** - it will look like: `https://tradelens-frontend.vercel.app`

---

## Step 6: Update Backend CORS

Go back to Render.com dashboard and add/update:

| Key | Value |
|-----|-------|
| `FRONTEND_URL` | `https://tradelens-frontend.vercel.app` (your Vercel URL) |

This allows your frontend to make requests to the backend.

---

## Step 7: Test the Deployment

1. Visit your frontend URL
2. The app should load and connect to the backend
3. Test stock search, ML predictions, etc.

---

## Local Development Workflow

After separation, your local development looks like:

### Terminal 1: Backend

```bash
cd ~/tradelens-backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Create .env with FRONTEND_URL=http://localhost:3000
uvicorn main:app --reload --port 8000
```

### Terminal 2: Frontend

```bash
cd ~/tradelens-frontend
npm install

# Create .env.local with NEXT_PUBLIC_API_URL=http://localhost:8000
npm run dev
```

---

## Quick Reference: File Structure After Separation

### Backend Repo (`tradelens-backend`)
```
tradelens-backend/
├── main.py
├── models.py
├── database.py
├── advanced_features.py
├── llm_service.py
├── sentiment_pipeline.py
├── requirements.txt
├── Dockerfile
├── render.yaml
├── railway.json
├── .gitignore
├── .env (local only, not committed)
└── README.md
```

### Frontend Repo (`tradelens-frontend`)
```
tradelens-frontend/
├── src/
│   ├── app/
│   │   ├── globals.css
│   │   ├── layout.tsx
│   │   └── page.tsx
│   ├── components/
│   │   └── TradeLens.tsx
│   └── lib/
│       └── api.ts
├── public/
├── package.json
├── next.config.js
├── tailwind.config.js
├── tsconfig.json
├── vercel.json
├── .gitignore
├── .env.local (local only, not committed)
└── README.md
```

---

## Troubleshooting

### CORS Errors

If you see CORS errors in the browser console:

1. Check that `FRONTEND_URL` is set correctly in Render
2. Make sure the URL includes `https://` 
3. Don't include trailing slashes
4. Redeploy the backend after changing env vars

### Backend Not Responding

1. Check Render logs for errors
2. Visit `https://your-backend-url.onrender.com/health` - should return `{"status": "healthy"}`
3. Free tier Render services spin down after inactivity - first request may take 30-60 seconds

### API Key Issues

1. Verify API keys are set correctly in Render dashboard
2. Check logs for specific error messages

---

## Alternative Deployment Options

### Backend Alternatives

| Platform | Pros | Cons |
|----------|------|------|
| **Render** | Easy setup, free tier | Cold starts on free tier |
| **Railway** | Fast deploys, generous free tier | $5/month after free credits |
| **Fly.io** | Global edge, great performance | More complex setup |
| **DigitalOcean App Platform** | Reliable, good docs | No free tier |

### Frontend Alternatives

| Platform | Pros | Cons |
|----------|------|------|
| **Vercel** | Best for Next.js, fast | Limited on free tier |
| **Netlify** | Great DX, easy setup | Less optimal for Next.js |
| **Cloudflare Pages** | Very fast, generous free tier | Less Next.js features |

---

## Need Help?

- Backend API docs: `https://your-backend-url.onrender.com/docs`
- Render docs: https://render.com/docs
- Vercel docs: https://vercel.com/docs

