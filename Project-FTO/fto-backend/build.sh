#!/bin/bash
set -e

echo "=== Installing backend dependencies ==="
pip install -r requirements.txt

echo "=== Installing frontend dependencies ==="
cd ../pmi-frontend
npm install

echo "=== Building frontend ==="
VITE_API_URL="" npm run build

echo "=== Copying frontend build to backend ==="
cp -r dist/ ../fto-backend/dist/

echo "=== Build complete ==="
ls -la ../fto-backend/dist/
