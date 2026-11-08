#!/usr/bin/env bash
set -e

mkdir -p "${DATA_DIR:-.}/media"

python manage.py migrate --noinput

exec gunicorn ImageIQ.wsgi:application \
  --bind "0.0.0.0:${PORT:-8000}" \
  --worker-class gthread \
  --workers 1 \
  --threads 4 \
  --timeout 300 \
  --access-logfile - \
  --error-logfile -