#!/bin/bash
# ============================================================
# Vitas AI — PostgreSQL Setup Script
# Run this AFTER installing PostgreSQL:
#   brew install postgresql@16
#   brew services start postgresql@16
# ============================================================

set -e

echo "🐘 Vitas AI — PostgreSQL Setup"
echo "================================"

# 1. Start PostgreSQL if not running
echo ""
echo "Step 1: Checking PostgreSQL status..."
if pg_isready -q 2>/dev/null; then
    echo "  ✅ PostgreSQL is running"
else
    echo "  ⏳ Starting PostgreSQL..."
    brew services start postgresql@16 2>/dev/null || brew services start postgresql 2>/dev/null
    sleep 2
    if pg_isready -q 2>/dev/null; then
        echo "  ✅ PostgreSQL started"
    else
        echo "  ❌ Could not start PostgreSQL. Run: brew services start postgresql@16"
        exit 1
    fi
fi

# 2. Create the database user and database
echo ""
echo "Step 2: Creating database user and database..."

# Create user (ignore error if already exists)
psql postgres -c "CREATE USER vitas_user WITH PASSWORD 'vitas_secure_2024';" 2>/dev/null || echo "  (user already exists)"

# Grant createdb (needed for Django tests)
psql postgres -c "ALTER USER vitas_user CREATEDB;" 2>/dev/null

# Create database (ignore error if already exists)
psql postgres -c "CREATE DATABASE vitas_db OWNER vitas_user;" 2>/dev/null || echo "  (database already exists)"

echo "  ✅ Database 'vitas_db' ready, owned by 'vitas_user'"

# 3. Install psycopg2 (PostgreSQL adapter for Python)
echo ""
echo "Step 3: Installing psycopg2..."
pip install psycopg2-binary 2>/dev/null || pip install psycopg2 2>/dev/null
echo "  ✅ psycopg2 installed"

# 4. Run Django migrations
echo ""
echo "Step 4: Running Django migrations..."
python manage.py migrate
echo "  ✅ All tables created in PostgreSQL"

# 5. Create a superuser (optional)
echo ""
echo "Step 5: Create a Django superuser (admin account)"
echo "  You can skip this by pressing Ctrl+C"
python manage.py createsuperuser || true

echo ""
echo "================================"
echo "🎉 PostgreSQL setup complete!"
echo ""
echo "Your database:"
echo "  Name:     vitas_db"
echo "  User:     vitas_user"
echo "  Password: vitas_secure_2024"
echo "  Host:     localhost:5432"
echo ""
echo "Run the server: python manage.py runserver"
echo "================================"
