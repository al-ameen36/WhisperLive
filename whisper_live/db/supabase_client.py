import os
import logging
from supabase import create_client

SUPABASE_URL = os.environ.get("SUPABASE_URL")
# Prefer the service role key for backend operations to bypass RLS, fallback to anon key
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    logging.warning("SUPABASE_URL or SUPABASE_KEY/SUPABASE_SERVICE_ROLE_KEY is missing. Database operations may fail.")
    supabase = None
else:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
