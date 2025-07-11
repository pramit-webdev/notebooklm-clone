# supabase_client.py

import os
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()  # Load from .env

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
