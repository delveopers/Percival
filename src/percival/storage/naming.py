import hashlib, re

def safe_name(text, limit=60):
  slug = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_.")[:limit] or "clip"
  return slug if slug == text else f"{slug}-{hashlib.sha1(text.encode('utf-8')).hexdigest()[:6]}"