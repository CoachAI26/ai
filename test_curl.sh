#!/bin/bash
# Test /api/v1/transcribe with audio file and title
curl -X POST "http://localhost:8000/api/v1/transcribe" \
  -H "accept: application/json" \
  -F "file=@/Users/khaneapple/seon_ai/5825784783795408424.ogg" \
  -F "title=Describe your favorite hobby"
