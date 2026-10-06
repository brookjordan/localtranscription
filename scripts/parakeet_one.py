#!/usr/bin/env python3
"""ASR worker for one candidate; runs only inside .venv-asr."""
from __future__ import annotations
import json
import sys
from parakeet_mlx import from_pretrained

path = sys.argv[1]
model_id = 'mlx-community/parakeet-tdt-0.6b-v3'
result = from_pretrained(model_id).transcribe(path)
segments = getattr(result, 'sentences', None) or []
print(json.dumps({'model': model_id, 'text': ' '.join(s.text.strip() for s in segments),
                  'segments': [{'start': float(s.start), 'end': float(s.end), 'text': s.text.strip()} for s in segments]}))
