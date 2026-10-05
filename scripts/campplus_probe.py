#!/usr/bin/env python3
"""Read-only CAM++ probe for exact clone references and one candidate."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

HEAR = Path('/Users/brook.jordan/git/motional/brook.jordan/hear')
SPEC = importlib.util.spec_from_file_location('hear_recognise', HEAR / 'worker/recognise.py')
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MOD)


def main() -> None:
    candidate, target, *others = map(Path, sys.argv[1:])
    if not others:
        raise ValueError('at least one non-target reference is required')
    config = MOD.system_config({})
    vectors = {name: MOD.embedding(path, config) for name, path in [('candidate', candidate), ('target', target)]}
    target_score = MOD.cosine(vectors['candidate'], vectors['target'])
    other_scores = [MOD.cosine(vectors['candidate'], MOD.embedding(path, config)) for path in others]
    scores = {'target': target_score, 'other': max(other_scores)}
    scores['margin'] = scores['target'] - scores['other']
    print(json.dumps({'systemId': config['id'], 'scores': scores, 'nonTargetScores': other_scores, 'targetIsTopMatch': scores['target'] > scores['other']}, indent=2))

if __name__ == '__main__':
    main()
