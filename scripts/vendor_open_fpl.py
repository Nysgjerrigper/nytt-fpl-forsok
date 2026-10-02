"""Rebuild the offline Solio compatibility module from the preserved upstream bytes."""
from pathlib import Path
import ast
import difflib
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'fpl/milp/_vendor/solio'
raw = (DEST / 'upstream_solver.py.txt').read_text()
if hashlib.sha256(raw.encode()).hexdigest() != '23d567bccc24ba8693b8cef23d8b41c375c6bc6a586b47eff8247faf175b2416':
    raise RuntimeError('preserved upstream bytes do not match the reviewed pinned revision')
node = next(n for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef) and n.name == 'solve_multi_period_fpl')
original = '\n'.join(raw.splitlines()[node.lineno-1:node.end_lineno]) + '\n'
source = original

def change(before: str, after: str) -> None:
    global source
    if source.count(before) != 1:
        raise RuntimeError(f'upstream patch anchor changed: {before[:80]}')
    source = source.replace(before, after)

change('    m.addConstr(fts[next_gw] == initial_ft * (1 - use_wc[next_gw]) + ft_base * use_wc[next_gw])',
       '    m.addConstr(fts[next_gw] == initial_ft)')
change('    use_bb = bin_vars(gws, name="use_bb")',
       '    use_bb = bin_vars(gws, name="use_bb")\n    bb_bench = bin_vars(players, gws, name="bb_bench")')
change('== LINEUP_SIZE + (SQUAD_SIZE - LINEUP_SIZE) * use_bb[w]', '== LINEUP_SIZE')
change('== 1 - use_bb[w] for w in gws])', '== 1 for w in gws])')
change('== 1 - use_bb[w] for w in gws for o in [1, 2, 3]])', '== 1 for w in gws for o in [1, 2, 3]])')
change('<= squad_max_play[t] + use_bb[w]', '<= squad_max_play[t]')
change('    ## Chip constraints', '''    # Keep a legal XI and score boosted bench slots separately.
    for p in players:
        for w in gws:
            benched = sum_(bench[p, w, o] for o in order)
            m.addConstr(bb_bench[p, w] <= use_bb[w])
            m.addConstr(bb_bench[p, w] <= benched)
            m.addConstr(bb_bench[p, w] >= use_bb[w] + benched - 1)

    ## Chip constraints''')
change('(lineup[p, w] + captain[p, w] + vcap_weight * vicecap[p, w] + use_tc[p, w] + sum_',
       '(lineup[p, w] + bb_bench[p, w] + captain[p, w] + vcap_weight * vicecap[p, w] + use_tc[p, w] + sum_')
change('multiplier = 1 * (is_lineup == 1) + 1 * (is_captain == 1) + 1 * (is_tc == 1)',
       'multiplier = is_lineup + int(val(bb_bench[p, w]) > BINARY_THRESHOLD) + is_captain + is_tc')
start = source.index('    for chip, var in chip_vars.items():')
end = source.index('    m.addConstrs([squad_fh[p, w] <= use_fh[w]', start)
source = source[:start] + '''    # Inventory is per half; first-half chips expire rather than carrying over.
    boundary = options["chip_half_boundary"]
    for chip, var in chip_vars.items():
        forced = options.get(f"use_{chip}", [])
        for w in forced:
            m.addConstr(var[w] == 1)
        allowed = allowed_chip_gws.get(chip)
        if allowed is not None:
            m.addConstrs(var[w] == 0 for w in gws if w not in allowed)
        for half in (1, 2):
            half_weeks = [w for w in gws if (1 if w <= boundary else 2) == half]
            m.addConstr(sum_(var[w] for w in half_weeks) <= options["chip_inventory"][chip][half])
    if next_gw == 1:
        m.addConstr(use_fh[next_gw] == 0)
        m.addConstr(use_wc[next_gw] == 0)
    previous_fh = options.get("previous_free_hit_gw")
    if previous_fh == next_gw - 1:
        m.addConstr(use_fh[next_gw] == 0)
    m.addConstrs(use_fh[w] + use_fh[w - 1] <= 1 for w in gws if w > next_gw)
''' + source[end:]
# Temporary FH holdings only retain the original selling-price discount until first sale.
before = '''    m.addConstrs(
        [
            sum_(fh_sell_price[p] * squad[p, w - 1] for p in players) + in_the_bank[w - 1] >= sum_(fh_sell_price[p] * squad_fh[p, w] for p in players)
            for w in gws
        ]
    )'''
after = '''    fh_original = bin_vars(price_modified_players, gws, name="fh_original")
    for p in price_modified_players:
        for w in gws:
            sold_before = sum_(transfer_out_first[p, earlier] for earlier in gws if earlier < w)
            m.addConstr(fh_original[p, w] <= squad_fh[p, w])
            m.addConstr(fh_original[p, w] <= 1 - sold_before)
            m.addConstr(fh_original[p, w] >= squad_fh[p, w] - sold_before)
    for w in gws:
        discount_owned = sum_((buy_price[p] - sell_price[p]) *
                             (1 - sum_(transfer_out_first[p, earlier] for earlier in gws if earlier < w))
                             for p in price_modified_players)
        temporary_cost = sum_(buy_price[p] * squad_fh[p, w] for p in players) - sum_(
            (buy_price[p] - sell_price[p]) * fh_original[p, w] for p in price_modified_players)
        available = sum_(buy_price[p] * squad[p, w - 1] for p in players) - discount_owned + in_the_bank[w - 1]
        m.addConstr(temporary_cost <= available)
'''
change(before, after)
change('            m.run()\n            # read', '''            m.run()
            info = m.getInfo()
            if m.getModelStatus() != highspy.HighsModelStatus.kOptimal or not m.getSolution().value_valid:
                raise RuntimeError(f"Solio solve rejected: {m.modelStatusToString(m.getModelStatus())}")
            if not np.isfinite(info.mip_gap) or info.mip_gap > 1e-9:
                raise RuntimeError(f"Solio solve rejected: nonzero MIP gap {info.mip_gap}")
            # read''')
change('"score": val(objective_expr),', '"score": val(objective_expr),\n                "solver_status": "Optimal",\n                "mip_gap": m.getInfo().mip_gap,\n                "solver_seconds": m.getRunTime(),')
change('''                "ft": val(fts[w]),
                "pt": val(penalized_transfers[w]),
                "nt": val(num_transfers[w]),''', '''                "ft": round(val(fts[w])),
                "pt": round(val(penalized_transfers[w])),
                "nt": round(val(num_transfers[w])),''')
header = '''"""Offline, locally patched Open FPL Solver; provenance is in MANIFEST.json."""
import os
import shlex
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import cast
import highspy
import numpy as np
import pandas as pd
BINARY_THRESHOLD = 0.5
SQUAD_SIZE = 15
LINEUP_SIZE = 11
MAX_GAMEWEEK = 38
MAX_PLAYERS_PER_TEAM = 3
BIN = highspy.HighsVarType.kInteger
INT = highspy.HighsVarType.kInteger
CONT = highspy.HighsVarType.kContinuous
def get_random_id(length):
    return uuid.uuid4().hex[:length]
'''
result = header + source
ast.parse(result)
(DEST / 'solver.py').write_text(result)
(DEST / 'PATCH.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),source.splitlines(True),fromfile='upstream/solve_multi_period_fpl',tofile='local/solve_multi_period_fpl')))
(DEST / '__init__.py').write_text('"""Pinned Open FPL Solver compatibility source, not a network client."""\n')
(DEST / 'MANIFEST.json').write_text(json.dumps({'repository':'https://github.com/solioanalytics/open-fpl-solver','commit':'ec65f5e2b2be34441cb4b2a9efafea1ec0fe3b79','upstream_sha256':hashlib.sha256(raw.encode()).hexdigest(),'local_sha256':hashlib.sha256(result.encode()).hexdigest(),'license':'LICENSE; includes upstream commercial-use wording','patch':'PATCH.diff'},indent=2)+'\n')
