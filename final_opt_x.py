#!/usr/bin/env python3
"""
final_opt_x.py - experiment X: wide-open STYLE search for the portable discone + HF vertical (layout C),
NEC2 via pynec-accel. Same antenna model as v3, much wider design space, new bands, one 20-hour run.

WHAT IS NEW COMPARED TO V3
  * 20 parameters with much wider ranges. New: m = disc slots (in v3 always = n). Hard limits reject designs
    that can't be built or simulated (crowded hubs, tips near the mast, more than 8,000 segments, ...).
  * Bands: 6m, 10m + CB40 (pair rule: the better of the two counts 90 %), 2m (target 1.3, harder wall),
    70cm, ISM 868 + 915 (one band), 23cm, RX 108-2500 MHz. 2.4 GHz is no longer a TX band.
  * Two scores per design: the RF score (what the search optimises, lower is better) and BUILDABILITY
    (0-10, higher is better), which the search never looks at. It only shows up in reports, and it picks
    the compute split for styles (see below).
  * The search looks for STYLES (valleys of the score):
      explorer  L-SHADE with 360 designs. It does not shrink until 4 styles exist; then the time controller
                steers its shrinking (down to 64) so that everything is finished at the deadline.
      styles    every explorer generation, its top 30 % are clustered (HDBSCAN). A cluster whose best design
                is in the top 10 % is checked with a hill-valley test: 3 designs on the line to the nearest
                style's best; if one is worse than both ends there is a ridge and the cluster is a NEW style.
                A new style runs 8 CMA-ES steps, then pauses. The first style (the pilot) is developed fully
                at once, to measure what a full development costs.
      split     while a style runs, it gets 50 % of the batch (inside the v3 ranges) or 20 % (outside).
      final     the explorer (64) runs until 10 generations bring < 0.1 %; its clusters become styles too.
      develop   every style resumes and runs until 10 steps bring < 0.005 (no design limit).
  * HF pruning in SHADOW mode: nothing is skipped; the log records which explorer trials an exact and a
    near-exact rule WOULD have skipped, how much HF time that would have saved and how often it was wrong.
  * One deadline: the planner aims to be done at DEADLINE_H (20 h of run time); a hidden buffer of 1 h covers
    forecast errors, `verify` and `report`. Styles still running at the end are flagged (cut off).
  * Per-run logs for the learnings: batches.csv, controller.csv, shadow.csv, clusters.csv; `stats` sums up.
  * CPU only (the GPU option of v3 is gone).

COMMANDS (run them from the folder that holds this file and the results folder results_x)
  python3 final_opt_x.py run              start, or continue after stop / crash / reboot / spot interruption
  python3 final_opt_x.py pause | resume | stop | throttle 100
  python3 final_opt_x.py status           phase, styles, time plan, speed
  python3 final_opt_x.py extend --hours 26   move the deadline (the hidden buffer stays 1 h); a finished run
                                          continues its cut-off styles with `run`
  python3 final_opt_x.py report           style catalogue, Pareto front (RF vs buildability), top designs, plots
  python3 final_opt_x.py report --rf      best designs by RF score only
  python3 final_opt_x.py report --band CB40     re-rank for one HF band (CB40 or 10m) instead of the pair rule
  python3 final_opt_x.py verify           every style's best: fine segmentation, stress tests, 2.75/3.0 GHz,
                                          roof scenario (10 m mast, fixed whip)
  python3 final_opt_x.py stats            the run's learnings: phases, styles, time plan, clustering, pruning
  python3 final_opt_x.py check --set n=16 --set m=24     evaluate one design (default: the v3 winner)
  python3 final_opt_x.py deck --rank 1 --freq 145 | --band CB40      NEC file of a design
  python3 final_opt_x.py csv              rewrite evals_clean.csv from the log
  options: --dir results_x   --workers N   --quick (tiny smoke test in results_quick_x)

Scores and buildability are recomputed from the stored raw results by `report`, so weights and targets below
can be changed after the run. Only the simulated frequencies and the model are fixed.
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")       # one thread per simulation: we run many in parallel
import argparse, csv, json, math, pickle, signal, sys, time
import multiprocessing as mp
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
import numpy as np
try:                     # load NEC first: its library needs "static TLS" memory, which scikit-learn's OpenMP (loaded
    import PyNEC         # later for HDBSCAN) can use up, and then the worker processes could not load NEC at all
except Exception:        # (fails with "cannot allocate memory in static TLS block"). Without NEC: report/status still work.
    PyNEC = None

VERSION = 10      # model/score version of experiment X (a results folder holds one version only)
C0 = 299.792458   # speed of light, m*MHz

# =============================== EDIT HERE ===============================
# ---- fixed physical assumptions (as in v3)
FEED_HEIGHT    = 3.2     # disc hub (feed) above ground, m: tripod 2.94 m + 0.3 m mount
MOUNT_LEN      = 0.30    # fibreglass mount tube between tripod top and disc hub, m
TRIPOD         = True    # aluminium tripod under the mount tube (HF model, and `verify` up to TRIPOD_MAX_MHZ)
TRIPOD_COLLAR  = 0.90    # height where the legs meet the centre column, m
TRIPOD_FOOT_R  = 1.00    # radius of the circle the feet stand on, m
TRIPOD_COL_R, TRIPOD_LEG_R = 0.0125, 0.008
TRIPOD_SEG_MIN = 0.05    # the tripod tubes are thick: never cut them into segments shorter than this, m
TRIPOD_MAX_MHZ = 750.0   # above this the tripod (and the roof mast) is left out: too thick for NEC there
TUBE_EXTRA     = 0.05    # body tube ends this far above the cone tips, m
SPINE_R_VHF, SPINE_R_HF = 0.003, 0.005   # radius used for the ~30 mm body tube in the VHF / HF model (see v3)
HF_FEED_LEN    = 0.03    # HF model only: feed segment length (numerically clean feed region), m
WHIP_R, WHIP_MAX, WHIP_STEP = 0.005, 3.5, 0.10   # whip radius, longest whip, step of the full scan (m)
HINT = {"6m": 0.90, "10m": 2.30, "CB40": 2.50}   # typical whip lengths: the HF tuning starts here, m
SPOKE_R        = 0.0015  # spokes are 3 mm
FEED_R         = 0.0008
COAX_R, COAX_OFFSET, COAX_RUNS = 0.0025, 0.05, (5.0, 3.0)   # coax: radius, distance from tripod, ground runs
CHOKE_OHM      = 2000.0  # resistive choke 3 cm below the hub
COIL_C, COIL_Q, COIL_LEN = 1.0e-12, 150.0, 0.03   # spoke coils: nominal self-capacitance (F), Q, real length (m)
COIL_SEG, COIL_GRADE = 0.015, 1.25   # the coil is one 15 mm loaded segment; spoke segments grow 1.25x away from it
CSELF_SCORE    = (0.7e-12, 1.4e-12)  # every coil-sensitive simulation runs at both; the worse result counts
CSELF_MAX_MHZ  = 1000.0
CONDUCTIVITY   = 3.5e7   # aluminium, S/m
GROUND         = (13.0, 0.005)   # average soil: relative permittivity, conductivity S/m
SEG_VHF, SEG_HF = 12, 80          # segments per wavelength during the search
SEG_VHF_FINE, SEG_HF_FINE = 20, 120   # used by `verify` (and `check --fine`)
MEM_LIMIT_GB   = 0.0     # RAM the sims in flight may use; 0 = automatic (60% of what is free at the start)
ROOF_HEIGHT, MAST_R = 10.0, 0.02      # verify's roof scenario: feed height, radius of the metal mast (m)

# ---- bands. Penalty per band = weight * [(SWR - 1) + wall * max(0, SWR - target)^2]
HF_BANDS = {   # the whip is slid to minimise the worst SWR over these frequencies
    "6m":   dict(freqs=(50.0, 50.5, 51.0),          target=1.45, weight=0.7, whip_max=2.0),
    "10m":  dict(freqs=(28.2, 28.5, 28.8),          target=1.45, weight=None, whip_max=WHIP_MAX),   # pair rule
    "CB40": dict(freqs=(26.965, 27.185, 27.405),    target=1.45, weight=None, whip_max=WHIP_MAX),   # pair rule
}
PAIR = ("10m", "CB40")    # pair penalty = PAIR_WEIGHT * [PAIR_BETTER * better band + (1 - PAIR_BETTER) * worse band]
PAIR_WEIGHT, PAIR_BETTER = 1.6, 0.9
VHF_BANDS = {  # hz = (pattern frequency, horizon-gain floor in dBi, pattern weight); wall default K_WALL
    "2m":   dict(freqs=(144.0, 145.0, 146.0),                        target=1.3,  wall=30.0, weight=1.0,
                 hz=(145.0, 0.5, 1.0)),
    "70cm": dict(freqs=(430.0, 435.0, 440.0),                        target=1.45, weight=1.0, hz=(435.0, 0.5, 1.0)),
    "ISM":  dict(freqs=(863.0, 869.5, 876.0, 902.0, 915.0, 928.0),   target=1.45, weight=1.0, hz=(869.5, -1.0, 1.0)),
    "23cm": dict(freqs=(1240.0, 1270.0, 1300.0),                     target=1.45, weight=0.4, hz=(1270.0, -1.0, 1.0)),
}
ISM_REGIONS = {"868": (863.0, 869.5, 876.0), "915": (902.0, 915.0, 928.0)}   # shown separately in reports
RX_RANGE, RX_POINTS, RX_OK, RX_PEN = (108.0, 2500.0), 24, 3.0, 0.5   # receive: penalty per SWR unit above RX_OK
K_WALL        = 10.0
HZ_PEN        = 0.2     # per dB below the horizon floor (x pattern weight)
HZ_BONUS      = 0.05    # reward per dB above the floor (x pattern weight) ...
HZ_BONUS_CAP  = 2.0     # ... up to this many dB
HF_DX_FLOOR, HF_DX_PEN = -3.0, 0.2   # HF gain 3-20 deg below the floor costs this per dB
VERIFY_EXTRA_MHZ = (2750.0, 3000.0)  # verify: unscored SWR above the RX range ("3 GHz on the box?")

# ---- search space: name: (low, high, integer?)
PARAMS = {
    "n":         (8,    28,   True),    # cone wires
    "m":         (8,    32,   True),    # disc slots (each holds a plain rod, an A spoke or a B spoke)
    "half":      (20.0, 70.0, False),   # cone half-angle from the axis, deg
    "L":         (0.30, 1.50, False),   # cone wire length, m
    "dratio":    (0.25, 1.50, False),   # disc diameter / cone base diameter
    "tilt":      (0.0,  60.0, False),   # disc rods below horizontal, deg
    "dtop":      (10.0, 100.0, False),  # apex / hub diameter, mm
    "gap":       (3.5,  25.0, False),   # feed gap, mm
    "cone_wire": (0.1,  4.0,  False),   # mm
    "disc_rod":  (0.1,  4.0,  False),   # mm
    "kA": (3, 16, True), "LA": (0.20, 1.60, False), "dA": (0.0, 75.0, False),
    "cA": (0.1, 10.0, False), "pA": (0.0, 1.0, False),
    "kB": (3, 16, True), "LB": (0.20, 1.80, False), "dB": (0.0, 75.0, False),
    "cB": (0.1, 10.0, False), "pB": (0.0, 1.0, False),
}
# spoke set A (6m) / B (10m+CB): count k, length L (m, from the hub centre), droop d (deg below horizontal),
# coil c (uH), coil position p (0 = 0.10 m from the hub centre, 1 = 0.10 m before the tip)
V3_RANGES = {   # the v3 search space (m uses n's); buildability gives points for staying inside it
    "n": (18, 22), "m": (18, 22), "half": (34.0, 41.0), "L": (0.80, 0.90), "dratio": (0.40, 0.60),
    "tilt": (5.0, 25.0), "dtop": (20.0, 40.0), "gap": (4.0, 6.5), "cone_wire": (1.0, 4.0), "disc_rod": (2.0, 4.0),
    "kA": (3, 4), "LA": (0.30, 0.80), "dA": (10.0, 60.0), "cA": (0.8, 4.0), "pA": (0.0, 1.0),
    "kB": (3, 4), "LB": (0.70, 1.00), "dB": (10.0, 60.0), "cB": (1.2, 4.0), "pB": (0.0, 1.0),
}
REFERENCE = dict(n=20, m=20, half=40.846, L=0.830, dratio=0.580, tilt=6.911, dtop=22.031, gap=6.494,  # v3 winner
                 cone_wire=1.838, disc_rod=2.019, kA=3, LA=0.647, dA=53.42, cA=1.175, pA=0.185,
                 kB=4, LB=0.918, dB=10.039, cB=3.276, pB=0.255)

# ---- hard limits (designs outside them are rejected without simulating; they score REJECTED + how far out)
REJECTED, FAILED = 1e5, 1e6   # far above any simulated score (random designs can score in the thousands)
HUB_SPACING      = 1.3      # elements around a hub need 30 % spacing: count x diameter x 1.3 <= pi x dtop
SPOKE_DIA_MM     = 2000 * SPOKE_R
DISC_CLEAR       = 0.010    # the disc diameter exceeds the hub diameter by at least this, m
TIP_FROM_MAST    = 0.15     # every rod and spoke tip at least this far from the mast (horizontally), m
TIP_ABOVE_GROUND = 1.0      # ... and this high above the ground (on the tripod), m
MAX_SEGMENTS     = 8000     # segments of the model at the top RX frequency (time and memory per simulation)
COAX_CLEAR       = 0.03     # rods and spokes stay this far from the coax that runs down the mast, m

# ---- buildability (the "imaginary" score, 0-10; the search ignores it)
BUILD = dict(range_pts=4.0,       # inside the v3 ranges: 4 points, ...
             range_step=0.2,      # ... -1 per 20 % of a parameter's X range outside its v3 range
             sturdy_mm=3.0,       # cone wire 2 points, disc rods 2 points, full at >= 3 mm
             compact_pts=2.0, compact_full=1.0, compact_zero=1.6,   # longest part <= 1.0 m: 2 points, >= 1.6 m: 0
             coils_free=7, coil_pen=0.5,     # -0.5 per coil beyond 7
             all_spokes_pen=2.0,  # ... and -2 if there is no plain disc rod at all
             in_range_min=3.0)    # "inside the v3 ranges" (for the compute split) = at least 3 range points

# ---- the search
POP0             = 360       # explorer population: 18 x 20 parameters
NP_MIN           = 64        # the explorer shrinks to this at most
DETECT_FROM      = 8         # explorer generations before style detection starts
DETECT_TOP, DETECT_BEST = 0.30, 0.10   # cluster the top 30 %; a style's best must be in the top 10 %
HDB_MIN_CLUSTER, HDB_MIN_SAMPLES, HDB_METHOD = 5, 2, "leaf"   # HDBSCAN (tested on a landscape with known valleys)
HV_POINTS        = (0.25, 0.5, 0.75)   # hill-valley test: designs on the line candidate -> style
HV_TOL           = (0.02, 0.01)        # a test design is "worse" if above max(ends) + 0.02 + 0.01 x |score|
HV_NEAREST       = 3                   # a candidate is tested against the 3 nearest styles (nearest first)
MERGE_EVERY      = 10        # every 10 steps a running style is hill-valley tested against its nearest style
EXPLORER_TABOO   = True      # explorer clusters in the valley of an existing style are replaced by random designs,
                             # so the explorer keeps looking for NEW valleys (False: plain L-SHADE, which collapses
                             # into the best valley; tested on a landscape with 7 valleys: see the learnings doc)
DEAD_FIT         = 1e9       # score given to a culled explorer member (its next trial always replaces it)
STORED_MAX       = 12        # a style shows the taboo where it is with its best, its mean and <= 12 recent points
ESTABLISH_STEPS  = 8         # a new style runs this many CMA-ES steps, then pauses until the develop phase
MAX_ESTABLISHING = 3         # styles establishing at the same time (the pilot doesn't count)
PATIENCE_STEPS, PATIENCE_GAIN = 10, 0.005     # a style is finished when 10 steps bring < 0.005
FINAL_PATIENCE, FINAL_GAIN = 10, 0.001        # final population: done when 10 generations bring < 0.1 % ...
FINAL_MAX_GENS   = 60        # ... or after 60 generations (it can creep along with tiny gains for hours)
FINAL_MIN_POP    = 6         # ... or when fewer designs than this are left outside the styles' valleys
SHRINK_FROM, SHRINK_PER_STYLE, SHRINK_MAX = 4, 0.005, 0.05   # from 4 styles: 0.5 %/generation per style beyond 3
SHRINK_PANIC_MAX = 0.10      # the time controller may shrink up to 10 % per generation ...
SHRINK_PANIC_FROM = 0.03     # ... and before 4 styles exist only when it needs >= 3 % per generation
SHRINK_SLOW      = 0.0       # with time to spare it may shrink slower than the natural rate, down to this x it
                             # (0: exactly as slowly as the plan allows, so discovery uses all the time it has)
FULL_LOAD        = 0.9       # the time plan assumes the develop phase keeps 90 % of the cores busy
                             # (it adds no other safety margin: that is what the hidden buffer is for)
BATCH_PER_WORKER = 1.0       # designs per batch = workers x this (each design is ~70 simulations)
SPLIT_IN, SPLIT_OUT = 1.0, 4.0   # explorer designs per style design while a style runs: 50/50, 20/80
LAMBDA_MIN       = 8         # smallest CMA-ES step (designs)
DEV_STEPS_PRIOR  = 40        # steps a full development takes, until the pilot has measured it
STYLE_RATE_PRIOR = (0.5, 7.5)   # new styles per explorer generation: (recent finds + 0.5) / (recent generations + 7.5)
STYLE_RATE_MEMORY = 0.97     # "recent": each generation counts 3 % less (half-life ~23 generations)
DEADLINE_H       = 20.0      # the planner's deadline (hours of run time)
HIDDEN_BUFFER_H  = 1.0       # hard stop = deadline + buffer; the search itself stops VERIFY_RESERVE_H earlier
VERIFY_RESERVE_H = 0.5
CMA_SIGMA        = (0.02, 0.15)   # first CMA-ES step size of a style (in the 0..1 parameter cube)
SEED = 12345
# ========================================================================

QUICK = False                 # set by --quick: shrinks everything for a smoke test
HF_TUNE = dict(step=0.12, iters=6, step2=0.06, iters2=5)   # HF whip tuning: walk step (m), golden-section steps
NEC_RUNS = [0]                # NEC runs of the HF model in this process (for tests)


def apply_quick():
    global QUICK, RX_POINTS, WHIP_STEP, WHIP_MAX, POP0, NP_MIN, DETECT_FROM, ESTABLISH_STEPS, DEADLINE_H
    global PATIENCE_STEPS, FINAL_PATIENCE, DEV_STEPS_PRIOR, FINAL_MAX_GENS
    QUICK = True
    RX_POINTS = 3
    WHIP_STEP, WHIP_MAX = 0.2, 3.0
    HF_TUNE.update(iters=3, iters2=2)
    POP0, NP_MIN, DETECT_FROM, ESTABLISH_STEPS = 12, 8, 2, 2
    PATIENCE_STEPS, FINAL_PATIENCE, DEV_STEPS_PRIOR, FINAL_MAX_GENS = 2, 2, 4, 6
    DEADLINE_H = 0.25
    for sp in VHF_BANDS.values():
        sp["freqs"] = (sp["hz"][0],)


ACTIVE_SEG = None    # (VHF, HF) segments per wavelength in this process; None = SEG_VHF, SEG_HF.
                     # Worker processes of a fine Engine (verify) get the fine values via worker_init.

def segmentation():
    return ACTIVE_SEG or (SEG_VHF, SEG_HF)


# ------------------------------------------------------------------------ helpers
def fkey(f):
    return f"{f:.3f}"

def swr(z, z0=50.0):
    g = abs((z - z0) / (z + z0))
    return (1 + g) / (1 - g) if g < 0.999 else 999.0

def horizon_gain(el, G, lo=-10, hi=10):
    m = (el >= lo) & (el <= hi)
    return float(10 * np.log10(np.mean(10 ** (G[m] / 10))))

def rx_freqs():
    return [float(x) for x in np.round(np.geomspace(*RX_RANGE, RX_POINTS), 1)]

def norm(d):
    """Complete a design (missing values: middle of the range) and round the whole numbers."""
    d = dict(d)
    for k, (lo, hi, isint) in PARAMS.items():
        if k not in d:
            d[k] = (lo + hi) / 2
        d[k] = int(round(d[k])) if isint else float(d[k])
    return d

def cone_geom(d):
    a = d["dtop"] / 2000; th = math.radians(d["half"])
    return a, th, a + d["L"] * math.sin(th), d["L"] * math.cos(th)   # hub radius, angle, base radius, height

def disc_radius(d):
    return d["dratio"] * cone_geom(d)[2]

def coil_pos(L, frac):
    return 0.10 + frac * max(0.0, L - 0.20)

def spoke_sets(d):
    return [(d["kA"], d["LA"], d["dA"], d["cA"], coil_pos(d["LA"], d["pA"])),
            (d["kB"], d["LB"], d["dB"], d["cB"], coil_pos(d["LB"], d["pB"]))]

def spoke_slots(d):
    """{disc slot index: (length, droop, coil uH, coil position)} - the two sets interleaved over the m slots."""
    m, sets, out = d["m"], spoke_sets(d), {}
    for j, (k, L, dr, cL, cp) in enumerate(sets):
        for i in range(k):
            if len(out) >= m:
                return out                            # more spokes than slots (rejected by limits())
            idx = int(round((i + j / len(sets)) * m / k)) % m
            while idx in out:
                idx = (idx + 1) % m
            out[idx] = (L, dr, cL, cp)
    return out

def from_rim(d, s_, droop_deg):
    """Length of a rod / spoke as you would cut it: from the hub rim to the point s_ m along its line."""
    a = d["dtop"] / 2000; dr = math.radians(droop_deg)
    return math.hypot(s_ * math.cos(dr) - a, s_ * math.sin(dr))

def limits(d):
    """Hard limits 1-4 (geometry). Returns (worst ratio, what): a ratio > 1 means rejected."""
    d = norm(d)
    circ = math.pi * d["dtop"]
    plain = d["m"] - d["kA"] - d["kB"]
    out = [(d["n"] * d["cone_wire"] * HUB_SPACING / circ, "apex hub crowded"),
           ((max(0, plain) * d["disc_rod"] + (d["kA"] + d["kB"]) * SPOKE_DIA_MM) * HUB_SPACING / circ,
            "disc hub crowded"),
           ((d["dtop"] / 1000 + DISC_CLEAR) / (2 * disc_radius(d)), "disc smaller than the hub")]
    if plain < 0:
        out.append((1.0 + (-plain) / d["m"], "more spokes than disc slots"))
    tips = [(disc_radius(d), d["tilt"])] if plain > 0 else []
    tips += [(L, dr) for k, L, dr, cL, cp in spoke_sets(d)]
    for L, dr in tips:
        r = math.radians(dr)
        out.append((TIP_FROM_MAST / max(L * math.cos(r), 1e-6), "a tip too close to the mast"))
        out.append((TIP_ABOVE_GROUND / max(FEED_HEIGHT - L * math.sin(r), 1e-6), "a tip too close to the ground"))
    if plain >= 0:                                    # (with too many spokes there are no slots to check)
        out.append((COAX_CLEAR / max(coax_gap(d), 1e-6), "a rod or spoke touches the coax"))
    return max(out)

def seg_dist(p1, q1, p2, q2):
    """Shortest distance between two line segments (3D)."""
    p1, q1, p2, q2 = (np.asarray(v, float) for v in (p1, q1, p2, q2))
    u, v, w = q1 - p1, q2 - p2, p1 - p2
    a, b, c, dd, e = u @ u, u @ v, v @ v, u @ w, v @ w
    D = a * c - b * b
    sN, sD, tN, tD = 0.0, D, 0.0, D
    if D < 1e-12:
        sN, sD, tN, tD = 0.0, 1.0, e, c
    else:
        sN, tN = b * e - c * dd, a * e - b * dd
        if sN < 0:
            sN, tN, tD = 0.0, e, c
        elif sN > sD:
            sN, tN, tD = sD, e + b, c
    if tN < 0:
        tN = 0.0
        sN, sD = (0.0, sD) if -dd < 0 else ((sD, sD) if -dd > a else (-dd, a))
    elif tN > tD:
        tN = tD
        sN, sD = (0.0, sD) if b - dd < 0 else ((sD, sD) if b - dd > a else (b - dd, a))
    sc = 0.0 if abs(sN) < 1e-12 else sN / sD
    tc = 0.0 if abs(tN) < 1e-12 else tN / tD
    return float(np.linalg.norm(w + sc * u - tc * v))

def coax_gap(d):
    """Smallest distance between a disc rod / spoke and the coax down the mast (as modelled on the tripod)."""
    a, H = d["dtop"] / 2000, FEED_HEIGHT
    coax = [(0, 0, H - 0.06), (0, 0, H - 0.15), (COAX_OFFSET, 0, H - 0.20), (COAX_OFFSET, 0, 0.05)]
    slots, M, R, tl = spoke_slots(d), d["m"], disc_radius(d), math.radians(d["tilt"])
    best = 1e9
    for i in range(M):
        ph = 2 * math.pi * (i + 0.5) / M; cp, sp = math.cos(ph), math.sin(ph)
        L, dr = (slots[i][0], math.radians(slots[i][1])) if i in slots else (R, tl)
        rim = (a * cp, a * sp, H)
        tip = (L * math.cos(dr) * cp, L * math.cos(dr) * sp, H - L * math.sin(dr))
        for p, q in zip(coax[:-1], coax[1:]):
            best = min(best, seg_dist(rim, tip, p, q))
    return best

def buildability(d):
    """The imaginary score (0-10, higher = easier to build and carry) and its parts. The search ignores it."""
    d = norm(d)
    out = 0.0
    for k, (lo, hi) in V3_RANGES.items():
        A, B = PARAMS[k][0], PARAMS[k][1]
        out += max(0.0, lo - d[k], d[k] - hi) / (BUILD["range_step"] * (B - A))
    rng_pts = max(0.0, BUILD["range_pts"] - out)
    sturdy = 2 * min(d["cone_wire"] / BUILD["sturdy_mm"], 1.0) + 2 * min(d["disc_rod"] / BUILD["sturdy_mm"], 1.0)
    a, th, base_r, height = cone_geom(d)
    parts = [d["L"], height + TUBE_EXTRA, from_rim(d, d["LA"], d["dA"]), from_rim(d, d["LB"], d["dB"])]
    if d["m"] - d["kA"] - d["kB"] > 0:
        parts.append(from_rim(d, disc_radius(d), d["tilt"]))
    longest = max(parts)
    z, f = BUILD["compact_zero"], BUILD["compact_full"]
    compact = BUILD["compact_pts"] * min(1.0, max(0.0, (z - longest) / (z - f)))
    coils = -BUILD["coil_pen"] * max(0, d["kA"] + d["kB"] - BUILD["coils_free"])
    if d["m"] - d["kA"] - d["kB"] <= 0:
        coils -= BUILD["all_spokes_pen"]
    total = min(10.0, max(0.0, rng_pts + sturdy + compact + coils))
    return total, {"range": rng_pts, "sturdy": sturdy, "compact": compact, "coils": coils, "longest": longest}

def in_v3_ranges(d):
    return buildability(d)[1]["range"] >= BUILD["in_range_min"]


# ------------------------------------------------------------------------ NEC model
# The model is first built as plain data (a Model: a list of Wires plus loads), and only then handed to NEC
# (to_nec) or written to a .nec file (to_deck). Plain data can also be measured without running NEC: the
# number of segments tells how much memory and time a simulation will need (job_size, the 8,000 limit).

@dataclass
class Wire:
    """One NEC wire (a GW card): a straight conductor cut into nseg segments."""
    tag: int                 # wire number (NEC calls it the tag)
    nseg: int                # number of segments
    p: tuple                 # start point (x, y, z), m
    q: tuple                 # end point (x, y, z), m
    r: float                 # radius, m
    ratio: float = 1.0       # each segment `ratio` times longer than the previous one (1 = all equal)


@dataclass
class Model:
    """A complete NEC model as plain data."""
    ground: bool = False                              # over average ground (True) or in free space
    wires: list = field(default_factory=list)         # [Wire]
    loads: list = field(default_factory=list)         # LD cards: (type, tag, first seg, last seg, a, b, c)
    feed_tag: int = 0                                 # wire with the 1 V source (on its first segment)
    shield_tags: list = field(default_factory=list)   # wires of the coax shield's outside (common-mode)

    def wire(self, nseg, p, q, r, ratio=1.0):
        """Add a straight wire; returns its tag."""
        tag = len(self.wires) + 1
        self.wires.append(Wire(tag, max(1, int(nseg)), tuple(p), tuple(q), r, ratio))
        return tag

    def rod(self, p, q, r, smin, smax, qr=1.25):
        """Straight wire p->q whose segments start at smin near p and grow to smax (one or two NEC wires:
        a graded part, then a uniform part). Returns the tag of the last wire."""
        L = math.dist(p, q)
        if L < 1e-6:
            return 0
        u = [(q[i] - p[i]) / L for i in range(3)]
        k, s, Lt = 0, smin, 0.0
        while s < smax and Lt + s < L - 0.5 * smin:
            Lt += s; s *= qr; k += 1
        while k >= 1 and 0 < L - Lt < 4 * r:             # no leftover piece shorter than 4 radii (NEC would
            s /= qr; Lt -= s; k -= 1                      # find its centre inside the graded part)
        tag = 0
        if k >= 2:
            m_ = [p[i] + u[i] * Lt for i in range(3)]
            tag = self.wire(k, p, m_, r, qr)
            p = m_
        else:
            Lt = 0.0
        if L - Lt > 1e-6:
            tag = self.wire(math.ceil((L - Lt) / smax), p, q, r)
        return tag

    def rod2(self, p, q, r, s_p, s_q, smax, qr=1.25):
        """Straight wire p->q with fine segments at BOTH ends: they start at s_p next to p and at s_q next
        to q and grow by qr towards the middle (at most smax). Up to three NEC wires."""
        L = math.dist(p, q)
        if L < 1e-6:
            return 0

        def graded(s0):                      # the growing segments at one end: [lengths]
            seg, s_, Lt = [], s0, 0.0
            while s_ < smax and Lt + s_ <= L / 2:
                seg.append(s_); Lt += s_; s_ *= qr
            return seg
        gp, gq = graded(s_p), graded(s_q)
        while gp or gq:                      # no middle piece much shorter than its neighbours
            mid = L - sum(gp) - sum(gq)
            last = max(gp[-1] if gp else 0.0, gq[-1] if gq else 0.0)
            if mid < 1e-6 or mid >= 0.5 * last:
                break
            (gp if (gp[-1] if gp else 0.0) >= (gq[-1] if gq else 0.0) else gq).pop()
        kp, Lp, kq, Lq = len(gp), sum(gp), len(gq), sum(gq)
        u = [(q[i] - p[i]) / L for i in range(3)]
        p1 = [p[i] + u[i] * Lp for i in range(3)]
        q1 = [q[i] - u[i] * Lq for i in range(3)]
        tag = 0
        if kp:
            tag = self.wire(kp, p, p1, r, qr if kp > 1 else 1.0)
        if L - Lp - Lq > 1e-6:
            tag = self.wire(math.ceil((L - Lp - Lq) / smax), p1, q1, r)
        if kq:
            tag = self.wire(kq, q, q1, r, qr if kq > 1 else 1.0)    # drawn from q: its fine end is at q
        return tag

    def spokes_to_rim(self, centre, rim, r, smax, count):
        """Hub ring: a radial wire from the hub centre to where a rod starts (`count` of them around the hub).
        Large hubs (up to 100 mm) get several segments at UHF. Near the centre the radials crowd together, so a
        radial is made thinner where the centre of its first segment would lie inside its neighbour (NEC's rule),
        and it stays <= 1/4 of a segment thick (thin-wire rule). Normal hubs (like v3's) are unchanged."""
        L = math.dist(centre, rim)
        k = max(1, int(round(L / smax)))
        s = L / k
        clear = 0.8 * (s / 2) * math.sin(2 * math.pi / max(count, 3))   # first-segment centre to the next radial
        rr = r if r < clear / 0.8 * 0.98 else clear
        return self.wire(k, centre, rim, min(rr, 0.25 * s))

    def segments(self):
        return sum(w.nseg for w in self.wires)


def make_model(d, fseg, ground=False, whip=0.0, coax_run=None, tripod=False, hf=False,
               cself=COIL_C, f_load=None, seg=None, height=None, mast=False):
    """Layout C. Free space: hub at z=0. Over ground: hub at FEED_HEIGHT (or `height`).
    fseg sets the segmentation (use the highest frequency of the run); seg = (VHF, HF) segments per
    wavelength, default: this process's setting (see segmentation()). mast=True: a grounded metal mast
    under the mount tube instead of the tripod (verify's roof scenario)."""
    d = norm(d)
    m = Model(ground=ground)
    H = (height or FEED_HEIGHT) if ground else 0.0
    seg_vhf, seg_hf = seg or segmentation()
    smax = C0 / fseg / (seg_hf if hf else seg_vhf)
    rc, rd, gap = d["cone_wire"] / 2000, d["disc_rod"] / 2000, d["gap"] / 1000
    if hf:
        # HF: the mm-scale feed detail is irrelevant at 6-11 m wavelength but breaks NEC's
        # thin-wire rules, so use a 3 cm feed segment and uniform segments.
        gap = max(gap, HF_FEED_LEN); smin = smax
    else:
        smin = min(smax, max(gap, 4 * max(rc, rd, SPOKE_R)))
    nseg = lambda L: max(1, math.ceil(L / smax))
    a, th, base_r, height_c = cone_geom(d)            # a = radius of the apex and hub rings (dtop / 2)
    hub, apex = (0.0, 0.0, H), (0.0, 0.0, H + gap)
    # feed: the coax ends on the disc hub; its centre conductor bridges the gap to the apex. The source sits
    # on the first segment; a long gap (up to 25 mm) gets more segments at UHF.
    m.feed_tag = m.wire(max(1, int(round(gap / smax))), hub, apex, FEED_R)
    n = d["n"]
    for i in range(n):                                # cone, opening upward
        ph = 2 * math.pi * i / n; cp, sp = math.cos(ph), math.sin(ph)
        rim = (a * cp, a * sp, H + gap)
        tip = (base_r * cp, base_r * sp, H + gap + height_c)
        m.spokes_to_rim(apex, rim, rc, smax, n)       # apex ring
        m.rod(rim, tip, rc, smin, smax)
    top = H + gap + height_c + TUBE_EXTRA             # body tube + whip
    m.rod(apex, (0.0, 0.0, top), min(SPINE_R_HF if hf else SPINE_R_VHF, 0.5 * a), smin, smax)
    if whip > 0.01:
        m.wire(nseg(whip), (0.0, 0.0, top), (0.0, 0.0, top + whip), WHIP_R)
    tl, disc_R, slots = math.radians(d["tilt"]), disc_radius(d), spoke_slots(d)
    fl = f_load or fseg
    coils = []                                        # LD cards of the spoke coils (added at the end)
    M = d["m"]
    for i in range(M):                                # disc rods + spokes hang down from the hub ring
        ph = 2 * math.pi * (i + 0.5) / M; cp, sp = math.cos(ph), math.sin(ph)
        rim = (a * cp, a * sp, H)
        if i in slots:
            Ls, dr, cL, cpos = slots[i]; dr = math.radians(dr)
            u = (math.cos(dr) * cp, math.cos(dr) * sp, -math.sin(dr))
            P = lambda s_: (s_ * u[0], s_ * u[1], H + s_ * u[2])     # point s_ metres along the spoke line
            clen = min(COIL_SEG, smax)                                # the coil: one short loaded segment ...
            a_, b_ = cpos - clen / 2, cpos + clen / 2                 # ... with fine segments on both sides
            m.spokes_to_rim(hub, rim, SPOKE_R, smax, M)               # hub ring
            m.rod2(rim, P(a_), SPOKE_R, smin, clen, smax, COIL_GRADE)  # inner part: fine at the hub and the coil
            coil = m.wire(1, P(a_), P(b_), SPOKE_R)
            Lh = cL * 1e-6                                            # coil = L parallel C_self parallel R (Q)
            coils.append((1, coil, 1, 1, COIL_Q * 2 * math.pi * fl * 1e6 * Lh, Lh, cself))
            m.rod(P(b_), P(Ls), SPOKE_R, clen, smax, COIL_GRADE)       # outer part: fine at the coil
        else:
            tip = (disc_R * math.cos(tl) * cp, disc_R * math.cos(tl) * sp, H - disc_R * math.sin(tl))
            m.spokes_to_rim(hub, rim, rd, smax, M)                    # hub ring
            m.rod(rim, tip, rd, smin, smax)
    if coax_run is not None:                          # coax shield exterior, with the choke 3 cm below the hub
        pts = [hub, (0, 0, H - 0.03), (0, 0, H - 0.06), (0, 0, H - 0.15), (COAX_OFFSET, 0, H - 0.20),
               (COAX_OFFSET, 0, 0.05), (COAX_OFFSET + coax_run, 0, 0.05)]
        for j, (p, q) in enumerate(zip(pts[:-1], pts[1:])):
            tag = m.wire(1 if j < 2 else nseg(math.dist(p, q)), p, q, COAX_R)
            m.shield_tags.append(tag)
            if j == 1 and CHOKE_OHM:
                m.loads.append((0, tag, 1, 1, CHOKE_OHM, 0.0, 0.0))
    if ground and (tripod or mast):
        sl = max(smax, TRIPOD_SEG_MIN)
        tseg = lambda L: max(1, math.ceil(L / sl))
        col_top = H - MOUNT_LEN
        if mast:                                      # metal mast from the ground (grounded) to the mount tube
            m.wire(tseg(col_top - 0.02), (0, 0, 0.0), (0, 0, col_top), MAST_R)
        else:                                         # aluminium tripod, not connected
            m.wire(tseg(col_top - TRIPOD_COLLAR), (0, 0, TRIPOD_COLLAR), (0, 0, col_top), TRIPOD_COL_R)
            for ang in (60, 180, 300):
                ar = math.radians(ang)
                foot = (TRIPOD_FOOT_R * math.cos(ar), TRIPOD_FOOT_R * math.sin(ar), 0.02)
                m.wire(tseg(math.dist((0, 0, TRIPOD_COLLAR), foot)), (0, 0, TRIPOD_COLLAR), foot, TRIPOD_LEG_R)
    m.loads = [(5, 0, 0, 0, CONDUCTIVITY, 0.0, 0.0)] + m.loads + coils   # all wires aluminium; choke; coils
    return m


def model_segments(d):
    """Segments of the free-space model at the top RX frequency (hard limit 6)."""
    return make_model(d, RX_RANGE[1], seg=(SEG_VHF, SEG_HF)).segments()


def misfit(d):
    """0.0 if the design passes every hard limit; otherwise (ratio > 1, reason)."""
    w, why = limits(d)
    if w > 1.0:
        return float(w), why
    ns = model_segments(d)
    if ns > MAX_SEGMENTS:
        return ns / MAX_SEGMENTS, f"{ns} segments (limit {MAX_SEGMENTS})"
    return 0.0, ""


def to_nec(m):
    """Hand a Model to NEC. Returns the nec_context, ready for fr_card / xq_card / rp_card."""
    from PyNEC import nec_context
    c = nec_context(); g = c.get_geometry()
    for w in m.wires:
        g.wire(w.tag, w.nseg, *w.p, *w.q, w.r, w.ratio, 1.0)
    c.geometry_complete(1 if m.ground else 0)
    for ld in m.loads:
        c.ld_card(*ld)
    if m.ground:
        c.gn_card(2, 0, GROUND[0], GROUND[1], 0, 0, 0, 0)
    c.ex_card(0, m.feed_tag, 1, 0, 1.0, 0, 0, 0, 0, 0)
    return c


def build(d, fseg, **kw):
    """The model, handed to NEC. Returns (nec_context, feed tag, coax shield tags)."""
    m = make_model(d, fseg, **kw)
    return to_nec(m), m.feed_tag, m.shield_tags


def to_deck(m, freqs, title):
    """The same Model as a NEC-2 card deck (text), for 4nec2 / xnec2c / nec2c."""
    xyz = lambda w: " ".join(f"{v:.6f}" for v in (*w.p, *w.q))
    cards = [f"CM {title}", "CM written by final_opt_x.py", "CE"]
    for w in m.wires:
        if abs(w.ratio - 1.0) > 1e-12:            # graded segments: radius 0 here, then a GC card
            cards.append(f"GW {w.tag} {w.nseg} {xyz(w)} 0")
            cards.append(f"GC 0 0 {w.ratio:.6f} {w.r:.6f} {w.r:.6f}")
        else:
            cards.append(f"GW {w.tag} {w.nseg} {xyz(w)} {w.r:.6f}")
    cards.append(f"GE {1 if m.ground else 0}")
    for t, tag, s1, s2, a, b, c in m.loads:
        cards.append(f"LD {t} {tag} {s1} {s2} {a:.6g} {b:.6g} {c:.6g}")
    if m.ground:
        cards.append(f"GN 2 0 0 0 {GROUND[0]:g} {GROUND[1]:g}")
    cards.append(f"EX 0 {m.feed_tag} 1 0 1 0")
    df = (freqs[-1] - freqs[0]) / (len(freqs) - 1) if len(freqs) > 1 else 0.0
    cards.append(f"FR 0 {len(freqs)} 0 0 {freqs[0]:.4f} {df:.4f}")
    cards.append("RP 0 19 73 1000 0 0 5 5" if m.ground else "RP 0 37 73 1000 0 0 5 5")
    cards.append("EN")
    return "\n".join(cards) + "\n"


# ------------------------------------------------------------------------ jobs (run in worker processes)
def job_vhf(d, f, kind, whip=0.0, cself=COIL_C):
    c, ft, _ = build(d, f, whip=whip, cself=cself)
    c.fr_card(0, 1, f, 0)
    if kind == "pat":
        c.rp_card(0, 91, 1, 0, 5, 0, 0, 0, 0, 2, 0, 0, 0)
    else:
        c.xq_card(0)
    out = {"swr": swr(c.get_input_parameters(0).get_impedance()[0])}
    if kind == "pat":
        G = np.array(c.get_radiation_pattern(0).get_gain()).ravel()
        out["hz"] = horizon_gain(90 - np.arange(0, 182, 2), G)
    return out


def hf_z(d, band, whip, run, freqs, cself, pattern=False):
    """One NEC model of the HF setup (whip length `whip`, coax ground run `run`): the feed impedance at
    each of `freqs`. With pattern=True also the gain at 3-20 deg elevation and the shield current at the
    band centre. Returns ([Z], {"dx", "icm"} or {})."""
    fr = list(HF_BANDS[band]["freqs"]); fc = fr[len(fr) // 2]
    c, ft, sh = build(d, max(fr), ground=True, whip=whip, coax_run=run, tripod=TRIPOD, hf=True,
                      cself=cself, f_load=fc)
    ic = None
    for i, f in enumerate(freqs):                 # one fill + solve per frequency, results in this order
        c.fr_card(0, 1, f, 0)
        if pattern and ic is None and abs(f - fc) < 1e-9:
            c.rp_card(0, 90, 1, 0, 5, 0, 0, 0.0, 0, 1.0, 0, 0, 0); ic = i
        else:
            c.xq_card(0)
    NEC_RUNS[0] += len(freqs)
    zs = [complex(c.get_input_parameters(i).get_impedance()[0]) for i in range(len(freqs))]
    if ic is None:
        return zs, {}
    G = np.array(c.get_radiation_pattern(0).get_gain()).ravel(); el = 90 - np.arange(90)
    dx = float(10 * np.log10(np.mean(10 ** (G[(el >= 3) & (el <= 20)] / 10))))
    cur = c.get_structure_currents(ic)
    tags = np.array(cur.get_current_segment_tag()); I = np.abs(np.array(cur.get_current()))
    icm = float(I[np.isin(tags, sh)].max() / I[tags == ft][0])
    return zs, {"dx": dx, "icm": icm}


def hf_sim(d, band, whip, run, freqs, cself, pattern=False):
    """Like hf_z, but SWR instead of impedance."""
    zs, extra = hf_z(d, band, whip, run, freqs, cself, pattern)
    return [swr(z) for z in zs], extra


def tune_whip(d, band, cself, hint, step, iters):
    """Slide the whip to the lowest worst-case SWR across the band (coax run 1). Starts at `hint`:
    1) walk downhill in `step` jumps, judging by the SWR at the band centre, until it rises again;
    2) golden-section search inside that bracket on the worse of the two band edges.
    If there is no usable dip near the hint, scan the whole whip range first (like v1)."""
    sp = HF_BANDS[band]; fr = list(sp["freqs"]); fc = fr[len(fr) // 2]
    wmax = min(WHIP_MAX, sp.get("whip_max", WHIP_MAX)); wmin = 0.05
    run0 = COAX_RUNS[0]
    centre_memo = {}

    def centre(w):
        w = round(min(max(w, wmin), wmax), 4)
        if w not in centre_memo:
            centre_memo[w] = hf_sim(d, band, w, run0, [fc], cself)[0][0]
        return centre_memo[w]

    w0 = min(max(hint, wmin + step), wmax - step)
    a, b = w0 - step, w0 + step
    if centre(a) < centre(w0):                    # downhill to the left: keep walking left
        while a - step >= wmin and centre(a - step) < centre(a):
            a -= step
        lo, hi = max(a - step, wmin), a + step
    elif centre(b) < centre(w0):                  # downhill to the right
        while b + step <= wmax and centre(b + step) < centre(b):
            b += step
        lo, hi = b - step, min(b + step, wmax)
    else:                                         # the hint is already in the dip
        lo, hi = a, b
    if min(centre_memo.values()) > 3.0:           # no usable dip near the hint: scan everything (v1 way)
        grid = np.arange(WHIP_STEP, wmax + 1e-9, WHIP_STEP)
        sw = np.array([centre(w) for w in grid])
        best = sw.min()
        cand = [i for i in range(len(grid)) if sw[i] <= best + 0.15
                and (i == 0 or sw[i] <= sw[i - 1]) and (i == len(grid) - 1 or sw[i] <= sw[i + 1])]
        i0 = min(cand) if cand else int(np.argmin(sw))     # shortest good dip = quarter-wave solution
        lo, hi = grid[max(i0 - 1, 0)], grid[min(i0 + 1, len(grid) - 1)]
        iters += 3
    edges = [fr[0], fr[-1]]
    edge_memo = {}

    def worst(w):
        k = round(w, 4)
        if k not in edge_memo:
            edge_memo[k] = max(hf_sim(d, band, w, run0, edges, cself)[0])
        return edge_memo[k]

    g = 0.6180339887
    x1, x2 = hi - g * (hi - lo), lo + g * (hi - lo)
    for _ in range(iters):
        if worst(x1) < worst(x2):
            hi, x2 = x2, x1; x1 = hi - g * (hi - lo)
        else:
            lo, x1 = x1, x2; x2 = lo + g * (hi - lo)
    return float(min((x1, x2), key=worst))


def job_hf(d, band, cselfs=None):
    """For each coil self-capacitance: tune the whip (the second case starts from the first one's whip),
    then check all band frequencies with both coax runs. Returns {coil key: result}."""
    fr = list(HF_BANDS[band]["freqs"])
    hint, step, iters = HINT.get(band, 1.5), HF_TUNE["step"], HF_TUNE["iters"]
    out = {}
    for cs in (cselfs or CSELF_SCORE):
        w = tune_whip(d, band, cs, hint, step, iters)
        s1, extra = hf_sim(d, band, w, COAX_RUNS[0], fr, cs, pattern=True)
        s2, _ = hf_sim(d, band, w, COAX_RUNS[1], fr, cs)
        out[ckey(cs)] = dict({"whip": w, "swr_runs": [s1, s2], "swr": float(max(s1 + s2))}, **extra)
        hint, step, iters = w, HF_TUNE["step2"], HF_TUNE["iters2"]
    return out


def job_ground(d, f, whip, cself=COIL_C, height=None, mast=False):
    """Verification: layout C over ground with tripod (or roof mast), coax (2 m run), choke and whip."""
    low = f <= TRIPOD_MAX_MHZ
    c, ft, sh = build(d, f, ground=True, whip=whip, coax_run=2.0, tripod=TRIPOD and low and not mast,
                      mast=mast and low, cself=cself, height=height)
    c.fr_card(0, 1, f, 0)
    c.rp_card(0, 120, 1, 0, 5, 0, 0, 30.0, 0, 0.5, 0, 0, 0)
    z = c.get_input_parameters(0).get_impedance()[0]
    G = np.array(c.get_radiation_pattern(0).get_gain()).ravel(); el = 90 - (30.0 + 0.5 * np.arange(120))
    avg = lambda lo, hi: float(10 * np.log10(np.mean(10 ** (G[(el >= lo) & (el < hi)] / 10))))
    return {"swr": swr(z), "g_0_5": avg(0.5, 5), "g_5_15": avg(5, 15), "tripod": low}


def job_roof(d, whip, cself=COIL_C):
    """Verify's roof scenario: the hub ROOF_HEIGHT up on a grounded metal mast, coax down the mast, ONE fixed
    whip length for all HF bands. Returns {band: worst SWR over its frequencies}."""
    fs = sorted({f for sp in HF_BANDS.values() for f in sp["freqs"]})
    c, ft, sh = build(d, max(fs), ground=True, whip=whip, coax_run=COAX_RUNS[0], mast=True, hf=True,
                      cself=cself, f_load=27.0, height=ROOF_HEIGHT)
    for f in fs:
        c.fr_card(0, 1, f, 0); c.xq_card(0)
    s = {f: swr(complex(c.get_input_parameters(i).get_impedance()[0])) for i, f in enumerate(fs)}
    return {b: max(s[f] for f in sp["freqs"]) for b, sp in HF_BANDS.items()}


def job_hfpoint(d, band, f, whip, cself=COIL_C):
    """Plot helper: SWR at one frequency with a fixed whip length."""
    return hf_sim(d, band, whip, COAX_RUNS[0], [f], cself)[0][0]


def job_pattern(d, f, whip, cself, band=None):
    """Elevation cuts through phi = 0 and 180 deg. With an HF band: over ground on the tripod
    (theta 0-90 deg). Without: free space (theta 0-180 deg)."""
    if band:
        fr = HF_BANDS[band]["freqs"]
        c, ft, sh = build(d, max(fr), ground=True, whip=whip, coax_run=COAX_RUNS[0], tripod=TRIPOD, hf=True,
                          cself=cself, f_load=f)
        nth = 91
    else:
        c, ft, sh = build(d, f, whip=whip, cself=cself)
        nth = 181
    c.fr_card(0, 1, f, 0)
    c.rp_card(0, nth, 2, 0, 5, 0, 0, 0.0, 0.0, 1.0, 180.0, 0, 0)
    z = c.get_input_parameters(0).get_impedance()[0]
    g = np.array(c.get_radiation_pattern(0).get_gain()).reshape(nth, 2)
    return {"swr": swr(z), "phi0": g[:, 0].tolist(), "phi180": g[:, 1].tolist()}


JOBS = {"vhf": job_vhf, "hf": job_hf, "ground": job_ground, "hfpoint": job_hfpoint, "pattern": job_pattern,
        "roof": job_roof}

def run_job(job):
    """Runs in a worker process. Any NEC failure returns None (design scored as failed)."""
    try:
        return JOBS[job[0]](*job[1:])
    except Exception:
        return None

def run_job_timed(job):
    """run_job plus the seconds it took (the search logs the compute of every design)."""
    t0 = time.perf_counter()
    res = run_job(job)
    return {"res": res, "sec": time.perf_counter() - t0}

def worker_init(seg):
    global ACTIVE_SEG
    signal.signal(signal.SIGINT, signal.SIG_IGN)     # Ctrl+C goes to the main process only
    ACTIVE_SEG = tuple(seg)


# ------------------------------------------------------------------------ scoring
def vhf_points():
    """[(freq, 'z' or 'pat', coil self-capacitance)] needed to score one design."""
    pts = {}
    for b, sp in VHF_BANDS.items():
        for f in sp["freqs"]:
            pts.setdefault(fkey(f), (f, "z"))
        pts[fkey(sp["hz"][0])] = (sp["hz"][0], "pat")
    for f in rx_freqs():
        pts.setdefault(fkey(f), (f, "z"))
    out = []
    for f, kind in pts.values():
        for cs in (CSELF_SCORE if f <= CSELF_MAX_MHZ else (COIL_C,)):
            out.append((f, kind, cs))
    return out

def ckey(cs):
    return f"{cs * 1e12:.2f}pF"

def band_pen(s, sp, weight):
    return weight * ((s - 1) + sp.get("wall", K_WALL) * max(0.0, s - sp["target"]) ** 2)

def hf_own(raw, b):
    """One HF band's own penalty (weight 1 for the pair bands): SWR terms + the gain term."""
    sp, r = HF_BANDS[b], raw["hf"][b]
    w = sp["weight"] if sp["weight"] is not None else 1.0
    return band_pen(min(r["swr"], 20.0), sp, w) + HF_DX_PEN * max(0.0, HF_DX_FLOOR - r["dx"])

def score_raw(raw, band=None):
    """RF score (lower is better) and its parts, from stored raw results. band=None: the pair rule for
    10m / CB40; band="CB40" or "10m": only that band counts (with the pair's weight)."""
    if raw.get("misfit"):
        return REJECTED + raw["misfit"], {}
    if raw.get("fail"):
        return FAILED, {}
    parts = {}
    for b, sp in HF_BANDS.items():
        if b not in PAIR:
            parts[b] = hf_own(raw, b)
    p = {b: hf_own(raw, b) for b in PAIR}
    if band in PAIR:
        parts[band + " only"] = PAIR_WEIGHT * p[band]
    else:
        lo, hi = min(p.values()), max(p.values())
        parts["10m/CB40"] = PAIR_WEIGHT * (PAIR_BETTER * lo + (1 - PAIR_BETTER) * hi)
    for b, sp in VHF_BANDS.items():
        s = min(max(raw["vhf"][fkey(f)]["swr"] for f in sp["freqs"]), 20.0)
        q = band_pen(s, sp, sp["weight"])
        fhz, floor, pw = sp["hz"]
        hz = raw["vhf"][fkey(fhz)]["hz"]
        q += pw * (HZ_PEN * (floor - hz) if hz < floor else -HZ_BONUS * min(hz - floor, HZ_BONUS_CAP))
        parts[b] = q
    rx = sum(max(0.0, raw["vhf"][fkey(f)]["swr"] - RX_OK) for f in rx_freqs())
    parts["RX"] = RX_PEN * rx
    return float(sum(parts.values())), parts

HF_PART_KEYS = [b for b in HF_BANDS if b not in PAIR] + ["10m/CB40", "CB40 only", "10m only"]

def hf_part(parts):
    """The HF share of a score (>= 0): what the HF simulations add to the non-HF part."""
    return sum(v for k, v in parts.items() if k in HF_PART_KEYS)

def hf_type(raw, good=1.5):
    """CB / 10m / both: which of the pair bands a design serves (worst SWR <= 1.5)."""
    if not raw or "hf" not in raw:
        return "-"
    s = {b: raw["hf"][b]["swr"] for b in PAIR}
    ok = [b for b in PAIR if s[b] <= good]
    if len(ok) == 2:
        return "both"
    if ok:
        return "CB" if ok[0] == "CB40" else "10m"
    return "(CB)" if s["CB40"] <= s["10m"] else "(10m)"

def summary(raw):
    """Per band SWR (and horizon gain) for printing / CSV."""
    out = {}
    if raw.get("misfit") or raw.get("fail"):
        return out
    for b in HF_BANDS:
        out[b] = raw["hf"][b]["swr"]; out[b + "_whip"] = raw["hf"][b]["whip"]; out[b + "_dx"] = raw["hf"][b]["dx"]
    for b, sp in VHF_BANDS.items():
        out[b] = max(raw["vhf"][fkey(f)]["swr"] for f in sp["freqs"])
        out[b + "_hz"] = raw["vhf"][fkey(sp["hz"][0])]["hz"]
    for r, fs in ISM_REGIONS.items():
        if all(fkey(f) in raw["vhf"] for f in fs):
            out["ISM" + r] = max(raw["vhf"][fkey(f)]["swr"] for f in fs)
    out["RXworst"] = max(raw["vhf"][fkey(f)]["swr"] for f in rx_freqs())
    return out


# ------------------------------------------------------------------------ files / control
class Paths:
    def __init__(self, root):
        self.root = root
        os.makedirs(root, exist_ok=True)
        j = lambda n: os.path.join(root, n)
        self.control, self.progress, self.ckpt = j("control.json"), j("progress.json"), j("checkpoint.json")
        self.log, self.csv, self.lock, self.version = j("evals.jsonl"), j("evals.csv"), j("run.lock"), j("version.json")
        self.state = j("search_state.pkl")
        self.batches, self.controller = j("batches.csv"), j("controller.csv")
        self.shadow, self.clusters, self.events = j("shadow.csv"), j("clusters.csv"), j("events.log")

def read_json(path, default):
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return default

def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh)
    os.replace(tmp, path)

def append_csv(path, header, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(header)
        w.writerow(row)

def read_csv(path):
    try:
        with open(path, newline="") as fh:
            return list(csv.DictReader(fh))
    except OSError:
        return []

class StopRequested(Exception):
    pass

def design_key(d):
    d = norm(d)
    return "x|" + json.dumps({k: (d[k] if PARAMS[k][2] else round(float(d[k]), 5)) for k in PARAMS}, sort_keys=True)

def load_log(log_path):
    """All simulated designs of a results folder: {design key: record}."""
    cache = {}
    if os.path.exists(log_path):
        with open(log_path) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                    cache[r["key"]] = r
                except Exception:
                    pass
    return cache

CSV_BANDS = list(HF_BANDS) + list(VHF_BANDS) + ["ISM" + r for r in ISM_REGIONS]

def csv_header():
    return (["score", "build", "kind", "batch"] + list(PARAMS) + CSV_BANDS + [b + "_hz" for b in VHF_BANDS]
            + ["RXworst"] + [b + "_whip" for b in HF_BANDS] + [b + "_dx" for b in HF_BANDS]
            + ["segments", "sec_total", "sec_hf", "misfit"])

def csv_row(rec):
    sm = summary(rec["raw"])
    sec = rec.get("sec") or {}
    row = [f"{rec['score']:.4f}", f"{buildability(rec['params'])[0]:.2f}", rec.get("kind", ""), rec.get("batch", "")]
    row += [rec["params"][k] for k in PARAMS]
    row += [f"{sm[b]:.3f}" if b in sm else "" for b in CSV_BANDS]
    row += [f"{sm[b + '_hz']:.2f}" if b + "_hz" in sm else "" for b in VHF_BANDS]
    row += [f"{sm['RXworst']:.2f}" if "RXworst" in sm else ""]
    row += [f"{sm[b + '_whip']:.2f}" if b + "_whip" in sm else "" for b in HF_BANDS]
    row += [f"{sm[b + '_dx']:.2f}" if b + "_dx" in sm else "" for b in HF_BANDS]
    row += [rec.get("nseg", ""), f"{sec['total']:.1f}" if "total" in sec else "",
            f"{sec['hf']:.1f}" if "hf" in sec else "", rec["raw"].get("why", "")]
    return row

def append_log(paths, rec):
    with open(paths.log, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    append_csv(paths.csv, csv_header(), csv_row(rec))

def cmd_csv(a, paths):
    """Write evals_clean.csv from the log (evals.jsonl), e.g. when evals.csv got mixed up. Safe while running."""
    recs = sorted(load_log(paths.log).values(), key=lambda r: r.get("t", 0))
    if not recs:
        print("No results in", paths.root); return
    out = os.path.join(paths.root, "evals_clean.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(csv_header())
        for r in recs:
            w.writerow(csv_row(r))
    print(f"wrote {out}: {len(recs)} designs")

def solver_info():
    """Which NEC package provides the `PyNEC` module (pynec-accel and stock PyNEC use the same name)."""
    import importlib.metadata as md
    found = []
    for dist in ("pynec-accel", "PyNEC"):
        try:
            found.append(f"{dist} {md.version(dist)}")
        except md.PackageNotFoundError:
            pass
    return found

def solver_check():
    found = solver_info()
    if not found:
        print("WARNING: no NEC package found. Install it with:  pip install pynec-accel", flush=True)
    elif len(found) > 1:
        print(f"WARNING: both {found[0]} and {found[1]} are installed; they overwrite each other's files.\n"
              "         Fix:  pip uninstall -y PyNEC pynec-accel && pip install pynec-accel", flush=True)
    elif found[0].startswith("PyNEC"):
        print(f"NOTE: using stock {found[0]}. pynec-accel is faster and matches the reference NEC-2 over ground:\n"
              "      pip uninstall -y PyNEC && pip install pynec-accel", flush=True)
    return ", ".join(found) or "none"


# ------------------------------------------------------------------------ size of a job (memory, time)
@dataclass
class Size:
    mem_mb: float        # estimated peak memory of the simulation process while it runs this job
    cost: float          # rough relative time, only used to start the slowest jobs first
    n: int = 0           # segments of the (largest) model in the job

def job_size(job, seg):
    """Estimate memory and time of a job from its number of segments N, without running NEC.
    NEC's table (the interaction matrix) has N x N complex numbers of 16 bytes; solving it takes ~N^3."""
    kind = job[0]
    if kind == "vhf":
        n, solves = make_model(job[1], job[2], seg=seg).segments(), 1
    elif kind == "hf":
        fr = HF_BANDS[job[2]]["freqs"]
        n = make_model(job[1], max(fr), ground=True, whip=WHIP_MAX, coax_run=max(COAX_RUNS), tripod=TRIPOD,
                       hf=True, seg=seg).segments()
        solves = 30 * len(job[3] or CSELF_SCORE)
    elif kind == "ground":
        n = make_model(job[1], job[2], ground=True, whip=job[3], coax_run=2.0,
                       tripod=TRIPOD and job[2] <= TRIPOD_MAX_MHZ, seg=seg).segments()
        solves = 1
    elif kind == "roof":
        n, solves = make_model(job[1], 51.0, ground=True, whip=job[2], coax_run=COAX_RUNS[0], mast=True, hf=True,
                               height=ROOF_HEIGHT, seg=seg).segments(), 9
    elif kind == "hfpoint":
        n, solves = make_model(job[1], max(HF_BANDS[job[2]]["freqs"]), ground=True, whip=job[4],
                               coax_run=COAX_RUNS[0], tripod=TRIPOD, hf=True, seg=seg).segments(), 1
    else:  # pattern
        band = job[5]
        if band:
            n = make_model(job[1], max(HF_BANDS[band]["freqs"]), ground=True, whip=job[3],
                           coax_run=COAX_RUNS[0], tripod=TRIPOD, hf=True, seg=seg).segments()
        else:
            n = make_model(job[1], job[2], whip=job[3], seg=seg).segments()
        solves = 1
    g = kind in ("ground", "hf", "hfpoint", "roof") or (kind == "pattern" and bool(job[5]))   # over ground?
    # measured (v2): peak = ~35 MB (Python + NEC) + 1.1 x the table; time per solve: fill ~N^2 (over ground
    # ~3x slower, plus the Sommerfeld tables), solve ~N^3 (seconds on one core of a fast PC)
    mem = 50.0 + 1.15 * 16 * n * n / 2**20
    cost = solves * ((0.15 if g else 0.02) + (5.5e-7 if g else 2e-7) * n * n + 6e-11 * n ** 3)
    return Size(mem_mb=mem, cost=cost, n=n)

def mem_budget_mb():
    """MEM_LIMIT_GB, or automatically 60% of the memory that is free when the run starts."""
    if MEM_LIMIT_GB > 0:
        return MEM_LIMIT_GB * 1024.0
    try:
        with open("/proc/meminfo") as fh:
            info = {ln.split(":")[0]: float(ln.split()[1]) for ln in fh if ":" in ln}
        return max(1024.0, 0.6 * info["MemAvailable"] / 1024.0)
    except Exception:
        return 4096.0


# ------------------------------------------------------------------------ parallel evaluation engine
class Engine:
    """Runs simulation jobs on a pool of worker processes.
      * pause / throttle / stop from the control file,
      * memory budget: a job only starts if the estimated memory of all jobs in flight stays inside it
        (smaller jobs further back in the queue may start first, so the cores stay busy),
      * if a worker process dies, a fresh pool is started and the jobs that were in flight run again
        (a job that was in flight during 3 such deaths counts as failed)."""

    def __init__(self, workers, paths=None, fine=False):
        self.workers = max(1, workers)
        self.paths = paths
        self.seg = (SEG_VHF_FINE, SEG_HF_FINE) if fine else (SEG_VHF, SEG_HF)
        self.mem_budget = mem_budget_mb()
        self.restarts = 0
        self.active_s = 0.0
        self.busy_s = 0.0                     # summed seconds of all timed jobs (core-seconds used)
        self.progress = {}
        self._start_pool()

    def _start_pool(self):
        self.pool = ProcessPoolExecutor(max_workers=self.workers, mp_context=mp.get_context("fork"),
                                        initializer=worker_init, initargs=(self.seg,))

    def _kill_pool(self):
        pool = self.pool
        procs = list((getattr(pool, "_processes", None) or {}).values())
        try:
            pool.shutdown(wait=False, cancel_futures=True)
        except TypeError:                                  # Python < 3.9
            pool.shutdown(wait=False)
        for p in procs:
            try:
                p.kill()
            except Exception:
                pass
        for p in procs:
            try:
                p.join(10)
            except Exception:
                pass

    def close(self):
        self.pool.shutdown(wait=True)

    def terminate(self):
        self._kill_pool()

    def control(self):
        if not self.paths:
            return {"state": "run", "max_inflight": 0}
        return read_json(self.paths.control, {"state": "run", "max_inflight": 0})

    def write_progress(self, **kw):
        self.progress.update(kw)
        if self.paths:
            self.progress["updated"] = time.time()
            write_json(self.paths.progress, self.progress)

    @staticmethod
    def _take(queue, free_mb, idle):
        """First job in the queue that fits into free_mb (or the first one if nothing is running at all)."""
        for i, item in enumerate(queue):
            if idle or item[2].mem_mb <= free_mb:
                del queue[i]
                return item
        return None

    def run_jobs(self, jobs, on_result, order=None, fn=run_job):
        """jobs: list of (tag, job). Calls on_result(tag, job, result) as results arrive.
        Default order: slowest first (over the whole list, so the long jobs never wait until the end)."""
        items = [(tag, job, job_size(job, self.seg)) for tag, job in jobs]
        queue = deque(sorted(items, key=order or (lambda it: -it[2].cost)))
        inflight = {}                           # future -> item
        lost = []                               # jobs whose worker pool died
        deaths = {}                             # per job tag: how often its pool died under it
        last_ctl, ctl, shown = 0.0, {"state": "run"}, "run"
        t_prev = time.time()
        while queue or inflight:
            now = time.time()
            used = sum(it[2].mem_mb for it in inflight.values())
            if now - last_ctl > 1.0:
                ctl = self.control(); last_ctl = now
                state = ctl.get("state", "run")
                if state != shown:
                    msg = {"pause": "PAUSED - finishing the sims in flight, then idle",
                           "stop": "STOP requested - finishing the sims in flight, then saving",
                           "run": "RESUMED"}.get(state, state)
                    print(f"  [{time.strftime('%H:%M:%S')}] {msg}", flush=True); shown = state
                self.write_progress(state=state, inflight=len(inflight), queued=len(queue),
                                    mem_inflight_gb=used / 1024, mem_budget_gb=self.mem_budget / 1024,
                                    restarts=self.restarts)
            state = ctl.get("state", "run")
            if state == "run":
                self.active_s += min(now - t_prev, 5.0)     # capped: a sleeping laptop doesn't count
                limit = ctl.get("max_inflight", 0) or self.workers
                limit = max(1, min(limit, self.workers))
                while queue and len(inflight) < limit:
                    item = self._take(queue, self.mem_budget - used, not inflight)
                    if item is None:
                        break                             # memory budget full
                    inflight[self.pool.submit(fn, item[1])] = item
                    used += item[2].mem_mb
            t_prev = now
            if state == "stop" and not inflight:
                raise StopRequested()
            done = [fu for fu in inflight if fu.done()]
            for fu in done:
                item = inflight.pop(fu)
                try:
                    res = fu.result()
                except BrokenProcessPool:
                    lost.append(item); continue
                except Exception:
                    res = None
                on_result(item[0], item[1], res)
            if lost and not inflight:
                self._recover(lost, queue, deaths, on_result)   # all its futures are back
                lost = []
            if not done:
                time.sleep(0.25 if state == "run" else 1.0)

    def _recover(self, lost, queue, deaths, on_result):
        """A worker process died (killed for memory, or NEC crashed): new pool, re-run what was lost."""
        self.restarts += 1
        mem_lost = sum(it[2].mem_mb for it in lost)
        self._kill_pool()
        msg = f"a simulation process died; restarting the workers and re-running {len(lost)} sims"
        if mem_lost > 0.5 * self.mem_budget:                # memory may have been the reason: be careful
            self.mem_budget = max(1024.0, 0.75 * mem_lost)
            msg += f"; memory budget lowered to {self.mem_budget / 1024:.1f} GB"
        print(f"  [{time.strftime('%H:%M:%S')}] {msg}", flush=True)
        for item in reversed(lost):
            deaths[repr(item[0])] = deaths.get(repr(item[0]), 0) + 1
            if deaths[repr(item[0])] >= 3:
                on_result(item[0], item[1], None)           # keeps killing its worker: count as failed
            else:
                queue.appendleft(item)
        self._start_pool()

    def evaluate(self, designs, batch, cache, kinds=None, log=True):
        """Simulate and score designs. Returns [record] in the same order (record = what evals.jsonl stores:
        params, raw results, score, kind, batch, seconds, segments). Uses / fills the cache."""
        recs = [None] * len(designs)
        partial, jobs = {}, []
        pts = vhf_points()
        for i, d in enumerate(designs):
            d = norm(d); key = design_key(d)
            kind = kinds[i] if kinds else ""
            if key in cache:
                recs[i] = cache[key]; continue
            mf, why = misfit(d)
            if mf:
                self._finish(i, d, {"misfit": mf, "why": why}, batch, kind, cache, recs, log, {}, 0); continue
            partial[i] = {"d": d, "vhf": {}, "hf": {}, "need": len(pts) + len(HF_BANDS), "fail": False,
                          "kind": kind, "sec": {"vhf": 0.0, "hf": 0.0, "max": 0.0, "by_band": {}},
                          "nseg": model_segments(d)}
            jobs += [((i, "vhf", fkey(f), ckey(cs)), ("vhf", d, f, k, 0.0, cs)) for f, k, cs in pts]
            jobs += [((i, "hf", b, None), ("hf", d, b, None)) for b in HF_BANDS]     # all coil cases in one job
        self._done, total = sum(r is not None for r in recs), len(designs)
        self.write_progress(done_in_batch=self._done, total_in_batch=total)

        def on_result(tag, job, out):
            i, kind, k, ck = tag; p = partial[i]
            res = out["res"] if isinstance(out, dict) else None
            sec = out["sec"] if isinstance(out, dict) else 0.0
            self.busy_s += sec
            p["sec"]["max"] = max(p["sec"]["max"], sec)
            if res is None:
                p["fail"] = True
            elif kind == "hf":
                p["hf"][k] = res                     # {coil key: result}
                p["sec"]["hf"] += sec; p["sec"]["by_band"][k] = round(sec, 2)
            else:
                p["vhf"].setdefault(k, {})[ck] = res
                p["sec"]["vhf"] += sec
            p["need"] -= 1
            if p["need"] == 0:
                raw = {"fail": True} if p["fail"] else worst_case(p)
                s = p["sec"]; s["total"] = s["vhf"] + s["hf"]
                self._finish(i, p["d"], raw, batch, p["kind"], cache, recs, log,
                             {k_: (round(v, 2) if isinstance(v, float) else v) for k_, v in s.items()}, p["nseg"])
                self._done += 1
                self.write_progress(done_in_batch=self._done, total_in_batch=total)

        self.run_jobs(jobs, on_result, fn=run_job_timed)
        return recs

    def _finish(self, i, d, raw, batch, kind, cache, recs, log, sec, nseg):
        sc = score_raw(raw)[0]
        rec = {"key": design_key(d), "batch": batch, "kind": kind, "params": norm(d), "raw": raw, "score": sc,
               "t": time.time(), "sec": sec, "nseg": nseg}
        cache[rec["key"]] = rec
        if log and self.paths:
            append_log(self.paths, rec)
        recs[i] = rec

def worst_case(p):
    """Collapse the self-capacitance variants into the worst case (kept per variant in 'by_c')."""
    raw = {"vhf": {}, "hf": {}}
    for fk, var in p["vhf"].items():
        r = {"swr": max(v["swr"] for v in var.values()), "by_c": {c: v["swr"] for c, v in var.items()}}
        hz = [v["hz"] for v in var.values() if "hz" in v]
        if hz:
            r["hz"] = min(hz)
        raw["vhf"][fk] = r
    for b, var in p["hf"].items():
        worst = max(var.items(), key=lambda kv: kv[1]["swr"])
        raw["hf"][b] = dict(worst[1], c=worst[0], by_c={c: v["swr"] for c, v in var.items()},
                            whips={c: v["whip"] for c, v in var.items()})
    return raw


# ------------------------------------------------------------------------ search: explorer + styles
# Everything works on the unit cube of the 20 parameters (0..1 each, see decode / encode).
KEYS = list(PARAMS)
INT_IDX = [j for j, k in enumerate(KEYS) if PARAMS[k][2]]
CONT_IDX = [j for j, k in enumerate(KEYS) if not PARAMS[k][2]]

def decode(x):
    return norm({k: PARAMS[k][0] + min(1.0, max(0.0, float(v))) * (PARAMS[k][1] - PARAMS[k][0])
                 for k, v in zip(KEYS, x)})

def encode(d):
    d = norm(d)
    return np.array([min(1.0, max(0.0, (d[k] - PARAMS[k][0]) / (PARAMS[k][1] - PARAMS[k][0]))) for k in KEYS])

def random_valid(n, rng, max_draws=200000):
    """n random designs that pass the hard limits (checked without simulating). If the limits reject almost
    everything (they shouldn't), the rest is filled with rejected designs (they score REJECTED+)."""
    out, draws = [], 0
    while len(out) < n and draws < max_draws:
        x = rng.random(len(KEYS)); draws += 1
        if not misfit(decode(x))[0]:
            out.append(x)
    while len(out) < n:
        out.append(rng.random(len(KEYS)))
    return np.array(out), draws

def import_cma():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")              # its note that cma.plot() needs matplotlib
        import cma
    return cma

def import_hdbscan():
    from sklearn.cluster import HDBSCAN
    return HDBSCAN

def hdbscan_labels(data):
    """HDBSCAN clusters (label -1 = noise). Leaf clusters (fine) first; a population that has converged into one
    tight blob comes back as all noise from them, so then the excess-of-mass clusters with a single cluster allowed."""
    import warnings
    H = import_hdbscan()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lab = H(min_cluster_size=HDB_MIN_CLUSTER, min_samples=HDB_MIN_SAMPLES, cluster_selection_method=HDB_METHOD,
                allow_single_cluster=True).fit(data).labels_
        if len(data) >= HDB_MIN_CLUSTER and not (lab >= 0).any():
            lab = H(min_cluster_size=HDB_MIN_CLUSTER, min_samples=HDB_MIN_SAMPLES, cluster_selection_method="eom",
                    allow_single_cluster=True).fit(data).labels_
    return lab

def hv_excess(fs, fa, fb):
    """Hill-valley test: how much the worst design on the line is worse than the worse end (> tolerance = a ridge).
    Rejected / failed test designs are no evidence of a ridge and are ignored; if all of them are rejected, the two
    ends are separated by designs that can't be built, so they count as different valleys."""
    ok = [v for v in fs if v < REJECTED]
    return (max(ok) - max(fa, fb)) if ok else float("inf")

def rec_parts(rec):
    """(score, non-HF part, HF part) of a stored record; the parts are None for rejected / failed designs."""
    s, parts = score_raw(rec["raw"])
    if not parts:
        return s, None, None
    h = hf_part(parts)
    return s, s - h, h


class Randn:
    """Normal sampler for CMA-ES with its own random state, so a style continues with the same samples
    after a stop / resume (it is saved with the search)."""
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def __call__(self, lam, n):
        return self.rng.standard_normal((lam, n))


class Explorer:
    """L-SHADE (Tanabe & Fukunaga 2014): DE that learns its step size F and mixing rate CR from the trials that
    beat their parent and keeps beaten parents in an archive. Here a generation is evaluated in slices (parts of
    it in several batches), and the population only shrinks when the time controller says so."""

    def __init__(self, X, fit, hfp, rng, H=6, p=0.11):
        self.X, self.fit, self.hfp = np.array(X, float), np.array(fit, float), np.array(hfp, float)
        self.rng, self.H, self.p, self.k = rng, H, p, 0
        self.MF, self.MCR = np.full(H, 0.5), np.full(H, 0.5)
        self.A = np.empty((0, self.X.shape[1]))
        self.gen, self.T, self.active = 0, None, True
        self.hist = [float(self.fit.min())]
        self.dead = np.zeros(len(self.X), bool)                     # culled by the taboo (see cull)
        self.culled = 0

    def new_generation(self):
        rng, X, fit = self.rng, self.X, self.fit
        NP, D = X.shape
        r = rng.integers(self.H, size=NP)
        CR = np.clip(rng.normal(self.MCR[r], 0.1), 0, 1)
        CR[self.MCR[r] < 0] = 0.0
        F = np.empty(NP)
        for i in range(NP):                                         # Cauchy around the remembered F, > 0, <= 1
            while True:
                v = self.MF[r[i]] + 0.1 * math.tan(math.pi * (rng.random() - 0.5))
                if v > 0:
                    F[i] = min(v, 1.0)
                    break
        top = np.argsort(fit)[:max(2, int(round(self.p * NP)))]
        pool = np.vstack([X, self.A]) if len(self.A) else X
        T = np.empty_like(X)
        for i in range(NP):
            pb = X[rng.choice(top)]
            r1 = rng.choice([j for j in range(NP) if j != i])
            while True:
                r2 = int(rng.integers(len(pool)))
                if r2 != i and r2 != r1:
                    break
            v = X[i] + F[i] * (pb - X[i]) + F[i] * (X[r1] - pool[r2])
            mask = rng.random(D) < CR[i]
            mask[rng.integers(D)] = True
            u = np.where(mask, v, X[i])
            lo, hi = u < 0, u > 1                                   # out of range: halfway between parent and bound
            u[lo] = X[i][lo] / 2
            u[hi] = (X[i][hi] + 1) / 2
            T[i] = u
        self.dead_gen = self.dead.copy()
        for i in np.where(self.dead)[0]:                            # taboo: a culled member's trial is a fresh
            T[i] = random_valid(1, rng)[0][0]                       # random design (it always replaces it)
        self._F, self._CR, self.T, self.next = F, CR, T, 0
        self.ft, self.fnh, self.fhp = np.full(NP, np.nan), np.full(NP, np.nan), np.full(NP, np.nan)
        self.hsec, self.tsec = np.zeros(NP), np.zeros(NP)
        ok = self.hfp[np.isfinite(self.hfp)]
        self.min_hfp = float(ok.min()) if len(ok) else float("nan")  # for the near-exact rule (shadow mode)

    def take(self, k):
        idx = list(range(self.next, min(len(self.T), self.next + max(0, int(k)))))
        self.next += len(idx)
        return idx

    def remaining(self):
        return 0 if self.T is None else len(self.T) - self.next

    def put(self, i, f, nonhf, hfp, hsec, tsec):
        self.ft[i], self.fnh[i], self.fhp[i], self.hsec[i], self.tsec[i] = f, nonhf, hfp, hsec, tsec

    def complete(self):
        return self.T is not None and self.next >= len(self.T) and not np.isnan(self.ft).any()

    def finish_generation(self):
        """Selection, memories, archive. Returns the shadow-pruning numbers of this generation: for every trial
        that was simulated, would the exact rule (its non-HF part alone loses to its parent) or the near-exact
        rule (it would need better HF than any member has) have skipped its HF simulations, and was that wrong?"""
        T, ft, par = self.T, self.ft, self.fit.copy()
        dead_gen = self.dead_gen                                    # culled before: their trials are immigrants
        dead_mid = self.dead & ~dead_gen                            # culled during this generation: stay culled
        dead = dead_gen | dead_mid
        sim = np.isfinite(self.fnh) & ~dead                         # immigrants (taboo) are no DE trials
        won = ft <= par
        exact = sim & (self.fnh > par)
        near = sim & (self.fnh + (self.min_hfp if np.isfinite(self.min_hfp) else np.inf) > par)
        sh = {"gen": self.gen + 1, "trials": len(T), "simulated": int(sim.sum()), "won": int((won & sim).sum()),
              "sec_all": float(self.tsec[sim].sum()), "sec_hf": float(self.hsec[sim].sum()),
              "exact_skip": int(exact.sum()), "exact_sec": float(self.hsec[exact].sum()),
              "exact_wrong": int((exact & won).sum()),
              "near_skip": int(near.sum()), "near_sec": float(self.hsec[near].sum()),
              "near_wrong": int((near & won).sum())}
        imp = (ft < self.fit) & ~dead
        if imp.any():                                               # remember what worked, weighted by the gain
            w = self.fit[imp] - ft[imp]
            w = w / w.sum()
            SF, SCR = self._F[imp], self._CR[imp]
            self.MF[self.k] = (w * SF ** 2).sum() / max((w * SF).sum(), 1e-12)
            self.MCR[self.k] = -1.0 if SCR.max() == 0 else (w * SCR ** 2).sum() / max((w * SCR).sum(), 1e-12)
            self.k = (self.k + 1) % self.H
            self.A = np.vstack([self.A, self.X[imp]])               # beaten parents go to the archive
        rep = ((ft <= self.fit) & ~dead_mid) | dead_gen
        self.X[rep], self.fit[rep], self.hfp[rep] = T[rep], ft[rep], self.fhp[rep]
        self.dead[dead_gen] = False
        self.improved = int(rep.sum())
        if len(self.A) > len(self.X):
            self.A = self.A[self.rng.choice(len(self.A), len(self.X), replace=False)]
        self.gen += 1
        self.T = None
        self.hist.append(float(self.fit.min()))
        return sh

    def best_f(self):
        return float(self.fit.min()) if len(self.fit) else float("nan")

    def shrink(self, NPn):
        NPn = max(1, min(int(NPn), len(self.X)))
        if NPn < len(self.X):                                       # the worst designs leave
            keep = np.argsort(self.fit)[:NPn]
            self.X, self.fit, self.hfp, self.dead = self.X[keep], self.fit[keep], self.hfp[keep], self.dead[keep]
            if len(self.A) > NPn:
                self.A = self.A[self.rng.choice(len(self.A), NPn, replace=False)]

    def remove(self, idx):
        """Final phase: members in a known valley simply leave (no random replacement), so what stays is the part of
        the population outside every style's valley, and it can converge."""
        keep = np.setdiff1d(np.arange(len(self.X)), np.asarray(list(idx), int))
        n = len(self.X) - len(keep)
        self.X, self.fit, self.hfp, self.dead = self.X[keep], self.fit[keep], self.hfp[keep], self.dead[keep]
        if len(self.A) > len(self.X):
            self.A = self.A[self.rng.choice(len(self.A), max(1, len(self.X)), replace=False)]
        self.culled += n
        return n

    def cull(self, idx):
        """Taboo: these members sit in the valley of an existing style (a style develops it already), so the
        explorer lets them go: in the next generation each is replaced by a fresh random design."""
        idx = [i for i in idx if not self.dead[i]]
        self.dead[idx] = True
        self.fit[idx] = DEAD_FIT
        self.hfp[idx] = np.nan
        self.culled += len(idx)
        return len(idx)


class Style:
    """One valley of the score, pushed down by CMA-ES. The whole-number parameters (n, m, kA, kB) stay those of
    the design that founded the style; CMA-ES moves the 16 others."""

    def __init__(self, sid, Xm, fm, seed, lam, origin, in_range, found):
        cma = import_cma()
        o = np.argsort(fm)
        Xm, fm = np.asarray(Xm, float)[o], np.asarray(fm, float)[o]
        mu = len(Xm)
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))        # the members' average, biased to the best
        w /= w.sum()
        C = Xm[:, CONT_IDX]
        mean = (w[:, None] * C).sum(0)
        spread = np.sqrt(((C - mean) ** 2).mean(0)) if mu > 1 else np.full(len(CONT_IDX), 0.05)
        sigma0 = float(np.clip(np.sqrt((spread ** 2).mean()), *CMA_SIGMA))
        stds = np.clip(spread / max(sigma0, 1e-9), 0.3, 3.0)
        self.es = cma.CMAEvolutionStrategy(mean, sigma0, {"bounds": [0, 1], "popsize": lam, "CMA_stds": stds,
                                                          "verbose": -9, "seed": seed, "randn": Randn(seed)})
        self.id, self.lam, self.origin, self.in_range, self.found = sid, lam, origin, in_range, found
        self.x0 = Xm[0].copy()                                      # its whole numbers stay these
        self.best_x, self.best_f = Xm[0].copy(), float(fm[0])
        self.start_f, self.hist, self.steps, self.designs, self.members = self.best_f, [self.best_f], 0, 0, mu
        self.state, self.pilot, self.end, self.prev_state = "establishing", False, None, None
        self.recent = Xm[:STORED_MAX].copy()                       # where the style is now (see stored())
        self.merge_check = 0                                        # step of its last merge check

    def ask(self):
        Z = np.array(self.es.ask())
        X = np.repeat(self.x0[None, :], len(Z), 0)
        X[:, CONT_IDX] = Z
        self._Z = Z
        return X

    def tell(self, X, f):
        f = np.asarray(f, float)
        self.es.tell(list(self._Z), list(np.minimum(f, 1e6)))
        del self._Z
        self.steps += 1; self.designs += len(f)
        j = int(np.argmin(f))
        if f[j] < self.best_f:
            self.best_f, self.best_x = float(f[j]), np.array(X[j], float)
        self.hist.append(self.best_f)
        self.recent = np.asarray(X, float)[np.argsort(f)[:max(1, min(STORED_MAX, len(f) // 2))]]

    def stored(self):
        """Points that show where the style is now: its best, its CMA-ES mean and the better half of its last step."""
        m = self.x0.copy()
        m[CONT_IDX] = np.asarray(self.es.mean)
        return np.vstack([self.best_x[None, :], m[None, :], self.recent])

    def stalled(self):
        h, P = self.hist, PATIENCE_STEPS
        return len(h) > P and h[-P - 1] - h[-1] < PATIENCE_GAIN

    def converged(self):
        try:
            return float(np.max(self.es.result.stds)) < 0.002 or bool(self.es.stop())
        except Exception:
            return False

    def patience_used(self):
        """Steps in a row (up to now) that brought less than PATIENCE_GAIN in total."""
        h, k = self.hist, 0
        while k + 1 < len(h) and h[-(k + 2)] - h[-1] < PATIENCE_GAIN:
            k += 1
        return min(k, PATIENCE_STEPS)

    def summary(self):
        d = decode(self.best_x)
        return {"id": self.id, "state": self.state, "pilot": self.pilot, "origin": self.origin,
                "in_range": bool(self.in_range), "lam": self.lam, "steps": self.steps, "designs": self.designs,
                "members": self.members, "start": round(self.start_f, 4), "best": round(self.best_f, 4),
                "build": round(buildability(d)[0], 2), "found": self.found, "end": self.end,
                "patience": self.patience_used(), "ints": {k: d[k] for k in ("n", "m", "kA", "kB")},
                "key": design_key(d)}


class Candidate:
    """A cluster of good explorer designs that may be a new style: it is hill-valley tested against the nearest
    styles first (one style per batch, nearest first)."""

    def __init__(self, cid, Xm, fm, origin, gen):
        o = np.argsort(fm)
        self.cid, self.origin, self.gen = cid, origin, gen
        self.Xm, self.fm = np.asarray(Xm, float)[o], np.asarray(fm, float)[o]
        self.x, self.f = self.Xm[0].copy(), float(self.fm[0])
        self.key = design_key(decode(self.x))
        self.member_keys = [design_key(decode(x)) for x in self.Xm]
        self.tested, self.test = {}, None          # {style id: max excess}, the test in flight


class StyleSearch:
    """Explorer + styles + hill-valley tests + time controller. One batch at a time:
         X, owners = search.ask();  (simulate);  search.tell(X, owners, records, now_h)
    The whole object is pickled after every batch (search_state.pkl), so a stop / crash / reboot loses at most
    the batch in flight, and a resumed run makes exactly the same designs."""

    def __init__(self, X0, f0, hfp0, rng, workers):
        self.rng, self.workers = rng, workers
        self.B = max(2 * LAMBDA_MIN, int(round(workers * BATCH_PER_WORKER)))
        self.lam_in = max(LAMBDA_MIN, int(round(self.B / 2)))
        self.lam_out = max(LAMBDA_MIN, int(round(self.lam_in / 2)))
        self.ex = Explorer(X0, f0, hfp0, rng)
        self.phase = "explore"
        self.styles, self.cands, self.memo, self.next_cid = [], [], {}, 1
        self.log, self.shown = [], 0
        self.batch_hist = []            # [(designs simulated, active hours, longest job s)] of recent batches
        self.gen_end_h = []             # active hours when each explorer generation ended
        self.phases = [("explore", None, 0)]
        self.ctrl, self.G_prev = {}, 20.0
        self.evals = len(X0)
        self.best_x, self.best_f = np.array(X0[int(np.argmin(f0))], float), float(np.min(f0))
        self.cut, self.final_from, self.panic = None, None, False
        self.hv_designs, self.cluster_tot = 0, {"detections": 0, "clusters": 0, "candidates": 0, "same": 0, "new": 0}
        self.cost = {"style": None, "ex": None}       # core-seconds per simulated design (running averages)
        self.merges, self.merge_log = [], []          # hill-valley tests between two styles (see check_merges)

    # ---- helpers
    def say(self, msg):
        self.log.append(msg)

    def new_log_lines(self):
        k, self.shown = self.shown, len(self.log)
        return self.log[k:]

    def style(self, sid):
        return next(s for s in self.styles if s.id == sid)

    def running(self):
        return [s for s in self.styles if s.state in ("establishing", "developing")]

    def unfinished(self):
        return [s for s in self.styles if s.state in ("establishing", "developing", "paused", "cut")]

    def set_phase(self, phase, now_h):
        self.phase = phase
        self.phases.append((phase, now_h, self.evals))
        self.say(f"phase: {phase}")

    # ---- one batch
    def ask(self):
        items = []
        run = self.running()
        for s in run:
            items += [(x, ("st", s.id, j)) for j, x in enumerate(s.ask())]
        n_hv = 0
        for c in self.cands:
            if c.test is None:
                self._plan_test(c)
            if c.test is not None and not c.test["asked"]:
                items += [(x, ("hv", c.cid, j)) for j, x in enumerate(c.test["pts"])]
                c.test["asked"] = True; n_hv += len(c.test["pts"])
        for k, m in enumerate(self.merges):
            if not m["asked"]:
                items += [(x, ("mg", k, j)) for j, x in enumerate(m["pts"])]
                m["asked"] = True; n_hv += len(m["pts"])
        if self.ex.active and self.phase in ("explore", "final"):
            if self.ex.T is None:
                self.ex.new_generation()
            if self.phase == "final":
                S = self.ex.remaining()                     # the small final population rides along
            elif run:
                comp = max(s.lam * (SPLIT_IN if s.in_range else SPLIT_OUT) for s in run)
                S = max(int(round(comp)), self.B - sum(s.lam for s in run) - n_hv)
            else:
                S = self.B
            items += [(self.ex.T[i], ("ex", i)) for i in self.ex.take(S)]
        if not items:
            return np.empty((0, len(KEYS))), []
        return np.array([x for x, _ in items]), [o for _, o in items]

    def tell(self, X, owners, recs, now_h, longest_s=0.0, busy_s=0.0):
        """Returns {"shadow": row, "controller": row, "clusters": row} for the logs (each may be missing)."""
        out = {}
        n_sim = sum(1 for r in recs if r.get("sec"))
        self.batch_hist = (self.batch_hist + [(n_sim, now_h, longest_s, busy_s)])[-40:]
        self.evals += len(recs)
        by_style = {}
        for x, o, r in zip(X, owners, recs):
            sec = (r.get("sec") or {}).get("total")
            if sec:                                         # compute per design, separately for style / explorer work
                k = "style" if o[0] == "st" else "ex"
                self.cost[k] = sec if self.cost.get(k) is None else 0.995 * self.cost[k] + 0.005 * sec
            s_, nh, hp = rec_parts(r)
            if s_ < self.best_f:
                self.best_f, self.best_x = s_, np.array(x, float)
            if o[0] == "st":
                by_style.setdefault(o[1], []).append((o[2], x, s_))
            elif o[0] == "hv":
                c = next(c for c in self.cands if c.cid == o[1])
                c.test["f"][o[2]] = s_
                self.hv_designs += 1
            elif o[0] == "mg":
                self.merges[o[1]]["f"][o[2]] = s_
                self.hv_designs += 1
            else:
                sec = r.get("sec") or {}
                self.ex.put(o[1], s_, np.nan if nh is None else nh, np.nan if hp is None else hp,
                            sec.get("hf", 0.0), sec.get("total", 0.0))
        for sid, lst in by_style.items():
            lst.sort(key=lambda t: t[0])
            self.style(sid).tell(np.array([t[1] for t in lst]), [t[2] for t in lst])
        for s in self.running():
            if s.stalled() or s.converged():
                s.state, s.end = "finished", {"h": round(now_h, 3), "evals": self.evals,
                                              "why": "stalled" if s.stalled() else "converged"}
                self.say(f"style {s.id} finished ({s.end['why']}): {s.start_f:.4f} -> {s.best_f:.4f} in {s.steps} steps,"
                         f" {s.designs} designs")
            elif s.state == "establishing" and s.steps >= ESTABLISH_STEPS:
                s.state = "paused"
                self.say(f"style {s.id} established and paused: {s.start_f:.4f} -> {s.best_f:.4f} in {s.steps} steps")
        self.check_merges(now_h)
        if self.ex.T is not None and self.ex.complete():
            out["shadow"] = self.ex.finish_generation()
            self.gen_end_h.append(now_h)
            if self.phase == "explore":
                out["controller"] = self.control(now_h)
                if self.ex.gen >= DETECT_FROM:
                    out["clusters"] = self.detect(now_h, "explorer")
                if len(self.ex.X) <= NP_MIN:
                    self.start_final(now_h)
            elif self.phase == "final":
                h = self.ex.hist[self.final_from:]
                conv = len(h) > FINAL_PATIENCE and h[-FINAL_PATIENCE - 1] - h[-1] < FINAL_GAIN * abs(h[-1])
                if len(self.ex.X) < FINAL_MIN_POP:
                    out["clusters"] = self.detect(now_h, "final")
                    self.ex.active = False
                    self.say(f"final population: only {len(self.ex.X)} designs left outside the styles' valleys; "
                             "they are clustered for styles, and the explorer stops")
                elif conv or len(h) > FINAL_MAX_GENS:
                    out["clusters"] = self.detect(now_h, "final")
                    self.ex.active = False
                    self.say(f"final population {'converged' if conv else f'stopped after {FINAL_MAX_GENS} generations'}"
                             f" at {h[-1]:.4f}: its clusters become styles too")
                elif EXPLORER_TABOO:
                    out["clusters"] = self.detect(now_h, "final", create=False)   # taboo only
                    if len(self.ex.X) < FINAL_MIN_POP:                             # (it removes members)
                        if len(self.ex.X):
                            out["clusters"] = self.detect(now_h, "final")
                        self.ex.active = False
                        self.say(f"final population: only {len(self.ex.X)} designs left outside the styles' "
                                 "valleys; the explorer stops")
        self.resolve(now_h)
        if self.phase == "final" and not self.ex.active and not self.cands:
            self.set_phase("develop", now_h)
        if self.phase == "develop" and not self.unfinished() and not self.cands:
            self.set_phase("done", now_h)
        return out

    # ---- styles
    def detect(self, now_h, origin, create=True):
        """Cluster the explorer's top 30 % (HDBSCAN); every cluster whose best design is in the top 10 % and not
        yet known becomes a candidate (or directly the first style)."""
        X, f = self.ex.X, self.ex.fit
        ok = np.where(f < REJECTED)[0]
        row = {"h": round(now_h, 3), "gen": self.ex.gen, "origin": origin, "pop": len(f), "top": 0, "clusters": 0,
               "noise": 0, "not_top10": 0, "known": 0, "candidates": 0, "culled": 0}
        self.cluster_tot["detections"] += 1
        if len(ok) < HDB_MIN_CLUSTER:
            return row
        order = ok[np.argsort(f[ok])]
        top = order[:max(HDB_MIN_CLUSTER, int(round(DETECT_TOP * len(f))))]
        best10 = set(order[:max(1, int(round(DETECT_BEST * len(f))))].tolist())
        if EXPLORER_TABOO and self.styles:
            row["culled"] += self.taboo(ok)
            ok = np.where(f < REJECTED)[0]
            if len(ok) < HDB_MIN_CLUSTER:
                return row
            order = ok[np.argsort(f[ok])]
            top = order[:max(HDB_MIN_CLUSTER, int(round(DETECT_TOP * len(f))))]
            best10 = set(order[:max(1, int(round(DETECT_BEST * len(f))))].tolist())
        S = [s.stored() for s in self.styles] if EXPLORER_TABOO else []
        data = np.vstack([X[top]] + S) if S else X[top]
        lab = hdbscan_labels(data)
        lp, ls = lab[:len(top)], lab[len(top):]
        groups = [(top[lp == c], bool((ls == c).any())) for c in sorted(set(lp.tolist()) - {-1})]
        row.update(top=len(top), clusters=len(groups), noise=int((lp == -1).sum()))
        self.cluster_tot["clusters"] += len(groups)
        groups.sort(key=lambda gs: f[gs[0]].min())
        gone = []
        for g, has_style in groups:
            b = int(g[np.argmin(f[g])])
            key = design_key(decode(X[b]))
            if any(c.key == key for c in self.cands):
                continue
            if has_style or any(design_key(decode(X[i])) in self.memo for i in g):
                row["known"] += 1                           # a style is there already (its points are in the
                if EXPLORER_TABOO:                          # cluster, or a member was tested to be in its valley)
                    gone.append(g)
                continue
            if b not in best10:
                row["not_top10"] += 1; continue
            if not create:
                continue
            c = Candidate(self.next_cid, X[g], f[g], origin, self.ex.gen); self.next_cid += 1
            row["candidates"] += 1; self.cluster_tot["candidates"] += 1
            if not self.styles and not self.cands:
                self.new_style(c, now_h)                    # the very first style needs no test: the pilot
            else:
                self.cands.append(c)
        if gone:
            g_all = np.concatenate(gone)
            row["culled"] += self.ex.remove(g_all) if self.phase == "final" else self.ex.cull(g_all)
        return row

    def taboo(self, idx):
        """Cull every explorer member (not only the top) whose cluster holds a style's points or a design that a
        hill-valley test put in a style's valley: the styles develop those valleys, the explorer should look for
        others. Returns how many were culled."""
        X = self.ex.X
        S = [s.stored() for s in self.styles]
        lab = hdbscan_labels(np.vstack([X[idx]] + S))
        lp, ls = lab[:len(idx)], lab[len(idx):]
        out = []
        for c in sorted(set(lp.tolist()) - {-1}):
            g = idx[lp == c]
            if (ls == c).any() or any(design_key(decode(X[i])) in self.memo for i in g):
                out += list(g)
        if self.phase == "final":
            return self.ex.remove(out)
        return self.ex.cull(out)

    def _plan_test(self, c):
        """Next hill-valley test of a candidate: the nearest style it has not been tested against (up to
        HV_NEAREST). None if all of them were tested (then resolve() decides)."""
        near = sorted(self.styles, key=lambda s: float(np.linalg.norm(s.best_x - c.x)))[:HV_NEAREST]
        todo = [s for s in near if s.id not in c.tested]
        if not todo:
            return
        s = todo[0]
        c.test = {"sid": s.id, "fs": s.best_f, "xs": s.best_x.copy(), "asked": False,
                  "pts": [c.x + t * (s.best_x - c.x) for t in HV_POINTS], "f": [np.nan] * len(HV_POINTS)}

    def resolve(self, now_h):
        for c in list(self.cands):
            t = c.test
            if t is not None and t["asked"] and not np.isnan(t["f"]).any():
                excess = hv_excess(t["f"], c.f, t["fs"])
                c.tested[t["sid"]] = round(excess, 4)
                c.test = None
                if excess <= HV_TOL[0] + HV_TOL[1] * abs(min(c.f, t["fs"])):
                    for k in c.member_keys:
                        self.memo[k] = t["sid"]
                    if EXPLORER_TABOO and self.ex.active:
                        keys = set(c.member_keys)
                        idx = [i for i in range(len(self.ex.X))
                               if not self.ex.dead[i] and design_key(decode(self.ex.X[i])) in keys]
                        if self.phase == "final" and self.ex.T is None:
                            self.ex.remove(idx)
                        else:
                            self.ex.cull(idx)
                    self.cands.remove(c); self.cluster_tot["same"] += 1
                    continue
            if c.test is None:
                self._plan_test(c)
                if c.test is None:                          # different from all its nearest styles: new style
                    self.cands.remove(c)
                    establishing = sum(1 for s in self.styles if s.state == "establishing")
                    if self.phase == "explore" and establishing >= MAX_ESTABLISHING:
                        self.say(f"candidate {c.cid} ({c.f:.4f}) is a new valley, but {establishing} styles are "
                                 "establishing already: dropped for now (it comes back if the valley persists)")
                        continue
                    self.new_style(c, now_h)

    def new_style(self, c, now_h):
        sid = len(self.styles) + 1
        d = decode(c.x)
        inr = in_v3_ranges(d)
        s = Style(sid, c.Xm, c.fm, int(self.rng.integers(1, 1 << 30)), self.lam_in if inr else self.lam_out,
                  c.origin, inr, {"h": round(now_h, 3), "evals": self.evals, "gen": self.ex.gen, "cand": c.cid,
                                  "tests": dict(c.tested)})
        if not self.styles:
            s.pilot, s.state = True, "developing"
        elif self.phase != "explore":
            s.state = "developing"
        self.styles.append(s)
        self.memo[c.key] = sid
        self.cluster_tot["new"] += 1
        what = "pilot (develops fully now)" if s.pilot else ("develops" if s.state == "developing" else "establishes")
        tests = ", ".join(f"style {k}: ridge {v:+.3f}" for k, v in c.tested.items())
        self.say(f"NEW STYLE {sid} from the {c.origin} ({c.f:.4f}; n={d['n']} m={d['m']} kA={d['kA']} kB={d['kB']}, "
                 f"{'inside' if inr else 'outside'} the v3 ranges, step {s.lam} designs): {what}"
                 + (f"; hill-valley: {tests}" if tests else ""))

    def check_merges(self, now_h):
        """A style's CMA-ES can drift out of its own valley into a neighbour's. Every MERGE_EVERY steps a running
        style gets a hill-valley test against its nearest style (best to best); if they share a valley, the worse
        of the two stops ("merged into" the other)."""
        for m in [m for m in self.merges if m["asked"] and not np.isnan(m["f"]).any()]:
            self.merges.remove(m)
            a, b = self.style(m["a"]), self.style(m["b"])
            excess = hv_excess(m["f"], m["fa"], m["fb"])
            m["excess"] = round(excess, 4)
            self.merge_log.append({k: m[k] for k in ("a", "b", "fa", "fb", "excess")})
            if excess > HV_TOL[0] + HV_TOL[1] * abs(min(m["fa"], m["fb"])):
                continue
            lose, keep = (a, b) if a.best_f > b.best_f else (b, a)
            if lose.state in ("finished", "cut"):
                continue
            lose.state, lose.end = "finished", {"h": round(now_h, 3), "evals": self.evals,
                                                "why": f"merged into style {keep.id}"}
            self.say(f"style {lose.id} shares the valley of style {keep.id} (ridge {excess:+.3f}): it stops, merged into "
                     f"style {keep.id} ({lose.best_f:.4f} vs {keep.best_f:.4f})")
        busy = {(m["a"], m["b"]) for m in self.merges}
        for a in self.running():
            if a.steps - a.merge_check < MERGE_EVERY:
                continue
            others = [b for b in self.styles if b.id != a.id and not (b.end or {}).get("why", "").startswith("merged")]
            if not others:
                continue
            b = min(others, key=lambda s: float(np.linalg.norm(s.best_x - a.best_x)))
            lo, hi = sorted((a.id, b.id))
            a.merge_check = a.steps
            if (lo, hi) in busy:
                continue
            sa, sb = self.style(lo), self.style(hi)
            self.merges.append({"a": lo, "b": hi, "fa": sa.best_f, "fb": sb.best_f, "asked": False,
                                "pts": [sa.best_x + t * (sb.best_x - sa.best_x) for t in HV_POINTS],
                                "f": [np.nan] * len(HV_POINTS)})
            busy.add((lo, hi))

    def start_final(self, now_h):
        """Discovery ends: the explorer (now NP_MIN) runs on until it converges; every style develops."""
        self.set_phase("final", now_h)
        self.final_from = len(self.ex.hist) - 1
        for s in self.styles:
            if s.state in ("establishing", "paused"):
                s.state = "developing"

    # ---- time controller
    def rate(self):
        """Designs per hour of run time, from the recent batches (what they really achieved)."""
        h = self.batch_hist
        if len(h) < 2:
            return None
        n = sum(x[0] for x in h[1:])
        dt = h[-1][1] - h[0][1]
        return n / dt if dt > 1e-6 and n > 0 else None

    def rate_full(self):
        """Designs per hour with every core busy: workers / (core-hours per design) x FULL_LOAD. Small batches
        (late in discovery) leave cores idle, so the measured rate would underrate the develop phase."""
        h = self.batch_hist[-20:]
        n, busy = sum(x[0] for x in h), sum(x[3] for x in h if len(x) > 3)
        if n < 1 or busy <= 0:
            return self.rate()
        return self.workers * 3600.0 / (busy / n) * FULL_LOAD

    def control(self, now_h, deadline_h=None):
        """Re-plan after every explorer generation: forecast the development of all styles (existing and
        expected), work out when discovery must end, and set the explorer's shrink rate for this generation."""
        deadline = deadline_h or getattr(self, "deadline_h", DEADLINE_H)
        NP, Ns = len(self.ex.X), len(self.styles)
        rate = self.rate_full() or 1.0                              # explorer-type designs per hour, all cores busy
        c_ex = self.cost.get("ex") or 1.0
        c_st = self.cost.get("style") or c_ex
        rate_st = rate * c_ex / c_st                                # style designs per hour
        g = self.gen_end_h
        t_gen = (g[-1] - g[-4]) / 3 if len(g) >= 4 else ((g[-1] - g[0]) / max(1, len(g) - 1) if len(g) >= 2 else 0.2)
        t_gen = max(t_gen, 1e-3)
        t_job = float(np.median([x[2] for x in self.batch_hist[-10:]])) / 3600 if self.batch_hist else 0.01
        done = [s.steps for s in self.styles if s.state == "finished"
                and not (s.end or {}).get("why", "").startswith("merged")]
        dev = float(np.median(done)) if done else float(DEV_STEPS_PRIOR)
        pil = next((s for s in self.styles if s.pilot and s.state != "finished"), None)
        if pil is not None:
            dev = max(dev, pil.steps + 1)
        lam_avg = float(np.mean([s.lam for s in self.styles])) if self.styles else (self.lam_in + self.lam_out) / 2
        left = [(s.lam, max(dev - s.steps, PATIENCE_STEPS)) for s in self.unfinished()]
        W0 = sum(l * k for l, k in left)
        tail_old = max([k * (l / rate_st + t_job) for l, k in left], default=0.0)
        g_det = max(0, self.ex.gen - DETECT_FROM + 1)
        q = STYLE_RATE_MEMORY                                       # recent discoveries count more
        n_eff = sum(q ** (self.ex.gen - s.found["gen"]) for s in self.styles if s.origin == "explorer")
        g_eff = (1 - q ** g_det) / (1 - q)
        new_rate = (n_eff + STYLE_RATE_PRIOR[0]) / (g_eff + STYLE_RATE_PRIOR[1])   # new styles per generation
        t_final = FINAL_MAX_GENS / 2 * max(1.2 * t_job, sum(l for l, k in left) / rate_st + NP_MIN / rate + t_job)
        tail_new = t_final + dev * (lam_avg / rate_st + t_job)
        W_fin = FINAL_MAX_GENS / 2 * NP_MIN                          # the final population (explorer designs)

        def T_dev_for(G):                                           # development time if discovery runs G more gens
            W = W0 + (new_rate * G + 1.0) * lam_avg * dev           # style designs (+1: the final population's)
            return max(W / rate_st + W_fin / rate, tail_old, tail_new), W + W_fin
        avail = deadline - now_h
        if T_dev_for(0.0)[0] >= avail:
            G = 0.0
        else:                                                       # the most generations that still fit
            lo, hi = 0.0, max(1.0, avail / t_gen)
            for _ in range(50):
                mid = (lo + hi) / 2
                if mid * t_gen + T_dev_for(mid)[0] <= avail:
                    lo = mid
                else:
                    hi = mid
            G = lo
        T_dev, W = T_dev_for(G)
        t_end = now_h + G * t_gen
        self.G_prev = G
        r_req = 1.0 - (NP_MIN / NP) ** (1.0 / G) if G >= 1 else 1.0
        r_nat = min(SHRINK_MAX, SHRINK_PER_STYLE * max(0, Ns - (SHRINK_FROM - 1)))
        if G < 1:
            r, mode = 1.0, "time is up: discovery ends"
        elif Ns >= SHRINK_FROM:
            r, mode = min(max(r_req, SHRINK_SLOW * r_nat), SHRINK_PANIC_MAX), "steered"
        elif r_req >= SHRINK_PANIC_FROM or self.panic:
            self.panic = True                               # once time got short, keep shrinking towards the plan
            r, mode = min(r_req, SHRINK_PANIC_MAX), "panic (fewer than 4 styles)"
        else:
            r, mode = 0.0, "no shrink until 4 styles"
        self.np_float = min(getattr(self, "np_float", float(NP)), float(NP)) * (1 - r)
        NPn = max(NP_MIN, int(round(self.np_float)))
        self.ex.shrink(NPn)
        self.ctrl = {"h": round(now_h, 3), "gen": self.ex.gen, "NP": NP, "NP_new": len(self.ex.X), "styles": Ns,
                     "rate": round(rate, 1), "rate_seen": round(self.rate() or 0.0, 1),
                     "sec_per_design_ex": round(c_ex, 1), "sec_per_design_style": round(c_st, 1),
                     "style_rate": round(new_rate, 4), "t_gen_min": round(t_gen * 60, 2), "dev_steps": round(dev, 1),
                     "work_designs": int(W), "T_dev_h": round(T_dev, 3), "discovery_end_h": round(t_end, 3),
                     "gens_left": round(G, 1), "r_req": round(r_req, 4), "r_nat": round(r_nat, 4), "r": round(r, 4),
                     "mode": mode, "predicted_done_h": round(max(now_h, t_end) + T_dev, 3)}
        return self.ctrl

    # ---- deadline
    def cut_off(self, now_h, why):
        """The run time is used up: every style that still runs or waits is flagged (it can go on after `extend`)."""
        self.cut = {"h": round(now_h, 3), "why": why, "phase": self.phase}
        for s in self.styles:
            if s.state in ("establishing", "developing", "paused"):
                s.prev_state, s.state = s.state, "cut"
                s.end = {"h": round(now_h, 3), "evals": self.evals, "why": f"cut off ({s.patience_used()}/"
                                                                             f"{PATIENCE_STEPS} patience used)"}
        self.say(f"cut off: {why}")

    def uncut(self, now_h):
        """After `extend`: continue where the cut-off left the styles."""
        if not self.cut:
            return
        for s in self.styles:
            if s.state == "cut":
                s.state, s.end = s.prev_state or "developing", None
        self.say(f"continuing after the cut-off at {self.cut['h']:.2f} h")
        self.cut = None

    def info(self):
        n = {k: sum(1 for s in self.styles if s.state == k) for k in ("establishing", "developing", "paused",
                                                                     "finished", "cut")}
        st = ", ".join(f"{v} {k}" for k, v in n.items() if v) or "none yet"
        ex = (f"explorer generation {self.ex.gen}, population {len(self.ex.X)}" if self.ex.active
              else "explorer finished")
        return f"phase {self.phase} | {ex} | styles: {st} | candidates {len(self.cands)}"


# ------------------------------------------------------------------------ run
def pid_alive(pid):
    """True if `pid` is a running final_opt process (a stale lock after a crash or reboot is ignored)."""
    if not pid:
        return False
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            return b"final_opt" in fh.read()
    except OSError:
        pass
    if os.path.isdir("/proc"):
        return False
    try:
        os.kill(pid, 0); return True
    except OSError:
        return False

def run_pid(paths):
    pid = read_json(paths.lock, {}).get("pid") if os.path.exists(paths.lock) else None
    return pid if pid_alive(pid) else None

def check_dir_version(paths):
    """A results folder holds results of one model version only (scores of different models don't mix)."""
    v = read_json(paths.version, {}).get("model")
    if v is None and any(os.path.exists(f) for f in (paths.log, paths.csv, paths.ckpt)):
        return False
    if v not in (None, VERSION):
        return False
    write_json(paths.version, {"model": VERSION})
    return True

def prog():
    n = os.path.basename(sys.argv[0]) if sys.argv and sys.argv[0] else ""
    return n if n.endswith(".py") else "final_opt_x.py"

def save_state(paths, ck, search, eng):
    """After every batch: search_state.pkl (the whole search: explorer, styles with their CMA-ES, candidates,
    controller, random states), then checkpoint.json. The previous state file stays as .prev until the new
    checkpoint.json is written, so whenever the program stops or crashes one of the two matches it."""
    ck.update(active_s=eng.active_s, phase=search.phase, best=search.best_f, evals=search.evals)
    blob = pickle.dumps({"batch": ck["batch"], "search": search}, protocol=pickle.HIGHEST_PROTOCOL)
    if os.path.exists(paths.state):
        os.replace(paths.state, paths.state + ".prev")
    with open(paths.state + ".tmp", "wb") as fh:
        fh.write(blob)
    os.replace(paths.state + ".tmp", paths.state)
    write_json(paths.ckpt, ck)

def load_state(paths, ck):
    for fn in (paths.state, paths.state + ".prev"):
        try:
            with open(fn, "rb") as fh:
                st = pickle.load(fh)
        except Exception:
            continue
        if st.get("batch") == ck.get("batch"):
            return st["search"]
    return None

def stop_hours(ck):
    """(planner deadline, when the search itself stops) in hours of run time."""
    dl = ck.get("deadline_h", DEADLINE_H)
    return dl, dl + HIDDEN_BUFFER_H - VERIFY_RESERVE_H

KIND = {"st": lambda o: f"style {o[1]}", "hv": lambda o: f"hv {o[1]}", "mg": lambda o: "hv merge",
        "ex": lambda o: "explorer"}

def cmd_run(a, paths):
    pid = run_pid(paths)
    if pid and pid != os.getpid():
        print(f"Another run (pid {pid}) is already using {paths.root}. Stop it first."); return
    if not check_dir_version(paths):
        print(f"{paths.root} holds results of another program (v1/v2/v3 or another model version); experiment X\n"
              f"needs its own folder. Use --dir results_x (or another new name).")
        return
    try:
        import_cma(); import_hdbscan()
    except ImportError as e:
        if "static TLS" in str(e):
            print(f"{e}\nNEC and scikit-learn don't fit into glibc's static TLS area together. Start the program as\n"
                  f"    GLIBC_TUNABLES=glibc.rtld.optional_static_tls=4096 python3 {prog()} run ...")
        else:
            print(f"missing package ({e}). Install them in this Python with:\n    pip install cma scikit-learn")
        return
    write_json(paths.lock, {"pid": os.getpid()})
    cache = load_log(paths.log)
    ck = read_json(paths.ckpt, {})
    ctl = read_json(paths.control, {})
    ctl.update(state="run", max_inflight=0)
    if not ck:
        ctl.pop("extend", None)
    write_json(paths.control, ctl)
    solver = solver_check()
    eng = Engine(a.workers, paths)
    eng.write_progress(workers=a.workers, solver=solver, version=VERSION, program="x")
    print(f"final_opt X: {a.workers} workers, solver {solver}, memory budget {eng.mem_budget / 1024:.1f} GB, "
          f"results in {paths.root}/  ({len(cache)} designs already simulated)", flush=True)
    try:
        run_search(a, ck, eng, cache, paths)
    except StopRequested:
        eng.write_progress(state="stopped", inflight=0, queued=0)
        print(f"  [{time.strftime('%H:%M:%S')}] stopped. `python3 {prog()} run` continues where it left off.", flush=True)
        ctl = read_json(paths.control, {}); ctl.update(state="run", max_inflight=0)
        write_json(paths.control, ctl)
    except KeyboardInterrupt:
        print(f"interrupted; continue with `python3 {prog()} run`", flush=True)
    finally:
        eng.terminate()
        try:
            os.remove(paths.lock)
        except OSError:
            pass

def run_search(a, ck, eng, cache, paths):
    search = None
    if ck.get("phase_init") == "done":
        search = load_state(paths, ck)
        if search is None:
            raise SystemExit(f"checkpoint.json says batch {ck.get('batch')}, but no matching search_state.pkl was "
                             "found (deleted?). Start a new run in a new folder.")
        eng.active_s = ck.get("active_s", 0.0)
        print(f"\n=== resuming at batch {ck['batch']} ({eng.active_s / 3600:.2f} h used, best {search.best_f:.4f}) ===\n"
              f"    {search.info()}", flush=True)
    else:
        rng = np.random.default_rng(SEED)
        X0, draws = random_valid(POP0, rng)
        designs = [decode(x) for x in X0] + [norm(REFERENCE)]
        ck = {"batch": 0, "deadline_h": DEADLINE_H, "phase_init": "running", "workers": a.workers}
        write_json(paths.ckpt, ck)
        print(f"\n=== initial population: {POP0} random designs that pass the hard limits ({draws} drawn), plus the "
              f"v3 winner as a reference ===", flush=True)
        eng.write_progress(phase="init", batch=0, batch_kind="initial population", batch_size=len(designs),
                           batch_started=time.time(), deadline_h=ck["deadline_h"])
        recs = eng.evaluate(designs, 0, cache, ["init"] * POP0 + ["reference"])
        pp = [rec_parts(r) for r in recs[:POP0]]
        f0 = np.array([p[0] for p in pp]); hfp0 = np.array([np.nan if p[2] is None else p[2] for p in pp])
        search = StyleSearch(X0, f0, hfp0, rng, a.workers)
        ref = recs[-1]
        ck.update(phase_init="done", reference={"score": ref["score"], "build": round(buildability(ref["params"])[0], 2),
                                                "key": ref["key"]})
        print(f"initial population: best {f0.min():.4f}, {int((f0 >= FAILED).sum())} failed; reference (v3 winner) "
              f"scores {ref['score']:.4f} in X's scoring", flush=True)
        search.say(f"batches of ~{search.B} designs; style steps {search.lam_in} (inside the v3 ranges) / "
                   f"{search.lam_out} (outside) designs")
        save_state(paths, ck, search, eng)
    deadline, stop_h = stop_hours(ck)
    search.deadline_h = deadline
    if search.cut and eng.active_s / 3600 < stop_h:
        search.uncut(eng.active_s / 3600)
    progress_search(ck, search, eng)
    while True:
        req = eng.control().get("extend")
        if req and req.get("id", 0) > (ck.get("extend_id") or 0):
            ck["deadline_h"], ck["extend_id"] = float(req["hours"]), req["id"]
            deadline, stop_h = stop_hours(ck)
            search.deadline_h = deadline
            print(f"  [{time.strftime('%H:%M:%S')}] new deadline {deadline:g} h (search stops at {stop_h:g} h)", flush=True)
            save_state(paths, ck, search, eng)
        now_h = eng.active_s / 3600
        if search.phase == "done":
            break
        if now_h >= stop_h:
            search.cut_off(now_h, f"the run time reached {stop_h:g} h (deadline {deadline:g} h + buffer)")
            for line in search.new_log_lines():
                print(f"  [{time.strftime('%H:%M:%S')}] {line}", flush=True)
            save_state(paths, ck, search, eng)
            break
        X, owners = search.ask()
        if not len(X):
            search.set_phase("done", now_h); break
        batch = ck["batch"] + 1
        kinds = [KIND[o[0]](o) for o in owners]
        mix = {}
        for k in kinds:
            mix[k.split()[0]] = mix.get(k.split()[0], 0) + 1
        t0, a0, b0 = time.time(), eng.active_s, eng.busy_s
        eng.write_progress(batch=batch, batch_kind=", ".join(f"{v} {k}" for k, v in mix.items()),
                           batch_size=len(X), batch_started=t0)
        recs = eng.evaluate([decode(x) for x in X], batch, cache, kinds)
        dt, busy = eng.active_s - a0, eng.busy_s - b0
        longest = max([(r.get("sec") or {}).get("max", 0.0) for r in recs] + [0.0])
        phase0 = search.phase
        out = search.tell(X, owners, recs, eng.active_s / 3600, longest, busy)
        ck["batch"] = batch
        append_csv(paths.batches, ["batch", "h", "phase", "designs", "explorer", "style", "hv", "simulated",
                                   "wall_s", "busy_core_s", "idle", "longest_job_s"],
                   [batch, f"{eng.active_s / 3600:.4f}", phase0, len(X), mix.get("explorer", 0), mix.get("style", 0),
                    mix.get("hv", 0), sum(1 for r in recs if r.get("sec")), f"{dt:.1f}", f"{busy:.1f}",
                    f"{max(0.0, 1 - busy / max(dt * eng.workers, 1e-9)):.3f}", f"{longest:.1f}"])
        for name, path in (("shadow", paths.shadow), ("controller", paths.controller), ("clusters", paths.clusters)):
            if name in out:
                append_csv(path, list(out[name]), list(out[name].values()))
        report_batch(batch, search, mix, eng, dt, out)
        save_state(paths, ck, search, eng)
        progress_search(ck, search, eng)
    finished = search.phase == "done"
    ck["finished"] = finished
    save_state(paths, ck, search, eng)
    progress_search(ck, search, eng)
    eng.write_progress(state="finished" if finished else "cut off", inflight=0, queued=0)
    print(f"\nSearch {'finished' if finished else 'stopped at the deadline (styles cut off; `extend` continues them)'}"
          f" after {eng.active_s / 3600:.2f} h. Run `python3 {prog()} report`, then `verify`.", flush=True)

def report_batch(batch, search, mix, eng, dt, out):
    ex = search.ex
    what = ", ".join(f"{v} {k}" for k, v in mix.items())
    print(f"[batch {batch:4d} | {eng.active_s / 3600:5.2f} h | {search.phase:7s}] {what} in {dt / 60:.1f} min | "
          f"best {search.best_f:.4f} | explorer gen {ex.gen} pop {len(ex.X)} best {ex.best_f():.4f}", flush=True)
    for s in search.running():
        print(f"      style {s.id}{' (pilot)' if s.pilot else ''}: step {s.steps}, best {s.best_f:.4f}"
              f" (from {s.start_f:.4f}), patience {s.patience_used()}/{PATIENCE_STEPS}", flush=True)
    c = out.get("controller")
    if c:
        print(f"      plan: discovery ends at {c['discovery_end_h']:.2f} h, development needs {c['T_dev_h']:.2f} h,"
              f" done at {c['predicted_done_h']:.2f} h | shrink {100 * c['r']:.1f}%/gen ({c['mode']})", flush=True)
    for line in search.new_log_lines():
        print(f"  [{time.strftime('%H:%M:%S')}] {line}", flush=True)

def progress_search(ck, search, eng):
    """What `status` shows (progress.json)."""
    dl, stop_h = stop_hours(ck)
    eng.write_progress(phase=search.phase, info=search.info(), best_score=search.best_f, evals=search.evals,
                       active_h=eng.active_s / 3600, deadline_h=dl, stop_h=stop_h, batch_done=ck.get("batch", 0),
                       rate=search.rate(), plan=search.ctrl, styles=[s.summary() for s in search.styles],
                       candidates=len(search.cands), hv_designs=search.hv_designs, clusters=search.cluster_tot,
                       explorer={"gen": search.ex.gen, "pop": len(search.ex.X), "best": search.ex.best_f(),
                                 "active": search.ex.active},
                       reference=ck.get("reference"), cut=search.cut, lam=[search.lam_in, search.lam_out],
                       batch_target=search.B)


# ------------------------------------------------------------------------ other commands
def cmd_control(a, paths, **kw):
    cur = read_json(paths.control, {"state": "run", "max_inflight": 0}); cur.update(kw)
    write_json(paths.control, cur)
    print(f"control: {cur}")

def cmd_extend(a, paths):
    ck = read_json(paths.ckpt, {})
    if not ck:
        print("No run found in", paths.root); return
    dl, stop_h = stop_hours(ck)
    if a.hours is None:
        print(f"deadline now {dl:g} h of run time (the search stops at {stop_h:g} h; the hard stop is "
              f"{dl + HIDDEN_BUFFER_H:g} h).\nChange it with e.g.  extend --hours 26")
        return
    req = {"hours": a.hours, "id": time.time()}
    if run_pid(paths):
        ctl = read_json(paths.control, {"state": "run", "max_inflight": 0}); ctl["extend"] = req
        write_json(paths.control, ctl)
        print(f"sent to the running optimizer: deadline {a.hours:g} h (applied before its next batch; check `status`)")
        return
    ck["deadline_h"], ck["extend_id"] = float(a.hours), req["id"]
    ck.pop("finished", None)
    write_json(paths.ckpt, ck)
    ctl = read_json(paths.control, {})
    if ctl.pop("extend", None) is not None:
        write_json(paths.control, ctl)
    p = read_json(paths.progress, {})
    if p:
        p["deadline_h"], p["stop_h"] = stop_hours(ck)
        if p.get("state") in ("finished", "cut off"):
            p["state"] = "stopped"
        write_json(paths.progress, p)
    print(f"deadline now {a.hours:g} h (search stops at {stop_hours(ck)[1]:g} h). Continue with `python3 {prog()} run`"
          f" (cut-off styles continue where they stopped).")

def style_line(s):
    st = s["state"] + (" (pilot)" if s.get("pilot") else "")
    if s["state"] == "cut":
        st = f"cut off, patience {s['patience']}/{PATIENCE_STEPS}"
    ints = s["ints"]
    return (f"  #{s['id']:<3d} {st:24s} {s['start']:8.4f} -> {s['best']:8.4f}  build {s['build']:4.1f}  "
            f"n{ints['n']} m{ints['m']} kA{ints['kA']} kB{ints['kB']}  {'in ' if s['in_range'] else 'out'}  "
            f"{s['steps']:3d} steps {s['designs']:6d} designs  found {s['found']['h']:.1f} h ({s['origin']})")

def cmd_status(a, paths):
    p = read_json(paths.progress, {}); c = read_json(paths.control, {"state": "run", "max_inflight": 0})
    if not p:
        print("No run found in", paths.root); return
    pid = run_pid(paths)
    age = time.time() - p.get("updated", 0)
    if pid:
        head = f"ACTIVE (pid {pid})"
    elif p.get("state") == "finished":
        head = "FINISHED - run `report`, then `verify`"
    elif p.get("state") == "cut off":
        head = "FINISHED AT THE DEADLINE (styles cut off) - `report`, `verify`; `extend --hours H` + `run` continues"
    elif p.get("state") == "stopped":
        head = "STOPPED - `run` continues where it left off"
    else:
        head = "NOT RUNNING (interrupted?) - `run` continues where it left off"
    print(f"run {head}; control: { {k: v for k, v in c.items() if k != 'extend'} }")
    print(f"solver {p.get('solver', '?')}, {p.get('workers', '?')} workers, memory budget "
          f"{p.get('mem_budget_gb', 0):.1f} GB (sims in flight now ~{p.get('mem_inflight_gb', 0):.1f} GB)"
          + (f", worker restarts {p['restarts']}" if p.get("restarts") else ""))
    print(f"run time {p.get('active_h', 0):.2f} h of {p.get('deadline_h', DEADLINE_H):g} h planned "
          f"(the search stops at {p.get('stop_h', 0):g} h at the latest)")
    if p.get("info"):
        print(p["info"])
    if pid and p.get("batch_kind"):
        print(f"current batch {p.get('batch', '?')} ({p.get('batch_kind')}): {p.get('done_in_batch', 0)}/"
              f"{p.get('total_in_batch', '?')} designs done, {p.get('inflight', 0)} sims in flight, "
              f"{p.get('queued', 0)} queued")
    if "best_score" in p:
        ref = p.get("reference") or {}
        print(f"best RF score so far {p['best_score']:.4f} ({p.get('evals', 0)} designs)"
              + (f"; the v3 winner scores {ref['score']:.4f} here" if ref.get("score") is not None else ""))
    if p.get("rate"):
        print(f"speed: {p['rate']:.0f} designs per hour (recent batches); batches of ~{p.get('batch_target')} designs, "
              f"style steps {p.get('lam', ['?', '?'])[0]} / {p.get('lam', ['?', '?'])[1]} designs (in / out of the v3 ranges)")
    pl = p.get("plan") or {}
    if pl:
        print(f"time plan (at {pl['h']:.2f} h): discovery ends at {pl['discovery_end_h']:.2f} h, development needs "
              f"{pl['T_dev_h']:.2f} h (~{pl['dev_steps']:.0f} steps per style), predicted done at "
              f"{pl['predicted_done_h']:.2f} h; explorer shrink {100 * pl['r']:.1f}%/gen ({pl['mode']})")
    st = p.get("styles") or []
    if st:
        print(f"styles ({len(st)}):   state, score at the start -> best, buildability, whole numbers, v3 ranges")
        for s in st:
            print(style_line(s))
    cl = p.get("clusters") or {}
    if cl.get("detections"):
        print(f"style detection: {cl['detections']} clusterings, {cl['candidates']} candidates -> {cl['new']} new styles,"
              f" {cl['same']} same valley as a style (hill-valley designs {p.get('hv_designs', 0)}),"
              f" {p.get('candidates', 0)} pending")
    sh = read_csv(paths.shadow)
    if sh:
        tot = {k: sum(float(r[k]) for r in sh) for k in ("simulated", "sec_all", "sec_hf", "exact_skip", "exact_sec",
                                                          "exact_wrong", "near_skip", "near_sec", "near_wrong")}
        ex_all = max(tot["sec_all"], 1e-9)
        print(f"HF pruning (shadow mode, {int(tot['simulated'])} explorer trials): exact rule would skip "
              f"{int(tot['exact_skip'])} ({100 * tot['exact_sec'] / ex_all:.1f}% of their compute, wrong "
              f"{int(tot['exact_wrong'])}); near-exact {int(tot['near_skip'])} ({100 * tot['near_sec'] / ex_all:.1f}%, "
              f"wrong {int(tot['near_wrong'])})")
    if pid and c.get("extend") and c["extend"].get("id", 0) > 0:
        print(f"extend request waiting for the next batch: {c['extend']}")
    print(f"last update {age:.0f} s ago")


# ------------------------------------------------------------------------ reports
def ranked(cache, band=None, kinds=None):
    """[(score, parts, record)] of all simulated designs, best first, with the CURRENT weights."""
    out = []
    for r in cache.values():
        if r["score"] >= REJECTED or (kinds and r.get("kind") not in kinds):
            continue
        s, parts = score_raw(r["raw"], band)
        out.append((s, parts, r))
    out.sort(key=lambda x: x[0])
    return out

def load_search(paths):
    try:
        with open(paths.state, "rb") as fh:
            return pickle.load(fh)["search"]
    except Exception:
        return None

def size_text(d):
    """Overall size: longest single part and the largest diameter."""
    d = norm(d)
    a, th, base_r, height = cone_geom(d)
    rad = [base_r] + [L * math.cos(math.radians(dr)) for k, L, dr, c, p in spoke_sets(d)]
    if d["m"] - d["kA"] - d["kB"] > 0:
        rad.append(disc_radius(d) * math.cos(math.radians(d["tilt"])))
    return f"longest part {buildability(d)[1]['longest']:.2f} m, diameter {2 * max(rad):.2f} m"

def dims_text(d, raw):
    """Building dimensions. Rods and spokes start on the hub rim (diameter dtop); lengths are given from the rim."""
    d = norm(d); a, th, base_r, height = cone_geom(d)
    disc_R, tl = disc_radius(d), math.radians(d["tilt"])
    plain = d["m"] - d["kA"] - d["kB"]
    lines = [
        f"  cone      : {d['n']} wires x {d['L'] * 1000:.0f} mm (from the apex rim), {d['cone_wire']:.1f} mm dia,"
        f" {d['half']:.1f} deg from the axis; base dia {2 * base_r * 1000:.0f} mm, height {height * 1000:.0f} mm",
        f"  feed      : apex and hub dia {d['dtop']:.1f} mm (the wires and rods start on their rims), gap {d['gap']:.2f} mm",
        f"  disc      : {d['m']} slots: {plain} plain rods"
        + (f" x {from_rim(d, disc_R, d['tilt']) * 1000:.0f} mm (from the hub rim), {d['disc_rod']:.1f} mm dia,"
           f" {d['tilt']:.1f} deg below horizontal; tips on a {2 * disc_R * math.cos(tl) * 1000:.0f} mm circle"
           if plain > 0 else " (every slot holds a spoke)"),
    ]
    for name, (k, L, dr, cL, cp) in zip(("A (6m)", "B (10m/CB)"), spoke_sets(d)):
        lines.append(f"  spokes {name:10s}: {k} x {from_rim(d, L, dr) * 1000:.0f} mm (3 mm, from the hub rim), {dr:.0f} deg"
                     f" down, coil {cL:.2f} uH centred {from_rim(d, cp, dr) * 1000:.0f} mm from the rim"
                     f" (outer section {(L - cp - COIL_LEN / 2) * 1000:.0f} mm)")
    tube = height + TUBE_EXTRA
    lines.append(f"  body tube : {tube * 1000:.0f} mm from the apex (top at {(FEED_HEIGHT + d['gap'] / 1000 + tube):.2f} m"
                 f" on the tripod); {size_text(d)}")
    if raw and "hf" in raw:
        def wtxt(b):
            ws = sorted((raw["hf"][b].get("whips") or {"": raw["hf"][b]["whip"]}).values())
            return f"{b} {ws[0]:.2f} m" if ws[-1] - ws[0] < 0.005 else f"{b} {ws[0]:.2f}-{ws[-1]:.2f} m"
        lines.append(f"  whip above the tube: {', '.join(wtxt(b) for b in HF_BANDS)}"
                     f"  (range = coil self-capacitance {min(CSELF_SCORE) * 1e12:.1f}-{max(CSELF_SCORE) * 1e12:.1f} pF)"
                     f"\n              with the real ~30 mm tube expect ~10-30 cm more; set it by SWR in the field")
    return "\n".join(lines)

def coil_turns(uH, form_mm=20.0, pitch_mm=2.0):
    """Rough single-layer turns estimate (Wheeler) for a given form and winding pitch."""
    dn = form_mm / 25.4; p = pitch_mm / 25.4
    for N in np.arange(1, 200, 0.5):
        if dn * dn * N * N / (18 * dn + 40 * N * p) >= uH:
            return float(N)
    return float("nan")

def pareto(rk):
    """Designs that no other design beats on both scores (RF lower AND buildability higher), best RF first."""
    front, best_b = [], -1.0
    for s, parts, r in rk:
        b = buildability(r["params"])[0]
        if b > best_b + 1e-9:
            front.append((s, b, r)); best_b = b
    return front

def band_table(top, title):
    hb, vb = list(HF_BANDS), list(VHF_BANDS)
    hdr = (f"{'#':>3} {'score':>6} {'build':>5} | " + " ".join(f"{b:>5}" for b in hb) + " | "
           + " ".join(f"{b:>11}" for b in vb) + " | 868  915 | RXwst | HF type")
    print(title); print(hdr); print("-" * len(hdr))
    for i, (s, parts, r) in enumerate(top):
        sm = summary(r["raw"])
        print(f"{i + 1:3d} {s:6.3f} {buildability(r['params'])[0]:5.1f} | " + " ".join(f"{sm[b]:5.2f}" for b in hb)
              + " | " + " ".join(f"{sm[b]:5.2f}/{sm[b + '_hz']:+5.1f}" for b in vb)
              + f" | {sm.get('ISM868', float('nan')):4.2f} {sm.get('ISM915', float('nan')):4.2f} | {sm['RXworst']:5.2f}"
              + f" | {hf_type(r['raw'])}" + ("   <- v3 winner (reference)" if r.get("kind") == "reference" else ""))

def best_details(r, band=None):
    s, parts = score_raw(r["raw"], band)
    print("\nscore breakdown (lower is better):")
    print("  " + "  ".join(f"{k} {v:+.3f}" for k, v in parts.items()))
    b, bp = buildability(r["params"])
    print(f"buildability {b:.2f} / 10:  v3 ranges {bp['range']:.2f}  sturdy {bp['sturdy']:.2f}  "
          f"compact {bp['compact']:.2f} (longest part {bp['longest']:.2f} m)  coils {bp['coils']:+.2f}")
    hf = r["raw"]["hf"]
    print(f"HF, worst case over coil self-capacitance {', '.join(ckey(c) for c in CSELF_SCORE)} and coax "
          f"{' / '.join(f'{x:g} m' for x in COAX_RUNS)}:")
    for bnd in HF_BANDS:
        h = hf[bnd]
        byc = ", ".join(f"{c}: SWR {v:.2f} (whip {h['whips'][c]:.2f} m)" for c, v in h.get("by_c", {}).items())
        print(f"    {bnd:5s} worst {h['swr']:.2f} | {byc} | gain {h['dx']:+.1f} dBi, shield current {h['icm'] * 100:.0f}%")
    print("dimensions:")
    print(dims_text(r["params"], r["raw"]))
    for name, (k, L, dr, cL, cp) in zip(("A", "B"), spoke_sets(norm(r["params"]))):
        srf = [1e-6 / (2 * math.pi * math.sqrt(cL * 1e-6 * c)) for c in (max(CSELF_SCORE), min(CSELF_SCORE))]
        print(f"  coil {name}: {cL:.2f} uH ~ {coil_turns(cL):.1f} turns on a 20 mm form at 2 mm pitch (rough; measure!)"
              f"\n          accept only if its self-resonance is {srf[0]:.0f}-{srf[1]:.0f} MHz"
              f" (= self-capacitance {min(CSELF_SCORE) * 1e12:.1f}-{max(CSELF_SCORE) * 1e12:.1f} pF, the range scored)")

def style_records(paths, cache, band=None):
    """[(style summary, its best record)] - the styles of the run with their best designs, rescored."""
    search = load_search(paths)
    out = []
    for s in (search.styles if search else []):
        sm = s.summary()
        r = cache.get(sm["key"])
        if r is None or r["score"] >= REJECTED:
            continue
        out.append((sm, r))
    out.sort(key=lambda t: score_raw(t[1]["raw"], band)[0])
    return out, search

def cmd_report(a, paths):
    cache = load_log(paths.log)
    rk = ranked(cache, a.band)
    if not rk:
        print("No results yet in", paths.root); return
    ref = next((r for r in cache.values() if r.get("kind") == "reference"), None)
    mode = f"only {a.band} counts for HF (weight {PAIR_WEIGHT:g})" if a.band else "pair rule for 10m / CB40"
    print(f"{len(cache)} designs simulated; ranked with the CURRENT weights ({mode}).")
    if ref and ref["score"] < REJECTED:
        print(f"reference: the v3 winner scores {score_raw(ref['raw'], a.band)[0]:.4f} here "
              f"(buildability {buildability(ref['params'])[0]:.1f})")
    print()
    if a.rf or a.band:
        top = rk[:a.top]
        band_table(top, f"Best designs by RF score{f' ({a.band} only)' if a.band else ''} "
                        "(SWR per band; VHF/UHF: SWR/horizon dBi; build = buildability 0-10)")
        print("\n#1 in detail:")
        best_details(top[0][2], a.band)
        if not a.no_plots:
            print("\nSimulating curves and patterns for the plots ...", flush=True)
            make_plots([t[2] for t in top[:3]], a, paths)
        return
    st, search = style_records(paths, cache)
    if st:
        print(f"STYLE CATALOGUE ({len(st)} styles; RF score of each style's best, lower is better)")
        hdr = (f"{'#':>3} {'RF':>7} {'build':>5}  {'HF type':7s} {'868':>4} {'915':>4}  {'v3 ranges':9s}  "
               f"{'n/m/kA/kB':11s} {'status':28s} {'designs':>7}  size")
        print(hdr); print("-" * len(hdr))
        for sm, r in st:
            s = score_raw(r["raw"])[0]
            su = summary(r["raw"]); i = sm["ints"]
            status = {"finished": f"finished ({(sm['end'] or {}).get('why', '')})",
                      "cut": f"cut off ({sm['patience']}/{PATIENCE_STEPS} patience)"}.get(sm["state"], sm["state"])
            print(f"{sm['id']:3d} {s:7.3f} {buildability(r['params'])[0]:5.1f}  {hf_type(r['raw']):7s} "
                  f"{su.get('ISM868', float('nan')):4.2f} {su.get('ISM915', float('nan')):4.2f}  "
                  f"{'inside' if sm['in_range'] else 'outside':9s}  {i['n']}/{i['m']}/{i['kA']}/{i['kB']:<5} "
                  f"{status:28s} {sm['designs'] + sm['members']:7d}  {size_text(r['params'])}")
        print("  (designs = the style's own CMA-ES designs + the explorer designs it was founded on)")
    front = pareto(rk)
    print(f"\nPARETO FRONT ({len(front)} designs that no other design beats on both RF score and buildability):")
    for s, b, r in front[:20]:
        i = norm(r["params"])
        print(f"   RF {s:7.3f}  build {b:4.1f}  {hf_type(r['raw']):6s}  n{i['n']} m{i['m']} kA{i['kA']} kB{i['kB']}  "
              f"{size_text(i)}  ({r.get('kind', '')})")
    print()
    band_table(rk[:a.top], "Best designs by RF score (SWR per band; VHF/UHF: SWR/horizon dBi; build = buildability)")
    print("\nBest design in detail:")
    best_details(rk[0][2])
    if a.no_plots:
        return
    pareto_plot(rk, front, st, paths)
    print("\nSimulating curves and patterns for the plots ...", flush=True)
    picks = [r for sm, r in st[:3]] or [t[2] for t in rk[:3]]
    make_plots(picks, a, paths)

def pareto_plot(rk, front, st, paths):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except ImportError:
        return
    s = np.array([x[0] for x in rk]); b = np.array([buildability(x[2]["params"])[0] for x in rk])
    fig, ax = plt.subplots(figsize=(9, 5.5))
    lim = np.percentile(s, 60) if len(s) > 20 else s.max()
    m = s <= lim
    ax.scatter(s[m], b[m], s=4, c="0.75", label=f"designs (best {m.sum()})")
    ax.plot([x[0] for x in front], [x[1] for x in front], "o-", color="#2a78d6", ms=4, label="Pareto front")
    for sm, r in st:
        x, y = score_raw(r["raw"])[0], buildability(r["params"])[0]
        ax.scatter([x], [y], marker="*", s=90, color="#d6572a", zorder=3)
        ax.annotate(f"style {sm['id']}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlabel("RF score (lower is better)"); ax.set_ylabel("buildability (0-10)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8); ax.set_title("RF score vs buildability")
    fig.tight_layout(); fig.savefig(os.path.join(paths.root, "pareto.png"), dpi=120); plt.close(fig)
    print(f"Saved pareto.png in {paths.root}/")

def make_plots(recs, a, paths):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except ImportError:
        print("(no plots: matplotlib is not installed - `pip install matplotlib` in your venv)"); return
    eng = Engine(a.workers)
    fs = [float(x) for x in np.geomspace(100, 3000, 6 if QUICK else 72)]
    curves = {}
    def on(tag, job, res):
        curves.setdefault((tag[0], tag[2]), {})[tag[1]] = res["swr"] if res else float("nan")
    vj = []
    for i, r in enumerate(recs):
        for f in fs:
            for cs in CSELF_SCORE:
                csr = cs if f <= CSELF_MAX_MHZ else COIL_C
                vj.append(((i, f, ckey(cs)), ("vhf", norm(r["params"]), f, "z", 0.0, csr)))
    eng.run_jobs(vj, on)
    fig, ax = plt.subplots(figsize=(11, 4.6))
    for i, r in enumerate(recs):
        for j, cs in enumerate(CSELF_SCORE):
            ax.semilogx(fs, [min(curves[(i, ckey(cs))][f], 6) for f in fs], lw=1.3, ls=("-", "--")[j % 2],
                        color=f"C{i}", label=f"#{i + 1} score {r['score']:.3f}, coil {ckey(cs)}")
    for b, sp in VHF_BANDS.items():
        ax.axvspan(min(sp["freqs"]), max(max(sp["freqs"]), min(sp["freqs"]) * 1.004), color="0.88", zorder=0)
    ax.axvline(RX_RANGE[1], color="0.6", lw=0.8, ls=":")
    ax.set_xticks([108, 146, 200, 300, 435, 600, 868, 1270, 2500, 3000])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlim(100, 3000); ax.set_ylim(1, 6); ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    ax.set_xlabel("MHz"); ax.set_ylabel("SWR"); ax.set_title("VHF/UHF SWR, free space (scored up to 2500 MHz)")
    fig.tight_layout(); fig.savefig(os.path.join(paths.root, "final_vhf_swr.png"), dpi=120); plt.close(fig)
    best = recs[0]; d = norm(best["params"]); hfr = best["raw"]["hf"]
    spans = {"6m": (49.5, 52.5), "10m": (28.0, 29.7), "CB40": (26.9, 27.5)}
    hfjobs = []
    for b, (lo, hi) in spans.items():
        whips = hfr[b].get("whips") or {ckey(COIL_C): hfr[b]["whip"]}
        for cs in CSELF_SCORE:
            w = whips.get(ckey(cs), hfr[b]["whip"])
            for f in np.linspace(lo, hi, 4 if QUICK else 15):
                hfjobs.append(((b, ckey(cs), float(f)), ("hfpoint", d, b, float(f), w, cs)))
    pts = {}
    def on2(tag, job, res):
        pts[tag] = res
    eng.run_jobs(hfjobs, on2)
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (b, (lo, hi)) in zip(axs, spans.items()):
        for j, cs in enumerate(CSELF_SCORE):
            xs = sorted(f for (bb, cc, f) in pts if bb == b and cc == ckey(cs))
            w = (hfr[b].get("whips") or {}).get(ckey(cs), hfr[b]["whip"])
            ax.plot(xs, [pts[(b, ckey(cs), f)] if pts[(b, ckey(cs), f)] else np.nan for f in xs],
                    marker="o", ls=("-", "--")[j % 2], ms=3, label=f"coil {ckey(cs)}, whip {w:.2f} m")
        ax.axvspan(min(HF_BANDS[b]["freqs"]), max(HF_BANDS[b]["freqs"]), color="0.88", zorder=0)
        ax.set_ylim(1, 3); ax.grid(alpha=0.3); ax.legend(fontsize=7)
        ax.set_title(f"{b} (coax {COAX_RUNS[0]:g} m, whip re-tuned per coil)"); ax.set_xlabel("MHz")
    axs[0].set_ylabel("SWR")
    fig.tight_layout(); fig.savefig(os.path.join(paths.root, "final_hf_swr.png"), dpi=120); plt.close(fig)
    make_pattern_plots(best, eng, plt, paths)
    eng.close()
    print(f"Saved final_vhf_swr.png, final_hf_swr.png, patterns_hf.png and patterns_vhf.png in {paths.root}/")

def make_pattern_plots(best, eng, plt, paths):
    """Elevation patterns of the best design: HF on the tripod over ground (whip and coil as in the worst
    scored case), VHF/UHF in free space with the whip at its CB length."""
    d = norm(best["params"]); hfr = best["raw"]["hf"]
    cval = {ckey(c): c for c in CSELF_SCORE}
    jobs = []
    for b, sp in HF_BANDS.items():
        fc = sp["freqs"][len(sp["freqs"]) // 2]
        jobs.append((("hf", b, fc), ("pattern", d, fc, hfr[b]["whip"], cval.get(hfr[b].get("c"), COIL_C), b)))
    wcb = hfr["CB40"]["whip"]
    for b, sp in VHF_BANDS.items():
        f = sp["hz"][0]
        jobs.append((("vhf", b, f), ("pattern", d, f, wcb, COIL_C, None)))
    out = {}
    def on(tag, job, res):
        out[tag] = res
    eng.run_jobs(jobs, on)
    ink2, grid, blue = "#52514e", "#dcdad3", "#2a78d6"

    def style(ax, lo, hi, ticks):
        ax.set_ylim(lo, hi); ax.set_yticks(ticks)
        ax.set_yticklabels(["" if t == lo else (f"+{t}" if t > 0 else str(t)) for t in ticks], fontsize=7)
        ax.grid(color=grid, linewidth=0.6); ax.spines["polar"].set_color(grid)

    hf = sorted([(t, r) for t, r in out.items() if t[0] == "hf" and r], key=lambda tr: list(HF_BANDS).index(tr[0][1]))
    if hf:
        fig, axes = plt.subplots(1, len(hf), figsize=(3.7 * len(hf), 4.3), subplot_kw={"projection": "polar"})
        for ax, ((_, b, f), r) in zip(np.atleast_1d(axes), hf):
            g0, g1 = np.array(r["phi0"])[::-1], np.array(r["phi180"])[::-1]     # index = elevation 0..90
            el = np.arange(91)
            ang = np.concatenate([el, 180 - el[::-1]])
            style(ax, -20, 10, [-20, -10, 0, 10])
            ax.set_thetamin(0); ax.set_thetamax(180)
            ax.set_xticks(np.radians([0, 30, 60, 90, 120, 150, 180]))
            ax.set_xticklabels(["0°", "30°", "60°", "90°", "60°", "30°", "0°"], fontsize=7)
            ax.plot(np.radians(ang), np.clip(np.concatenate([g0, g1[::-1]]), -20, None), color=blue, lw=1.8)
            ax.set_title(f"{b}  ({f:g} MHz, whip {hfr[b]['whip']:.2f} m)", fontsize=10, pad=2)
            ax.text(0.5, 0.06, f"peak {max(g0.max(), g1.max()):+.1f} dBi · at 10° {g0[10]:+.1f} · at 5° {g0[5]:+.1f}",
                    transform=ax.transAxes, ha="center", va="top", fontsize=8, color=ink2)
        fig.suptitle("HF, best design on the tripod over average ground (dBi; rings at -10 / 0 / +10)", fontsize=11)
        fig.tight_layout(rect=(0, 0.02, 1, 0.93))
        fig.savefig(os.path.join(paths.root, "patterns_hf.png"), dpi=130); plt.close(fig)
    vhf = [(t, r) for t, r in out.items() if t[0] == "vhf" and r]
    if vhf:
        fig, axes = plt.subplots(1, len(vhf), figsize=(3.1 * len(vhf), 4.3), subplot_kw={"projection": "polar"})
        th = np.arange(181)
        for ax, ((_, b, f), r) in zip(np.atleast_1d(axes), sorted(vhf, key=lambda tr: tr[0][2])):
            ang = np.concatenate([90 - th, 90 + th[::-1]])          # phi 0 side, then the phi 180 side
            g = np.clip(np.concatenate([r["phi0"], r["phi180"][::-1]]), -20, None)
            style(ax, -20, 6, [-20, -10, 0])
            ax.set_rlabel_position(300)
            ax.set_xticks(np.radians([0, 45, 90, 135, 180, 225, 315]))
            ax.set_xticklabels(["0°", "45°", "up", "45°", "0°", "-45°", "-45°"], fontsize=7)
            ax.plot(np.radians(ang), g, color=blue, lw=1.8)
            hz = horizon_gain(90 - th, np.array(r["phi0"]))
            ax.set_title(f"{b}  ({f:g} MHz)", fontsize=10, pad=6)
            ax.text(0.5, -0.10, f"horizon ±10°: {hz:+.1f} dBi · SWR {r['swr']:.2f}", transform=ax.transAxes,
                    ha="center", va="top", fontsize=8, color=ink2)
        fig.suptitle("VHF/UHF, best design in free space, whip at its CB length (dBi; rings at -10 / 0)",
                     fontsize=11)
        fig.tight_layout(rect=(0, 0.04, 1, 0.93))
        fig.savefig(os.path.join(paths.root, "patterns_vhf.png"), dpi=130); plt.close(fig)

def cmd_verify(a, paths):
    """Re-check every style's best (or the top designs if the run has no styles) with fine segmentation, plus
    stress tests the search never saw and the unscored extras (2.75 / 3.0 GHz, roof scenario)."""
    cache = load_log(paths.log)
    st, search = style_records(paths, cache)
    if st:
        recs, names = [r for sm, r in st][:a.top], [f"style {sm['id']}" for sm, r in st][:a.top]
    else:
        recs = [t[2] for t in ranked(cache)[:a.top]]; names = [f"#{i + 1}" for i in range(len(recs))]
    if not recs:
        print("No results yet."); return
    solver_check()
    eng = Engine(a.workers, fine=True)
    print(f"verify: {len(recs)} designs at {SEG_VHF_FINE} (VHF/UHF) and {SEG_HF_FINE} (HF) segments per wavelength"
          f" (the search used {SEG_VHF} / {SEG_HF})", flush=True)
    t0 = time.time()
    fine = eng.evaluate([r["params"] for r in recs], 0, {}, log=False)
    print(f"  fine re-scoring done ({(time.time() - t0) / 60:.1f} min); stress tests and extras ...", flush=True)
    centres = [sp["hz"][0] for sp in VHF_BANDS.values()]
    jobs = []
    for i, r in enumerate(recs):
        d = norm(r["params"]); hfr = r["raw"]["hf"]; wcb = hfr["CB40"]["whip"]
        main = "10m" if hf_type(r["raw"]) in ("10m", "(10m)") else "CB40"
        for f in centres:
            jobs.append(((i, "whip", f), ("vhf", d, f, "pat", wcb, COIL_C)))
            for cs in (0.5e-12, 2.0e-12):
                jobs.append(((i, f"C{cs * 1e12:.1f}", f), ("vhf", d, f, "z", 0.0, cs)))
            jobs.append(((i, "ground", f), ("ground", d, f, wcb, COIL_C)))
            jobs.append(((i, "roofv", f), ("ground", d, f, hfr[main]["whip"], COIL_C, ROOF_HEIGHT, True)))
        for f in VERIFY_EXTRA_MHZ:
            jobs.append(((i, "extra", f), ("vhf", d, f, "z", 0.0, COIL_C)))
        for b in HF_BANDS:
            jobs.append(((i, "hfC", b), ("hf", d, b, (0.5e-12, 2.0e-12))))
        jobs.append(((i, "roof", main), ("roof", d, hfr[main]["whip"], COIL_C)))
    out = {}
    def on(tag, job, res):
        out[tag] = res
    eng.run_jobs(jobs, on); eng.close()
    nan = float("nan")
    g = lambda t, k: out[t].get(k, nan) if isinstance(out.get(t), dict) else nan
    for i, r in enumerate(recs):
        fs, fraw = fine[i]["score"], fine[i]["raw"]
        print(f"\n{names[i]}  score {score_raw(r['raw'])[0]:.3f} in the search  ->  {fs:.3f} at fine segmentation"
              + ("   (fine simulation FAILED)" if fraw.get("fail") else "")
              + f"   buildability {buildability(r['params'])[0]:.1f}, HF type {hf_type(r['raw'])}")
        print(f"  {'VHF/UHF':12s} " + " ".join(f"{f:>15g}" for f in centres) + "   MHz, SWR/horizon dBi")
        cell = lambda v: f"{v.get('swr', nan):6.2f}/{v.get('hz', nan):+5.1f}".rjust(15)
        print(f"  {'search':12s} " + " ".join(cell(r["raw"]["vhf"][fkey(f)]) for f in centres)
              + "   worst of the scored coil range")
        if not fraw.get("fail"):
            print(f"  {'fine':12s} " + " ".join(cell(fraw["vhf"][fkey(f)]) for f in centres))
        print(f"  {'whip up':12s} " + " ".join(cell(out.get((i, "whip", f)) or {}) for f in centres)
              + "   whip at CB length, nominal coil")
        for cs in ("C0.5", "C2.0"):
            print(f"  {'coil ' + cs[1:] + ' pF':12s} " + " ".join(f"{g((i, cs, f), 'swr'):6.2f}".rjust(15) for f in centres))
        for key, label in (("ground", "on tripod"), ("roofv", "roof 10 m")):
            print(f"  {label:12s} " + " ".join(
                (f"{g((i, key, f), 'swr'):5.2f} {g((i, key, f), 'g_0_5'):+4.1f}/{g((i, key, f), 'g_5_15'):+4.1f}"
                 + ("" if g((i, key, f), "tripod") is True else "*")).rjust(15) for f in centres))
        print("               SWR, then avg dBi at 0.5-5 / 5-15 deg elevation; on tripod: feed at "
              f"{FEED_HEIGHT:g} m, whip at CB length; roof: feed at {ROOF_HEIGHT:g} m on a grounded metal mast,\n"
              f"               whip fixed for the main HF band (* = tripod / mast left out above {TRIPOD_MAX_MHZ:g} MHz)")
        print(f"  {'above RX':12s} " + "  ".join(f"{f:g} MHz: SWR {g((i, 'extra', f), 'swr'):.2f}" for f in VERIFY_EXTRA_MHZ)
              + "   (unscored: may '3 GHz' go on the box?)")
        hfn = r["raw"]["hf"]
        whip = lambda h: f" (whip {h['whip']:.2f})" if isinstance(h, dict) and "whip" in h else ""
        print(f"  {'HF':12s} " + " | ".join(f"{b}: {hfn[b]['swr']:.2f}{whip(hfn[b])}" for b in HF_BANDS) + "   search")
        if not fraw.get("fail"):
            print(f"  {'':12s} " + " | ".join(f"{b}: {fraw['hf'][b]['swr']:.2f}{whip(fraw['hf'][b])}" for b in HF_BANDS)
                  + "   fine")
        for cs in ("0.50pF", "2.00pF"):
            row = []
            for b in HF_BANDS:
                h = (out.get((i, "hfC", b)) or {}).get(cs)
                row.append(f"{b}: {h['swr']:.2f}{whip(h)}" if h else f"{b}: failed")
            print(f"  {'coil ' + cs[:-2] + ' pF':12s} " + " | ".join(row) + "   fine, whip re-tuned")
        rf = [t for t in out if t[0] == i and t[1] == "roof"]
        if rf and out.get(rf[0]):
            main = rf[0][2]
            print(f"  {'roof 10 m':12s} " + " | ".join(f"{b}: {v:.2f}" for b, v in out[rf[0]].items())
                  + f"   one fixed whip ({hfn[main]['whip']:.2f} m, tuned for {main} on the tripod), worst SWR per band")

def parse_set(items):
    d = {}
    for it in items or []:
        k, v = it.split("=", 1)
        if k not in PARAMS:
            raise SystemExit(f"unknown parameter {k}; choose from {list(PARAMS)}")
        d[k] = float(v)
    return d

def cmd_check(a, paths):
    d = norm(dict(REFERENCE, **parse_set(a.set)))
    print("design:", {k: d[k] for k in PARAMS})
    b, bp = buildability(d)
    print(f"buildability {b:.2f} (v3 ranges {bp['range']:.2f}, sturdy {bp['sturdy']:.2f}, compact {bp['compact']:.2f},"
          f" coils {bp['coils']:+.2f}); {size_text(d)}")
    m, why = misfit(d)
    if m:
        print(f"rejected by the hard limits: {why} (x{m:.2f})"); return
    print(f"segments at {RX_RANGE[1]:g} MHz: {model_segments(d)} (limit {MAX_SEGMENTS})")
    solver_check()
    eng = Engine(a.workers, fine=a.fine)
    rec, = eng.evaluate([d], 0, {}, log=False)
    eng.close()
    s, parts = score_raw(rec["raw"])
    if not parts:
        print("simulation failed"); return
    sm = summary(rec["raw"])
    print(f"score {s:.4f}   " + "  ".join(f"{k} {v:+.3f}" for k, v in parts.items()))
    print("  " + "  ".join(f"{k} {sm[k]:.2f}" for k in CSV_BANDS if k in sm) + f"  RX worst {sm.get('RXworst', 0):.2f}"
          + f"  HF type {hf_type(rec['raw'])}")
    print(f"  compute: {rec['sec'].get('total', 0):.0f} core-seconds ({rec['sec'].get('hf', 0):.0f} of them HF)")
    print(dims_text(d, rec["raw"]))

def cmd_deck(a, paths):
    """Write a .nec file (NEC-2 card deck) of a design, to look at in 4nec2 / xnec2c or run in nec2c."""
    if a.set:
        d, raw, name = norm(dict(REFERENCE, **parse_set(a.set))), None, "check"
    else:
        top = ranked(load_log(paths.log))
        if len(top) < a.rank:
            print(f"No design #{a.rank} in {paths.root}"); return
        r = top[a.rank - 1][2]; d, raw, name = norm(r["params"]), r["raw"], f"design{a.rank}"
    if a.band:
        if a.band not in HF_BANDS:
            print(f"--band must be one of {list(HF_BANDS)}"); return
        fr = HF_BANDS[a.band]["freqs"]; fc = fr[len(fr) // 2]
        h = (raw or {}).get("hf", {}).get(a.band, {})
        whip = h.get("whip", HINT[a.band]); cval = {ckey(c): c for c in CSELF_SCORE}.get(h.get("c"), COIL_C)
        m = make_model(d, max(fr), ground=True, whip=whip, coax_run=COAX_RUNS[0], tripod=TRIPOD, hf=True,
                       cself=cval, f_load=fc)
        lo, hi = {"6m": (49.5, 52.5), "10m": (28.0, 29.7), "CB40": (26.9, 27.5)}.get(a.band, (min(fr), max(fr)))
        freqs = list(np.linspace(lo, hi, 13))
        title = f"{name}, HF model {a.band}: over average ground on the tripod, whip {whip:.3f} m, coil C {cval * 1e12:.2f} pF"
        fn = f"{name}_{a.band}.nec"
    else:
        f = a.freq or 145.0
        m = make_model(d, f)
        freqs = [f]
        title = f"{name}, VHF/UHF model at {f:g} MHz, free space, {segmentation()[0]} segments per wavelength"
        fn = f"{name}_{f:g}MHz.nec"
    path = os.path.join(paths.root, fn)
    with open(path, "w") as fh:
        fh.write(to_deck(m, freqs, title))
    print(f"wrote {path}: {len(m.wires)} wires, {m.segments()} segments")


def cmd_stats(a, paths):
    """The run's learnings in one block: phases, styles, time plan vs reality, clustering, compute per design,
    shadow pruning. Works while running and after."""
    search, p = load_search(paths), read_json(paths.progress, {})
    if search is None:
        print("No run state found in", paths.root); return
    workers = int(p.get("workers") or 1)
    B = read_csv(paths.batches)
    now = search.batch_hist[-1][1] if search.batch_hist else 0.0
    print(f"=== experiment X: learnings ({paths.root}, {search.evals} designs, {now:.2f} h) ===")
    print("\nPHASES                 from h    hours   designs   batches  idle cores")
    ph = search.phases + [("(now)", now, search.evals)]
    for (name, h0, e0), (_, h1, e1) in zip(ph[:-1], ph[1:]):
        h0 = h0 or 0.0
        rows = [r for r in B if r["phase"] == name]
        busy = sum(float(r["busy_core_s"]) for r in rows); wall = sum(float(r["wall_s"]) for r in rows)
        idle = 1 - busy / max(wall * workers, 1e-9) if rows else float("nan")
        print(f"  {name:20s} {h0:6.2f}  {h1 - h0:7.2f}  {e1 - e0:8d}  {len(rows):8d}  {100 * idle:9.1f}%")
    print("\nSTYLES  (found at h / designs | origin | steps, designs | start -> best | state)")
    for s in search.styles:
        f = s.found
        curve = s.hist[::max(1, len(s.hist) // 8)] + ([s.hist[-1]] if len(s.hist) > 1 else [])
        print(f"  #{s.id:<3d} {f['h']:6.2f} h / {f['evals']:6d} | {s.origin:8s} | {s.steps:3d} steps, {s.designs:6d} designs"
              f" | {s.start_f:.4f} -> {s.best_f:.4f} | {s.state}{' (pilot)' if s.pilot else ''}"
              f"{' ' + s.end['why'] if s.end else ''}")
        print("        curve: " + " ".join(f"{v:.3f}" for v in curve))
        if f.get("tests"):
            print("        hill-valley at its start: " + ", ".join(f"style {k} ridge {v:+.3f}" for k, v in f["tests"].items()))
    C = read_csv(paths.controller)
    if C:
        print("\nTIME PLAN (every ~5th re-plan): at h | styles | population -> next | forecast: discovery end, "
              "development h, done at | shrink")
        rows = C[::max(1, len(C) // 12)]
        for r in rows + ([C[-1]] if C[-1] is not rows[-1] else []):
            print(f"  {float(r['h']):6.2f} | {int(r['styles']):3d} | {int(r['NP']):4d} -> {int(r['NP_new']):4d} | "
                  f"{float(r['discovery_end_h']):6.2f}  {float(r['T_dev_h']):6.2f}  {float(r['predicted_done_h']):6.2f}"
                  f" | {100 * float(r['r']):4.1f}% {r['mode']}")
        fin = [x for x in search.phases if x[0] in ("final", "done")]
        print("  actual: " + ", ".join(f"{n} at {h:.2f} h" for n, h, e in fin) if fin else "  actual: not there yet")
    K = read_csv(paths.clusters)
    if K:
        n = len(K)
        avg = lambda k: sum(float(r[k]) for r in K) / n
        ct = search.cluster_tot
        print(f"\nCLUSTERING: {n} clusterings; per clustering on average {avg('top'):.0f} designs -> {avg('clusters'):.1f}"
              f" clusters ({avg('noise'):.0f} designs noise), {avg('not_top10'):.1f} without a top-10% design, "
              f"{avg('known'):.1f} already known, {avg('candidates'):.2f} candidates")
        print(f"  candidates {ct['candidates']}: {ct['new']} new styles, {ct['same']} same valley as a style; "
              f"hill-valley designs {search.hv_designs}")
        ml = getattr(search, "merge_log", [])
        print(f"  merge checks (style vs nearest style, every {MERGE_EVERY} steps): {len(ml)} tests, "
              f"{sum(1 for s in search.styles if (s.end or {}).get('why', '').startswith('merged'))} styles merged")
    cache = load_log(paths.log)
    pts = [(r["nseg"], r["sec"]["total"], r["sec"].get("hf", 0.0)) for r in cache.values()
           if r.get("sec") and r["sec"].get("total") and r.get("nseg")]
    if pts:
        a_ = np.array(pts)
        print("\nCOMPUTE PER DESIGN vs SIZE (segments at 2500 MHz): designs, median core-seconds total / HF")
        edges = [0, 1000, 2000, 3000, 4000, 5000, 6000, 8001]
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (a_[:, 0] >= lo) & (a_[:, 0] < hi)
            if m.any():
                print(f"  {lo:5d}-{hi - 1:<5d} {int(m.sum()):6d}   {np.median(a_[m, 1]):7.0f} / {np.median(a_[m, 2]):6.0f}")
        print(f"  all          {len(a_):6d}   {np.median(a_[:, 1]):7.0f} / {np.median(a_[:, 2]):6.0f}   "
              f"(HF = {100 * a_[:, 2].sum() / a_[:, 1].sum():.0f}% of all compute)")
    S = read_csv(paths.shadow)
    if S:
        tot = {k: sum(float(r[k]) for r in S) for k in S[0] if k != "gen"}
        allc = max(tot["sec_all"], 1e-9)
        print(f"\nHF PRUNING, SHADOW MODE ({int(tot['simulated'])} explorer trials simulated, {int(tot['won'])} beat their"
              f" parent; their HF = {100 * tot['sec_hf'] / allc:.0f}% of their compute)")
        for rule in ("exact", "near"):
            print(f"  {rule + ' rule':16s} would skip {int(tot[rule + '_skip']):6d} trials "
                  f"({100 * tot[rule + '_skip'] / max(tot['simulated'], 1):.1f}%), saving {tot[rule + '_sec'] / 3600:.1f}"
                  f" core-hours = {100 * tot[rule + '_sec'] / allc:.1f}% of the explorer's compute; wrong "
                  f"{int(tot[rule + '_wrong'])} times ({100 * tot[rule + '_wrong'] / max(tot[rule + '_skip'], 1):.2f}%"
                  f" of its skips)")


def ensure_static_tls():
    """NEC (pynec-accel) and scikit-learn's OpenMP library both need glibc's small "static TLS" area, and by
    default they don't fit together ("cannot allocate memory in static TLS block"). On Linux the program therefore
    restarts itself once with a larger area (the glibc tunable has to be set before Python starts)."""
    if not sys.platform.startswith("linux") or os.environ.get("FINAL_OPT_REEXEC"):
        return
    tun = os.environ.get("GLIBC_TUNABLES", "")
    if "rtld.optional_static_tls" in tun:
        return
    os.environ["GLIBC_TUNABLES"] = (tun + ":" if tun else "") + "glibc.rtld.optional_static_tls=4096"
    os.environ["FINAL_OPT_REEXEC"] = "1"
    try:
        os.execv(sys.executable, [sys.executable] + sys.argv)
    except OSError as e:
        print(f"(could not restart with a larger static TLS area: {e})", flush=True)


def main():
    ensure_static_tls()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "pause", "resume", "stop", "throttle", "status", "extend", "report",
                                    "verify", "stats", "check", "deck", "csv"])
    ap.add_argument("value", nargs="?", default=None, help="for throttle: number of cores (0 = all)")
    ap.add_argument("--dir", default=None)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--hours", type=float, default=None, help="extend: new deadline (hours of run time)")
    ap.add_argument("--top", type=int, default=None, help="report: how many designs to list (15); verify: how many (20)")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--rf", action="store_true", help="report: best designs by RF score only")
    ap.add_argument("--band", default=None, help="report: re-rank for one HF band (CB40 or 10m); deck: HF band")
    ap.add_argument("--set", action="append", help="check / deck: name=value")
    ap.add_argument("--fine", action="store_true", help="check: use the fine segmentation of `verify`")
    ap.add_argument("--rank", type=int, default=1, help="deck: which design of the ranking")
    ap.add_argument("--freq", type=float, default=None, help="deck: VHF/UHF frequency in MHz")
    a = ap.parse_args()
    if a.quick:
        apply_quick()
    if a.cmd == "report" and a.band and a.band not in PAIR:
        raise SystemExit(f"report --band: choose {' or '.join(PAIR)}")
    paths = Paths(a.dir or ("results_quick_x" if a.quick else "results_x"))
    if a.cmd == "run":
        cmd_run(a, paths)
    elif a.cmd == "pause":
        cmd_control(a, paths, state="pause")
    elif a.cmd == "resume":
        cmd_control(a, paths, state="run")
    elif a.cmd == "stop":
        cmd_control(a, paths, state="stop")
    elif a.cmd == "throttle":
        cmd_control(a, paths, state="run", max_inflight=int(a.value or 0))
    elif a.cmd == "status":
        cmd_status(a, paths)
    elif a.cmd == "extend":
        cmd_extend(a, paths)
    elif a.cmd == "report":
        a.top = a.top or 15
        cmd_report(a, paths)
    elif a.cmd == "verify":
        a.top = a.top or 20
        cmd_verify(a, paths)
    elif a.cmd == "stats":
        cmd_stats(a, paths)
    elif a.cmd == "check":
        cmd_check(a, paths)
    elif a.cmd == "deck":
        cmd_deck(a, paths)
    elif a.cmd == "csv":
        cmd_csv(a, paths)

if __name__ == "__main__":
    main()
