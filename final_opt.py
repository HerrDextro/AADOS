#!/usr/bin/env python3
"""
final_opt.py - combined optimiser: portable discone + HF vertical (layout C), NEC2 via PyNEC.

LAYOUT C (inverted discone, shield -> disc, centre conductor -> cone)
  * The coax comes up through a fibreglass mount tube and ends on the DISC hub.
  * The CONE opens upward from the apex (feed gap between disc hub and apex).
  * A metal body tube ("spine") is bonded to the apex, runs up inside the cone and ends
    TUBE_EXTRA above the cone tips. The telescopic WHIP sits on top of it (no coil).
  * Two sets of straight, coil-loaded SPOKES hang from the disc hub: set A for 6m,
    set B for 10m / CB. Every spoke takes the place of one disc rod.

WHAT IS SIMULATED
  * VHF/UHF: discone + spokes + spine in free space (clean comparison; the whip is left
    out because the cone shields it - `verify` re-checks that for the finalists).
  * HF: everything over average ground, disc hub at FEED_HEIGHT, aluminium tripod under
    the mount tube, coax down the tripod and along the ground (two lengths), choke at
    the hub. The whip is slid to the best SWR across each band, like you do in the field.

TWO STAGES
  1  fast: the discone core is frozen at the v4 winner; only the disc and the spokes are
     optimised, and VHF/UHF is checked up to STAGE1_MAX_MHZ (where spokes matter).
  2  full: everything free (spoke ranges narrowed around the stage-1 elite), full
     frequency coverage. Starts automatically when stage 1 finishes.

COMMANDS (run them from the folder that holds this file and the results folder)
  python3 final_opt.py run             start, or continue after stop / crash / reboot
  python3 final_opt.py pause           sims already running finish, then 0% CPU
  python3 final_opt.py resume          continue after pause
  python3 final_opt.py throttle 6      use at most 6 cores   (throttle 0 = all again)
  python3 final_opt.py stop            sims already running finish, then save and exit
  python3 final_opt.py status          progress, best score, time per generation
  python3 final_opt.py report          top designs, score breakdown, dimensions, plots
                                       (add --no-plots while a run is going: the plots need extra sims)
  python3 final_opt.py verify          stress-test the top designs (whip, coils, ground, tripod)
  python3 final_opt.py check --set kA=3 --set gap=5.5    evaluate one hand-made design
  options: --dir results_final   --workers N   --quick (tiny smoke test in results_quick)

Scores are recomputed from the stored raw results every time you run `report`, so you can
change weights/targets below and re-rank everything that was simulated without re-running.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse, csv, json, math, signal, time
import multiprocessing as mp
from collections import deque
import numpy as np

C0 = 299.792458   # speed of light, m*MHz

# =============================== EDIT HERE ===============================
# ---- fixed physical assumptions
FEED_HEIGHT    = 3.2     # disc hub (feed) above ground, m: tripod 2.94 m + 0.3 m mount
MOUNT_LEN      = 0.30    # fibreglass mount tube between tripod top and disc hub, m
TRIPOD         = True    # aluminium tripod under the mount tube (HF model)
TRIPOD_COLLAR  = 0.90    # height where the legs meet the centre column, m
TRIPOD_FOOT_R  = 1.00    # radius of the circle the feet stand on, m
TRIPOD_COL_R, TRIPOD_LEG_R = 0.0125, 0.008
TUBE_EXTRA     = 0.05    # body tube ends this far above the cone tips, m
SPINE_R_VHF, SPINE_R_HF = 0.003, 0.005   # radius used for the body tube in the VHF / HF model
# The real body tube is ~30 mm. A 30 mm tube next to the cone wires at the apex breaks NEC's thin-wire
# rules, so the model uses a thinner one. Tested with several 30 mm models: from 2m upward the change is
# < 0.1 SWR (the cone shields the tube); on HF the real tube needs the whip ~10-30 cm LONGER than the
# report says and the SWR moves by up to ~0.2 (mostly down). The whip is tuned in the field anyway.
HF_FEED_LEN    = 0.03    # HF model only: feed segment length (numerically clean feed region), m
WHIP_R, WHIP_MAX, WHIP_STEP = 0.005, 3.5, 0.10   # whip radius, longest whip searched, coarse search step (m)
SPOKE_R        = 0.0015  # spokes are 3 mm
FEED_R         = 0.0008
COAX_R, COAX_OFFSET, COAX_RUNS = 0.0025, 0.05, (5.0, 3.0)   # coax: radius, distance from tripod, ground runs
CHOKE_OHM      = 2000.0  # resistive choke 3 cm below the hub
COIL_C, COIL_Q, COIL_LEN = 1.0e-12, 150.0, 0.03   # spoke coils: nominal self-capacitance (F), Q, length (m)
# Robustness: a real coil's self-capacitance is uncertain (~0.5-1.5 pF) and the design turned out to be
# very sensitive to it. Every coil-sensitive simulation (HF, and VHF/UHF up to CSELF_MAX_MHZ) runs at each
# of these values and the WORSE result counts. On HF the whip is re-tuned for each case (you would too).
CSELF_SCORE    = (0.7e-12, 1.4e-12)
CSELF_MAX_MHZ  = 1000.0
CONDUCTIVITY   = 3.5e7   # aluminium, S/m
GROUND         = (13.0, 0.005)   # average soil: relative permittivity, conductivity S/m
SEG_VHF, SEG_HF = 15, 80         # segments per wavelength (both checked for convergence)

# ---- bands. Penalty per band = weight * [(SWR - 1) + K_WALL * max(0, SWR - target)^2]
HF_BANDS = {   # the whip is slid to minimise the worst SWR over these frequencies
    "6m":   dict(freqs=(50.0, 50.5, 51.0),       target=1.5, weight=0.7, whip_max=2.0),
    "10m":  dict(freqs=(28.2, 28.5, 28.8),       target=1.5, weight=0.6, whip_max=WHIP_MAX),
    "CB80": dict(freqs=(26.565, 26.985, 27.405), target=1.5, weight=1.0, whip_max=WHIP_MAX),
}
VHF_BANDS = {  # hz = (pattern frequency, horizon-gain floor in dBi, pattern weight)
    "2m":     dict(freqs=(144.0, 145.0, 146.0),               target=1.5, weight=1.0, hz=(145.0, 0.5, 1.0)),
    "70cm":   dict(freqs=(430.0, 435.0, 440.0),               target=1.5, weight=1.0, hz=(435.0, 0.5, 1.0)),
    "LoRa":   dict(freqs=(863.0, 869.5, 902.0, 915.0, 928.0), target=1.5, weight=0.7, hz=(869.5, -1.0, 1.0)),
    "23cm":   dict(freqs=(1240.0, 1270.0, 1300.0),            target=1.8, weight=0.5, hz=(1270.0, -1.0, 1.0)),
    "2.4GHz": dict(freqs=(2400.0, 2425.0, 2450.0),            target=2.0, weight=0.3, hz=(2425.0, -1.0, 1.5)),
}
RX_RANGE, RX_POINTS, RX_OK, RX_PEN = (108.0, 2600.0), 24, 3.0, 0.5   # receive: penalty per SWR unit above RX_OK
K_WALL        = 10.0
HZ_PEN        = 0.2     # per dB below the horizon floor (x pattern weight)
HZ_BONUS      = 0.05    # reward per dB above the floor (x pattern weight) ...
HZ_BONUS_CAP  = 2.0     # ... up to this many dB
HF_DX_FLOOR, HF_DX_PEN = -3.0, 0.2   # HF gain 3-20 deg below the floor costs this per dB
STAGE1_MAX_MHZ = 1000.0

# ---- search space: name: (low, high, integer?, value while frozen in stage 1 or None = optimised)
PARAMS = {
    "n":         (18,   22,   True,  20),     # cone wires = disc slots
    "half":      (34.0, 41.0, False, 37.8),   # cone half-angle from the axis, deg
    "L":         (0.80, 0.90, False, 0.851),  # cone wire length, m
    "dratio":    (0.40, 0.60, False, None),   # disc diameter / cone base diameter
    "tilt":      (5.0,  25.0, False, None),   # disc rods below horizontal, deg
    "dtop":      (20.0, 40.0, False, 22.0),   # apex / hub diameter, mm
    "gap":       (4.0,  6.5,  False, 5.0),    # feed gap, mm
    "cone_wire": (1.0,  4.0,  False, 1.6),    # mm
    "disc_rod":  (2.0,  4.0,  False, None),   # mm
    "kA": (3, 4, True, None), "LA": (0.30, 0.80, False, None), "dA": (10.0, 60.0, False, None),
    "cA": (0.8, 4.0, False, None), "pA": (0.0, 1.0, False, None),
    "kB": (3, 4, True, None), "LB": (0.70, 1.00, False, None), "dB": (10.0, 60.0, False, None),
    "cB": (1.2, 4.0, False, None), "pB": (0.0, 1.0, False, None),
}
# spoke set A (6m) / B (10m+CB): count k, length L (m), droop d (deg below horizontal),
# coil c (uH), coil position p (0 = 0.10 m from the hub, 1 = 0.10 m before the tip)
SEEDS = [  # good spoke sets from the `loaded` scan (found at 2.0 m height, so only starting points)
    dict(dratio=0.47, tilt=15, disc_rod=2.2, kA=4, LA=0.40, dA=45, cA=3.0, pA=0.75, kB=4, LB=1.0, dB=20, cB=2.6, pB=0.1875),
    dict(dratio=0.47, tilt=15, disc_rod=2.2, kA=4, LA=0.50, dA=30, cA=2.0, pA=0.50, kB=4, LB=1.0, dB=20, cB=2.6, pB=0.1875),
    dict(dratio=0.47, tilt=15, disc_rod=2.2, kA=4, LA=0.60, dA=45, cA=1.5, pA=0.375, kB=4, LB=1.0, dB=20, cB=2.6, pB=0.1875),
    dict(dratio=0.47, tilt=15, disc_rod=2.2, kA=4, LA=0.40, dA=45, cA=3.0, pA=0.75, kB=4, LB=1.0, dB=30, cB=2.6, pB=0.1875),
    dict(dratio=0.47, tilt=15, disc_rod=2.2, kA=4, LA=0.50, dA=30, cA=2.0, pA=0.50, kB=4, LB=1.0, dB=30, cB=2.6, pB=0.1875),
    dict(dratio=0.47, tilt=15, disc_rod=2.2, kA=4, LA=0.50, dA=45, cA=2.0, pA=0.50, kB=4, LB=1.0, dB=45, cB=2.8, pB=0.1875),
]
STAGES = {  # population = pop_per_dim x number of free parameters
    1: dict(pop_per_dim=8, max_gen=60, hours=12.0, patience=10),
    2: dict(pop_per_dim=6, max_gen=100, hours=100.0, patience=10),
}
DE_F, DE_CR, DE_PBEST = (0.5, 0.9), 0.8, 0.2   # mutation factor range, crossover rate, p-best fraction
SEED = 12345
# ========================================================================

QUICK = False   # set by --quick: shrinks everything for a smoke test


def apply_quick():
    global QUICK, RX_POINTS, WHIP_STEP, WHIP_MAX, STAGES
    QUICK = True
    RX_POINTS = 3
    WHIP_STEP, WHIP_MAX = 0.2, 3.0
    STAGES = {1: dict(pop_per_dim=0, pop=5, max_gen=1, hours=1, patience=5),
              2: dict(pop_per_dim=0, pop=5, max_gen=0, hours=1, patience=5)}
    for sp in VHF_BANDS.values():
        sp["freqs"] = (sp["hz"][0],)


# ------------------------------------------------------------------------ helpers
def fkey(f):
    return f"{f:.3f}"

def swr(z, z0=50.0):
    g = abs((z - z0) / (z + z0))
    return (1 + g) / (1 - g) if g < 0.999 else 999.0

def horizon_gain(el, G, lo=-10, hi=10):
    m = (el >= lo) & (el <= hi)
    return float(10 * np.log10(np.mean(10 ** (G[m] / 10))))

def rx_freqs(stage):
    fs = [float(x) for x in np.round(np.geomspace(*RX_RANGE, RX_POINTS), 1)]
    return [f for f in fs if stage == 2 or f <= STAGE1_MAX_MHZ]

def band_active(band, stage):
    return stage == 2 or max(VHF_BANDS[band]["freqs"]) <= STAGE1_MAX_MHZ

def norm(d):
    """Complete a design with frozen/default values and round the integers."""
    d = dict(d)
    for k, (lo, hi, isint, frozen) in PARAMS.items():
        if k not in d:
            d[k] = frozen if frozen is not None else (lo + hi) / 2
        if isint:
            d[k] = int(round(d[k]))
    return d

def cone_geom(d):
    a = d["dtop"] / 2000; th = math.radians(d["half"])
    return a, th, a + d["L"] * math.sin(th), d["L"] * math.cos(th)   # apex radius, angle, base radius, height

def disc_radius(d):
    return d["dratio"] * cone_geom(d)[2]

def coil_pos(L, frac):
    return 0.10 + frac * max(0.0, L - 0.20)

def spoke_sets(d):
    return [(d["kA"], d["LA"], d["dA"], d["cA"], coil_pos(d["LA"], d["pA"])),
            (d["kB"], d["LB"], d["dB"], d["cB"], coil_pos(d["LB"], d["pB"]))]

def spoke_slots(d):
    """{disc slot index: (length, droop, coil uH, coil position)} - sets interleaved."""
    n, sets, out = d["n"], spoke_sets(d), {}
    for j, (k, L, dr, cL, cp) in enumerate(sets):
        for i in range(k):
            idx = int(round((i + j / len(sets)) * n / k)) % n
            while idx in out:
                idx = (idx + 1) % n
            out[idx] = (L, dr, cL, cp)
    return out

def misfit(d):
    """0 if buildable, otherwise how badly the hub limits are exceeded (>1)."""
    d = norm(d)
    circ = math.pi * d["dtop"]
    n_disc = d["n"] - d["kA"] - d["kB"]
    over = max(d["n"] * d["cone_wire"] * 1.3 / circ,
               (n_disc * d["disc_rod"] + (d["kA"] + d["kB"]) * 2000 * SPOKE_R) * 1.3 / circ)
    small = (d["dtop"] / 1000 + 0.010) / (2 * disc_radius(d))    # disc must clear the hub by 10 mm
    w = max(over, small)
    return 0.0 if w <= 1.0 else float(w)


# ------------------------------------------------------------------------ NEC model
def rod(g, tag, p, q, r, smin, smax, qr=1.25):
    """Straight wire p->q whose segments start at smin near p and grow to smax."""
    L = math.dist(p, q)
    if L < 1e-6:
        return tag
    u = [(q[i] - p[i]) / L for i in range(3)]
    k, s, Lt = 0, smin, 0.0
    while s < smax and Lt + s < L - 0.5 * smin:
        Lt += s; s *= qr; k += 1
    if k >= 2:
        m_ = [p[i] + u[i] * Lt for i in range(3)]
        g.wire(tag, k, *p, *m_, r, qr, 1.0); tag += 1
        p = m_
    else:
        Lt = 0.0
    if L - Lt > 1e-6:
        g.wire(tag, max(1, math.ceil((L - Lt) / smax)), *p, *q, r, 1.0, 1.0); tag += 1
    return tag

def build(d, fseg, ground=False, whip=0.0, coax_run=None, tripod=False, hf=False,
          cself=COIL_C, f_load=None):
    """Layout C. Free space: hub at z=0. Over ground: hub at FEED_HEIGHT.
    fseg sets the segmentation (use the highest frequency of the run)."""
    from PyNEC import nec_context
    d = norm(d)
    H = FEED_HEIGHT if ground else 0.0
    c = nec_context(); g = c.get_geometry()
    smax = C0 / fseg / (SEG_HF if hf else SEG_VHF)
    rc, rd, gap = d["cone_wire"] / 2000, d["disc_rod"] / 2000, d["gap"] / 1000
    if hf:
        # HF: the mm-scale feed detail is irrelevant at 6-11 m wavelength but breaks NEC's
        # thin-wire rules, so use a 3 cm feed segment, cone wires from the axis, uniform segments.
        gap = max(gap, HF_FEED_LEN); smin = smax
    else:
        smin = min(smax, max(gap, 4 * max(rc, rd, SPOKE_R)))
    nseg = lambda L: max(1, math.ceil(L / smax))
    a, th, base_r, height = cone_geom(d)
    hub, apex = (0.0, 0.0, H), (0.0, 0.0, H + gap)
    t, loads, shield, choke_tag = 1, [], [], None
    g.wire(t, 1, *hub, *apex, FEED_R, 1, 1); ft = t; t += 1                 # feed gap
    n = d["n"]
    for i in range(n):                                                       # cone, opening upward
        ph = 2 * math.pi * i / n; cp, sp = math.cos(ph), math.sin(ph)
        ring = (a * cp, a * sp, H + gap); tip = (base_r * cp, base_r * sp, H + gap + height)
        if hf:
            t = rod(g, t, apex, tip, rc, smin, smax)
        else:
            g.wire(t, 1, *apex, *ring, rc, 1, 1); t += 1
            t = rod(g, t, ring, tip, rc, smin, smax)
    top = H + gap + height + TUBE_EXTRA                                      # body tube + whip
    t = rod(g, t, apex, (0.0, 0.0, top), SPINE_R_HF if hf else SPINE_R_VHF, smin, smax)
    if whip > 0.01:
        g.wire(t, nseg(whip), 0, 0, top, 0, 0, top + whip, WHIP_R, 1, 1); t += 1
    tl, disc_R, slots = math.radians(d["tilt"]), disc_radius(d), spoke_slots(d)
    for i in range(n):                                                       # disc rods + spokes hang down
        ph = 2 * math.pi * (i + 0.5) / n; cp, sp = math.cos(ph), math.sin(ph)
        if i in slots:
            Ls, dr, cL, cpos = slots[i]; dr = math.radians(dr)
            u = (math.cos(dr) * cp, math.cos(dr) * sp, -math.sin(dr))
            P = lambda s_: (hub[0] + s_ * u[0], hub[1] + s_ * u[1], hub[2] + s_ * u[2])
            a_, b_ = cpos - COIL_LEN / 2, cpos + COIL_LEN / 2
            t = rod(g, t, hub, P(a_), SPOKE_R, smin, smax)
            g.wire(t, 1, *P(a_), *P(b_), SPOKE_R, 1, 1); loads.append((t, cL)); t += 1
            g.wire(t, nseg(Ls - b_), *P(b_), *P(Ls), SPOKE_R, 1, 1); t += 1
        else:
            tip = (disc_R * math.cos(tl) * cp, disc_R * math.cos(tl) * sp, H - disc_R * math.sin(tl))
            t = rod(g, t, hub, tip, rd, smin, smax)
    if coax_run is not None:                                                 # coax shield exterior
        pts = [hub, (0, 0, H - 0.03), (0, 0, H - 0.06), (0, 0, H - 0.15), (COAX_OFFSET, 0, H - 0.20),
               (COAX_OFFSET, 0, 0.05), (COAX_OFFSET + coax_run, 0, 0.05)]
        for j, (p, q) in enumerate(zip(pts[:-1], pts[1:])):
            g.wire(t, 1 if j < 2 else nseg(math.dist(p, q)), *p, *q, COAX_R, 1, 1)
            shield.append(t)
            if j == 1:
                choke_tag = t
            t += 1
    if tripod and ground:                                                    # aluminium tripod, not connected
        col_top = H - MOUNT_LEN
        g.wire(t, nseg(col_top - TRIPOD_COLLAR), 0, 0, TRIPOD_COLLAR, 0, 0, col_top, TRIPOD_COL_R, 1, 1); t += 1
        for ang in (60, 180, 300):
            ar = math.radians(ang)
            foot = (TRIPOD_FOOT_R * math.cos(ar), TRIPOD_FOOT_R * math.sin(ar), 0.02)
            g.wire(t, nseg(math.dist((0, 0, TRIPOD_COLLAR), foot)), 0, 0, TRIPOD_COLLAR, *foot, TRIPOD_LEG_R, 1, 1); t += 1
    c.geometry_complete(1 if ground else 0)
    c.ld_card(5, 0, 0, 0, CONDUCTIVITY, 0, 0)
    if choke_tag and CHOKE_OHM:
        c.ld_card(0, choke_tag, 1, 1, CHOKE_OHM, 0, 0)
    fl = f_load or fseg
    for tg, uH in loads:                         # coil = L parallel C_self parallel R (R sets Q)
        Lh = uH * 1e-6
        c.ld_card(1, tg, 1, 1, COIL_Q * 2 * math.pi * fl * 1e6 * Lh, Lh, cself)
    if ground:
        c.gn_card(2, 0, GROUND[0], GROUND[1], 0, 0, 0, 0)
    c.ex_card(0, ft, 1, 0, 1.0, 0, 0, 0, 0, 0)
    return c, ft, shield


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

def job_hf(d, band, cself=COIL_C):
    """Slide the whip to the lowest worst-case SWR across the band (coax run 1), then
    report both coax runs, the gain at 3-20 deg and the shield current."""
    fr = list(HF_BANDS[band]["freqs"]); fc = fr[len(fr) // 2]; fmax = max(fr)

    def zs(w, run, freqs):
        c, ft, sh = build(d, fmax, ground=True, whip=w, coax_run=run, tripod=TRIPOD, hf=True,
                          cself=cself, f_load=fc)
        if len(freqs) > 1:
            c.fr_card(0, len(freqs), freqs[0], freqs[1] - freqs[0])
        else:
            c.fr_card(0, 1, freqs[0], 0)
        c.xq_card(0)
        return [swr(c.get_input_parameters(i).get_impedance()[0]) for i in range(len(freqs))]

    grid = np.arange(WHIP_STEP, min(WHIP_MAX, HF_BANDS[band].get("whip_max", WHIP_MAX)) + 1e-9, WHIP_STEP)
    sw = np.array([zs(w, COAX_RUNS[0], [fc])[0] for w in grid])
    best = sw.min()
    cand = [i for i in range(len(grid)) if sw[i] <= best + 0.15
            and (i == 0 or sw[i] <= sw[i - 1]) and (i == len(grid) - 1 or sw[i] <= sw[i + 1])]
    i0 = min(cand) if cand else int(np.argmin(sw))          # shortest good dip = quarter-wave solution
    lo, hi = grid[max(i0 - 1, 0)], grid[min(i0 + 1, len(grid) - 1)]
    memo = {}
    def worst(w):
        k = round(w, 4)
        if k not in memo:
            memo[k] = max(zs(w, COAX_RUNS[0], fr))
        return memo[k]
    gr = 0.6180339887
    x1, x2 = hi - gr * (hi - lo), lo + gr * (hi - lo)
    for _ in range(4 if QUICK else 10):                     # golden-section search
        if worst(x1) < worst(x2):
            hi, x2 = x2, x1; x1 = hi - gr * (hi - lo)
        else:
            lo, x1 = x1, x2; x2 = lo + gr * (hi - lo)
    w = min([x1, x2, grid[i0]], key=worst)
    s1 = zs(w, COAX_RUNS[0], fr); s2 = zs(w, COAX_RUNS[1], fr)
    c, ft, sh = build(d, fmax, ground=True, whip=w, coax_run=COAX_RUNS[0], tripod=TRIPOD, hf=True,
                      cself=cself, f_load=fc)
    c.fr_card(0, 1, fc, 0)
    c.rp_card(0, 90, 1, 0, 5, 0, 0, 0.0, 0, 1.0, 0, 0, 0)
    G = np.array(c.get_radiation_pattern(0).get_gain()).ravel(); el = 90 - np.arange(90)
    dx = float(10 * np.log10(np.mean(10 ** (G[(el >= 3) & (el <= 20)] / 10))))
    cur = c.get_structure_currents(0)
    tags = np.array(cur.get_current_segment_tag()); I = np.abs(np.array(cur.get_current()))
    icm = float(I[np.isin(tags, sh)].max() / I[tags == ft][0])
    return {"whip": float(w), "swr_runs": [s1, s2], "swr": float(max(s1 + s2)), "dx": dx, "icm": icm}

def job_ground(d, f, whip, cself=COIL_C):
    """Verification: layout C over ground with tripod, coax (2 m run), choke and whip."""
    c, ft, sh = build(d, f, ground=True, whip=whip, coax_run=2.0, tripod=TRIPOD, cself=cself)
    c.fr_card(0, 1, f, 0)
    c.rp_card(0, 120, 1, 0, 5, 0, 0, 30.0, 0, 0.5, 0, 0, 0)
    z = c.get_input_parameters(0).get_impedance()[0]
    G = np.array(c.get_radiation_pattern(0).get_gain()).ravel(); el = 90 - (30.0 + 0.5 * np.arange(120))
    avg = lambda lo, hi: float(10 * np.log10(np.mean(10 ** (G[(el >= lo) & (el < hi)] / 10))))
    return {"swr": swr(z), "g_0_5": avg(0.5, 5), "g_5_15": avg(5, 15)}

def job_hfpoint(d, band, f, whip, cself=COIL_C):
    """Plot helper: SWR at one frequency with a fixed whip length."""
    c, ft, sh = build(d, f, ground=True, whip=whip, coax_run=COAX_RUNS[0], tripod=TRIPOD, hf=True, cself=cself)
    c.fr_card(0, 1, f, 0); c.xq_card(0)
    return swr(c.get_input_parameters(0).get_impedance()[0])

JOBS = {"vhf": job_vhf, "hf": job_hf, "ground": job_ground, "hfpoint": job_hfpoint}

def run_job(job):
    """Runs in a worker process. Any NEC failure returns None (design scored as failed)."""
    try:
        return JOBS[job[0]](*job[1:])
    except Exception:
        return None

def worker_init():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


# ------------------------------------------------------------------------ scoring
def vhf_points(stage):
    """[(freq, 'z' or 'pat', coil self-capacitance)] needed to score one design."""
    pts = {}
    for b, sp in VHF_BANDS.items():
        if band_active(b, stage):
            for f in sp["freqs"]:
                pts.setdefault(fkey(f), (f, "z"))
            pts[fkey(sp["hz"][0])] = (sp["hz"][0], "pat")
    for f in rx_freqs(stage):
        pts.setdefault(fkey(f), (f, "z"))
    out = []
    for f, kind in pts.values():
        for cs in (CSELF_SCORE if f <= CSELF_MAX_MHZ else (COIL_C,)):
            out.append((f, kind, cs))
    return out

def ckey(cs):
    return f"{cs * 1e12:.2f}pF"

def score_raw(raw, stage):
    """Total score (lower is better) and per-band breakdown, from stored raw results."""
    if raw.get("misfit"):
        return 100.0 + raw["misfit"], {}
    if raw.get("fail"):
        return 999.0, {}
    parts, total = {}, 0.0
    for b, sp in HF_BANDS.items():
        r = raw["hf"][b]
        s = min(r["swr"], 20.0)
        p = sp["weight"] * ((s - 1) + K_WALL * max(0.0, s - sp["target"]) ** 2)
        p += HF_DX_PEN * max(0.0, HF_DX_FLOOR - r["dx"])
        parts[b] = p; total += p
    for b, sp in VHF_BANDS.items():
        if not band_active(b, stage):
            continue
        s = min(max(raw["vhf"][fkey(f)]["swr"] for f in sp["freqs"]), 20.0)
        p = sp["weight"] * ((s - 1) + K_WALL * max(0.0, s - sp["target"]) ** 2)
        fhz, floor, pw = sp["hz"]
        hz = raw["vhf"][fkey(fhz)]["hz"]
        p += pw * (HZ_PEN * (floor - hz) if hz < floor else -HZ_BONUS * min(hz - floor, HZ_BONUS_CAP))
        parts[b] = p; total += p
    rx = sum(max(0.0, raw["vhf"][fkey(f)]["swr"] - RX_OK) for f in rx_freqs(stage))
    parts["RX"] = RX_PEN * rx; total += RX_PEN * rx
    return float(total), parts

def summary(raw, stage):
    """Per band SWR (and horizon gain) for printing / CSV."""
    out = {}
    if raw.get("misfit") or raw.get("fail"):
        return out
    for b in HF_BANDS:
        out[b] = raw["hf"][b]["swr"]; out[b + "_whip"] = raw["hf"][b]["whip"]; out[b + "_dx"] = raw["hf"][b]["dx"]
    for b, sp in VHF_BANDS.items():
        if band_active(b, stage):
            out[b] = max(raw["vhf"][fkey(f)]["swr"] for f in sp["freqs"])
            out[b + "_hz"] = raw["vhf"][fkey(sp["hz"][0])]["hz"]
    out["RXworst"] = max(raw["vhf"][fkey(f)]["swr"] for f in rx_freqs(stage))
    return out


# ------------------------------------------------------------------------ files / control
class Paths:
    def __init__(self, root):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.control = os.path.join(root, "control.json")
        self.progress = os.path.join(root, "progress.json")
        self.ckpt = os.path.join(root, "checkpoint.json")
        self.log = os.path.join(root, "evals.jsonl")
        self.csv = os.path.join(root, "evals.csv")
        self.lock = os.path.join(root, "run.lock")

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

class StopRequested(Exception):
    pass

def design_key(d, stage):
    d = norm(d)
    return f"s{stage}|" + json.dumps({k: (d[k] if PARAMS[k][2] else round(float(d[k]), 5)) for k in PARAMS},
                                      sort_keys=True)

def load_log(paths):
    cache = {}
    if os.path.exists(paths.log):
        with open(paths.log) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                    cache[r["key"]] = r
                except Exception:
                    pass
    return cache

CSV_BANDS = list(HF_BANDS) + list(VHF_BANDS)

def append_log(paths, rec):
    with open(paths.log, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    new = not os.path.exists(paths.csv)
    sm = summary(rec["raw"], rec["stage"])
    with open(paths.csv, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["score", "stage", "gen"] + list(PARAMS) + CSV_BANDS + [b + "_hz" for b in VHF_BANDS]
                       + ["RXworst"] + [b + "_whip" for b in HF_BANDS] + [b + "_dx" for b in HF_BANDS])
        row = [f"{rec['score']:.4f}", rec["stage"], rec["gen"]] + [rec["params"][k] for k in PARAMS]
        row += [f"{sm[b]:.3f}" if b in sm else "" for b in CSV_BANDS]
        row += [f"{sm[b + '_hz']:.2f}" if b + "_hz" in sm else "" for b in VHF_BANDS]
        row += [f"{sm['RXworst']:.2f}" if "RXworst" in sm else ""]
        row += [f"{sm[b + '_whip']:.2f}" if b + "_whip" in sm else "" for b in HF_BANDS]
        row += [f"{sm[b + '_dx']:.2f}" if b + "_dx" in sm else "" for b in HF_BANDS]
        w.writerow(row)


# ------------------------------------------------------------------------ parallel evaluation engine
def est_cost(job):
    """Rough seconds per job, used to start the slowest jobs first."""
    if job[0] == "hf":
        return 12.0
    if job[0] == "hfpoint":
        return 0.5
    base = 0.3 + 20.0 * (job[2] / 2425.0) ** 2.5
    return 3 * base if job[0] == "ground" else base

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

class Engine:
    def __init__(self, workers, paths=None):
        self.workers = workers
        self.paths = paths
        self.pool = mp.get_context("fork").Pool(workers, initializer=worker_init)
        self.active_s = 0.0
        self.progress = {}

    def close(self):
        self.pool.close(); self.pool.join()

    def control(self):
        if not self.paths:
            return {"state": "run", "max_inflight": 0}
        return read_json(self.paths.control, {"state": "run", "max_inflight": 0})

    def write_progress(self, **kw):
        self.progress.update(kw)
        if self.paths:
            self.progress["updated"] = time.time()
            write_json(self.paths.progress, self.progress)

    def run_jobs(self, jobs, on_result, order=None):
        """jobs: list of (tag, job). Calls on_result(tag, job, result) as results arrive.
        Honours pause / throttle / stop from the control file. Default order: slowest first."""
        queue = deque(sorted(jobs, key=order or (lambda tj: -est_cost(tj[1]))))
        inflight = {}
        last_ctl, ctl, shown = 0.0, {"state": "run"}, "run"
        t_prev = time.time()
        while queue or inflight:
            now = time.time()
            if now - last_ctl > 1.0:
                ctl = self.control(); last_ctl = now
                state = ctl.get("state", "run")
                if state != shown:
                    msg = {"pause": "PAUSED - finishing the sims in flight, then idle",
                           "stop": "STOP requested - finishing the sims in flight, then saving",
                           "run": "RESUMED"}.get(state, state)
                    print(f"  [{time.strftime('%H:%M:%S')}] {msg}", flush=True); shown = state
                self.write_progress(state=state, inflight=len(inflight), queued=len(queue))
            state = ctl.get("state", "run")
            if state == "run":
                self.active_s += min(now - t_prev, 5.0)     # capped: laptop sleep doesn't count
                limit = ctl.get("max_inflight", 0) or self.workers
                limit = max(1, min(limit, self.workers))
                while queue and len(inflight) < limit:
                    tag, job = queue.popleft()
                    inflight[self.pool.apply_async(run_job, (job,))] = (tag, job)
            t_prev = now
            if state == "stop" and not inflight:
                raise StopRequested()
            done = [ar for ar in inflight if ar.ready()]
            for ar in done:
                tag, job = inflight.pop(ar)
                try:
                    res = ar.get()
                except Exception:
                    res = None
                on_result(tag, job, res)
            if not done:
                time.sleep(0.25 if state == "run" else 1.0)

    def evaluate(self, designs, stage, gen, cache, log=True):
        """Returns [(score, raw), ...]. Uses / fills the cache; logs finished designs."""
        results = [None] * len(designs)
        partial, jobs = {}, []
        for i, d in enumerate(designs):
            d = norm(d); key = design_key(d, stage)
            if key in cache:
                r = cache[key]; results[i] = (score_raw(r["raw"], stage)[0], r["raw"]); continue
            m = misfit(d)
            if m:
                self._finish(i, d, {"misfit": m}, stage, gen, cache, results, log); continue
            pts = vhf_points(stage)
            partial[i] = {"d": d, "vhf": {}, "hf": {}, "need": len(pts) + len(HF_BANDS) * len(CSELF_SCORE),
                          "fail": False}
            jobs += [((i, "vhf", fkey(f), ckey(cs)), ("vhf", d, f, kind, 0.0, cs)) for f, kind, cs in pts]
            jobs += [((i, "hf", b, ckey(cs)), ("hf", d, b, cs)) for b in HF_BANDS for cs in CSELF_SCORE]
        total = len(designs); self._done = sum(r is not None for r in results)
        self.write_progress(done_in_gen=self._done, total_in_gen=total)

        def on_result(tag, job, res):
            i, kind, k, ck = tag; p = partial[i]
            if res is None:
                p["fail"] = True
            else:
                p[kind].setdefault(k, {})[ck] = res
            p["need"] -= 1
            if p["need"] == 0:
                raw = {"fail": True} if p["fail"] else worst_case(p)
                self._finish(i, p["d"], raw, stage, gen, cache, results, log)
                self._done += 1
                self.write_progress(done_in_gen=self._done, total_in_gen=total)

        # designs are handled in small blocks (slowest jobs first inside a block), so finished
        # designs reach the log steadily and a stop/resume loses at most one block of work
        blk = max(1, self.workers // 2)
        self.run_jobs(jobs, on_result, order=lambda tj: (tj[0][0] // blk, -est_cost(tj[1])))
        return results

    def _finish(self, i, d, raw, stage, gen, cache, results, log):
        sc = score_raw(raw, stage)[0]
        rec = {"key": design_key(d, stage), "stage": stage, "gen": gen, "params": norm(d),
               "raw": raw, "score": sc, "t": time.time()}
        cache[rec["key"]] = rec
        if log and self.paths:
            append_log(self.paths, rec)
        results[i] = (sc, raw)


# ------------------------------------------------------------------------ differential evolution
def free_params(stage):
    return [k for k, v in PARAMS.items() if stage == 2 or v[3] is None]

def stage_bounds(stage, ckpt):
    if stage == 2 and ckpt.get("bounds2"):
        return {k: tuple(v) for k, v in ckpt["bounds2"].items()}
    return {k: (PARAMS[k][0], PARAMS[k][1]) for k in free_params(stage)}

def decode(x, keys, bounds):
    d = {k: bounds[k][0] + float(xi) * (bounds[k][1] - bounds[k][0]) for k, xi in zip(keys, x)}
    return norm(d)

def encode(d, keys, bounds):
    return np.array([min(1.0, max(0.0, (d[k] - bounds[k][0]) / ((bounds[k][1] - bounds[k][0]) or 1)))
                     for k in keys])

def lhs(n, dim, rng):
    out = np.empty((n, dim))
    for j in range(dim):
        out[:, j] = (rng.permutation(n) + rng.random(n)) / n
    return out

def de_trials(X, fit, rng):
    """DE 'current-to-pbest/1/bin': move each member towards one of the best p% and add the
    scaled difference of two random members; then mix coordinates with the parent."""
    NP, D = X.shape
    top = np.argsort(fit)[:max(2, int(round(DE_PBEST * NP)))]
    T = np.empty_like(X)
    for i in range(NP):
        pb = X[rng.choice(top)]
        r1, r2 = rng.choice([j for j in range(NP) if j != i], 2, replace=False)
        F = rng.uniform(*DE_F)
        v = X[i] + F * (pb - X[i]) + F * (X[r1] - X[r2])
        mask = rng.random(D) < DE_CR; mask[rng.integers(D)] = True
        u = np.where(mask, v, X[i])
        lo, hi = u < 0, u > 1                       # out of range: halfway between parent and bound
        u[lo] = X[i][lo] / 2; u[hi] = (X[i][hi] + 1) / 2
        T[i] = u
    return T

def pop_size(stage, dim):
    st = STAGES[stage]
    return st.get("pop") or st["pop_per_dim"] * dim

def initial_population(stage, keys, bounds, rng, cache):
    NP = pop_size(stage, len(keys))
    X = lhs(NP, len(keys), rng)
    seeds = []
    if stage == 1:
        seeds = [norm(s) for s in SEEDS]
    else:                                            # best third of stage 1
        elite = sorted([r for r in cache.values() if r["stage"] == 1 and r["score"] < 100],
                       key=lambda r: r["score"])
        seen = set()
        for r in elite:
            if r["key"] not in seen:
                seen.add(r["key"]); seeds.append(r["params"])
            if len(seeds) >= NP // 3:
                break
    for i, s in enumerate(seeds[:NP]):
        X[i] = encode(s, keys, bounds)
    return X

def narrowed_bounds(cache):
    """Stage-2 bounds: discone ranges in full, spoke ranges around the stage-1 elite."""
    elite = sorted([r for r in cache.values() if r["stage"] == 1 and r["score"] < 100], key=lambda r: r["score"])[:15]
    b = {}
    for k, (lo, hi, isint, frozen) in PARAMS.items():
        if frozen is not None or not elite or k in ("dratio", "tilt", "disc_rod") or isint:
            b[k] = (lo, hi); continue
        vals = [r["params"][k] for r in elite]
        m = 0.2 * (hi - lo)
        b[k] = (max(lo, min(vals) - m), min(hi, max(vals) + m))
    return b


def pid_alive(pid):
    """True if `pid` is a running final_opt process (a stale lock after a crash or reboot is ignored,
    even if the old process number has been reused by something else)."""
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

def cmd_run(a, paths):
    pid = read_json(paths.lock, {}).get("pid") if os.path.exists(paths.lock) else None
    if pid_alive(pid) and pid != os.getpid():
        print(f"Another run (pid {pid}) is already using {paths.root}. Stop it first."); return
    write_json(paths.lock, {"pid": os.getpid()})
    write_json(paths.control, {"state": "run", "max_inflight": 0})
    cache = load_log(paths)
    ckpt = read_json(paths.ckpt, {})
    eng = Engine(a.workers, paths)
    print(f"final_opt: {a.workers} workers, results in {paths.root}/  ({len(cache)} designs already simulated)",
          flush=True)
    try:
        stage = ckpt.get("stage", 1)
        while stage <= 2:
            run_stage(stage, ckpt if ckpt.get("stage") == stage else {}, eng, cache, paths)
            stage += 1
            ckpt = read_json(paths.ckpt, {})
        eng.write_progress(state="finished", inflight=0, queued=0)
        print("\nAll stages finished. Run `python3 final_opt.py report`.", flush=True)
    except StopRequested:
        eng.write_progress(state="stopped", inflight=0, queued=0)
        print(f"  [{time.strftime('%H:%M:%S')}] stopped. `python3 final_opt.py run` continues where it left off.",
              flush=True)
        write_json(paths.control, {"state": "run", "max_inflight": 0})
    except KeyboardInterrupt:
        print("interrupted; continue with `python3 final_opt.py run`", flush=True)
    finally:
        eng.pool.terminate()
        try:
            os.remove(paths.lock)
        except OSError:
            pass

def run_stage(stage, ckpt, eng, cache, paths):
    st = STAGES[stage]
    if stage == 2 and not ckpt.get("bounds2"):
        ckpt = dict(ckpt, bounds2=narrowed_bounds(cache))
    keys = free_params(stage); bounds = stage_bounds(stage, ckpt)
    eng.active_s = ckpt.get("active_s", 0.0)
    eng.write_progress(stage=stage, pop=pop_size(stage, len(keys)), dims=len(keys))
    if ckpt.get("phase") != "gen" or ckpt.get("finished"):
        if ckpt.get("finished"):
            return
        rng = np.random.default_rng(SEED + stage)
        X = initial_population(stage, keys, bounds, rng, cache)
        write_json(paths.ckpt, {"stage": stage, "phase": "init", "bounds2": ckpt.get("bounds2"),
                                "active_s": eng.active_s})
        print(f"\n=== stage {stage}: {len(keys)} free parameters, population {len(X)}, "
              f"initial population ===", flush=True)
        eng.write_progress(gen=0, gen_started=time.time())
        res = eng.evaluate([decode(x, keys, bounds) for x in X], stage, 0, cache)
        fit = np.array([r[0] for r in res]); gen = 0; hist = [float(fit.min())]; gen_times = []
        save_ckpt(paths, stage, X, fit, gen, rng, ckpt.get("bounds2"), eng.active_s, hist, gen_times)
        report_gen(stage, gen, X, fit, keys, bounds, cache, eng)
    else:
        X = np.array(ckpt["pop"]); fit = np.array(ckpt["fit"]); gen = ckpt["gen"]
        rng = np.random.default_rng(); rng.bit_generator.state = ckpt["rng"]
        hist = ckpt.get("hist", []); gen_times = ckpt.get("gen_times", [])
        print(f"\n=== stage {stage}: resuming at generation {gen + 1} (best so far {fit.min():.4f}) ===", flush=True)
    while True:
        if gen >= st["max_gen"]:
            print(f"stage {stage}: reached {st['max_gen']} generations", flush=True); break
        if eng.active_s / 3600 >= st["hours"]:
            print(f"stage {stage}: reached the {st['hours']} h limit", flush=True); break
        if len(hist) > st["patience"] and hist[-st["patience"] - 1] - hist[-1] < 1e-3 * abs(hist[-1]):
            print(f"stage {stage}: converged (no real improvement in {st['patience']} generations)", flush=True); break
        t0 = time.time()
        eng.write_progress(gen=gen + 1, gen_started=t0, gen_times=gen_times[-5:])
        T = de_trials(X, fit, rng)
        res = eng.evaluate([decode(x, keys, bounds) for x in T], stage, gen + 1, cache)
        ft = np.array([r[0] for r in res])
        better = ft <= fit
        X[better] = T[better]; fit[better] = ft[better]
        gen += 1; hist.append(float(fit.min())); gen_times.append(time.time() - t0)
        save_ckpt(paths, stage, X, fit, gen, rng, ckpt.get("bounds2"), eng.active_s, hist, gen_times)
        report_gen(stage, gen, X, fit, keys, bounds, cache, eng, improved=int(better.sum()))
    c2 = read_json(paths.ckpt, {}); c2["finished"] = True
    write_json(paths.ckpt, c2)
    if stage == 1:
        write_json(paths.ckpt, {"stage": 2, "bounds2": narrowed_bounds(cache)})

def save_ckpt(paths, stage, X, fit, gen, rng, bounds2, active_s, hist, gen_times):
    write_json(paths.ckpt, {"stage": stage, "phase": "gen", "gen": gen, "pop": X.tolist(), "fit": fit.tolist(),
                            "rng": rng.bit_generator.state, "bounds2": bounds2, "active_s": active_s,
                            "hist": hist, "gen_times": gen_times})

def report_gen(stage, gen, X, fit, keys, bounds, cache, eng, improved=None):
    b = int(np.argmin(fit)); d = decode(X[b], keys, bounds)
    rec = cache.get(design_key(d, stage))
    sm = summary(rec["raw"], stage) if rec else {}
    bands = " ".join(f"{k} {sm[k]:.2f}" for k in CSV_BANDS if k in sm)
    imp = f", {improved}/{len(X)} improved" if improved is not None else ""
    print(f"[stage {stage} | gen {gen:3d} | {eng.active_s / 3600:5.2f} h] best {fit[b]:.4f}{imp} | {bands}",
          flush=True)
    eng.write_progress(best_score=float(fit[b]), best_gen=gen, gen_done=gen, active_h=eng.active_s / 3600)


# ------------------------------------------------------------------------ other commands
def cmd_control(a, paths, **kw):
    cur = read_json(paths.control, {"state": "run", "max_inflight": 0}); cur.update(kw)
    write_json(paths.control, cur)
    print(f"control: {cur}")

def cmd_status(a, paths):
    p = read_json(paths.progress, {}); c = read_json(paths.control, {"state": "run", "max_inflight": 0})
    if not p:
        print("No run found in", paths.root); return
    pid = read_json(paths.lock, {}).get("pid")
    alive = pid_alive(pid)
    age = time.time() - p.get("updated", 0)
    if alive:
        head = f"ACTIVE (pid {pid})"
    elif p.get("state") == "finished":
        head = "FINISHED - run `report`, then `verify`"
    elif p.get("state") == "stopped":
        head = "STOPPED - `run` continues where it left off"
    else:
        head = "NOT RUNNING (interrupted?) - `run` continues where it left off"
    print(f"run {head}; control: {c}")
    print(f"stage {p.get('stage')} ({p.get('dims')} parameters, population {p.get('pop')}), generation {p.get('gen')}: "
          f"{p.get('done_in_gen', 0)}/{p.get('total_in_gen', '?')} designs done, "
          f"{p.get('inflight', 0)} sims in flight, {p.get('queued', 0)} queued")
    if "best_score" in p:
        print(f"best score so far {p['best_score']:.4f} (after generation {p.get('best_gen')})")
    gt = p.get("gen_times") or []
    if gt:
        print(f"recent generations took {', '.join(f'{x / 60:.0f}' for x in gt)} min")
    if "active_h" in p and p.get("stage") in STAGES:
        st = STAGES[p["stage"]]
        print(f"stage {p['stage']} has used {p['active_h']:.1f} h of {st['hours']:g} h"
              f" (limit {st['max_gen']} generations; stops early after {st['patience']} without real progress)")
    print(f"last update {age:.0f} s ago")

def ranked(cache, stage=None):
    recs = [r for r in cache.values() if r["score"] < 100 and (stage is None or r["stage"] == stage)]
    if stage is None:
        st2 = [r for r in recs if r["stage"] == 2]
        recs = st2 or recs
    out = []
    for r in recs:
        s, parts = score_raw(r["raw"], r["stage"])
        out.append((s, parts, r))
    out.sort(key=lambda x: x[0])
    return out

def dims_text(d, raw):
    d = norm(d); a, th, base_r, height = cone_geom(d)
    lines = [
        f"  cone      : {d['n']} wires x {d['L'] * 1000:.0f} mm, {d['cone_wire']:.1f} mm dia, {d['half']:.1f} deg from the axis;"
        f" base dia {2 * base_r * 1000:.0f} mm, height {height * 1000:.0f} mm",
        f"  feed      : apex/hub dia {d['dtop']:.1f} mm, gap {d['gap']:.2f} mm",
        f"  disc      : {d['n'] - d['kA'] - d['kB']} rods x {disc_radius(d) * 1000:.0f} mm, {d['disc_rod']:.1f} mm dia,"
        f" {d['tilt']:.1f} deg below horizontal",
    ]
    for name, (k, L, dr, cL, cp) in zip(("A (6m)", "B (10m/CB)"), spoke_sets(d)):
        lines.append(f"  spokes {name:10s}: {k} x {L * 1000:.0f} mm (3 mm), {dr:.0f} deg down, coil {cL:.2f} uH centred"
                     f" {cp * 1000:.0f} mm from the hub (outer section {(L - cp - COIL_LEN / 2) * 1000:.0f} mm)")
    tube = height + TUBE_EXTRA
    lines.append(f"  body tube : {tube * 1000:.0f} mm from the apex (top at {(FEED_HEIGHT + d['gap'] / 1000 + tube):.2f} m)")
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
    for N in np.arange(1, 80, 0.5):
        if dn * dn * N * N / (18 * dn + 40 * N * p) >= uH:
            return float(N)
    return float("nan")

def cmd_report(a, paths):
    cache = load_log(paths)
    rk = ranked(cache)
    if not rk:
        print("No results yet in", paths.root); return
    stage = rk[0][2]["stage"]
    print(f"{len(cache)} designs simulated; ranking stage-{stage} results with the CURRENT weights.\n")
    hb, vb = list(HF_BANDS), [b for b in VHF_BANDS if band_active(b, stage)]
    hdr = f"{'#':>2} {'score':>6} | " + " ".join(f"{b:>6}" for b in hb) + " | " + " ".join(f"{b:>11}" for b in vb) + " | RXwrst"
    print("SWR per band (VHF/UHF: SWR/horizon dBi)"); print(hdr); print("-" * len(hdr))
    top, seen = [], set()
    for s, parts, r in rk:
        if r["key"] in seen:
            continue
        seen.add(r["key"]); top.append((s, parts, r))
        sm = summary(r["raw"], r["stage"])
        print(f"{len(top):2d} {s:6.3f} | " + " ".join(f"{sm[b]:6.2f}" for b in hb) + " | "
              + " ".join(f"{sm[b]:5.2f}/{sm[b + '_hz']:+5.1f}" for b in vb) + f" | {sm['RXworst']:5.2f}")
        if len(top) >= a.top:
            break
    print("\nParameters of the top 5:")
    keys = list(PARAMS)
    print("   " + " ".join(f"{k:>7}" for k in keys))
    for i, (s, parts, r) in enumerate(top[:5]):
        p = r["params"]
        print(f"{i + 1:2d} " + " ".join(f"{p[k]:7d}" if PARAMS[k][2] else f"{p[k]:7.3f}" for k in keys))
    s, parts, r = top[0]
    print("\nBest design, score breakdown (lower is better):")
    print("  " + "  ".join(f"{k} {v:+.3f}" for k, v in parts.items()))
    hf = r["raw"]["hf"]
    print(f"  HF, worst case over coil self-capacitance {', '.join(ckey(c) for c in CSELF_SCORE)} and coax "
          f"{' / '.join(f'{x:g} m' for x in COAX_RUNS)}:")
    for b in HF_BANDS:
        h = hf[b]
        byc = ", ".join(f"{c}: SWR {v:.2f} (whip {h['whips'][c]:.2f} m)" for c, v in h.get("by_c", {}).items())
        print(f"    {b:5s} worst {h['swr']:.2f} | {byc} | gain {h['dx']:+.1f} dBi, shield current {h['icm'] * 100:.0f}%")
    print("\nBest design, dimensions:")
    print(dims_text(r["params"], r["raw"]))
    for name, (k, L, dr, cL, cp) in zip(("A", "B"), spoke_sets(norm(r["params"]))):
        srf = [1e-6 / (2 * math.pi * math.sqrt(cL * 1e-6 * c)) for c in (max(CSELF_SCORE), min(CSELF_SCORE))]
        print(f"  coil {name}: {cL:.2f} uH ~ {coil_turns(cL):.1f} turns on a 20 mm form at 2 mm pitch (rough; measure!)"
              f"\n          accept only if its self-resonance is {srf[0]:.0f}-{srf[1]:.0f} MHz"
              f" (= self-capacitance {min(CSELF_SCORE) * 1e12:.1f}-{max(CSELF_SCORE) * 1e12:.1f} pF, the range scored)")
    if a.no_plots:
        return
    print("\nSimulating curves for the plots ...", flush=True)
    make_plots([t[2] for t in top[:3]], a, paths)

def make_plots(recs, a, paths):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except ImportError:
        print("(no plots: matplotlib is not installed - `pip install matplotlib` in your venv)"); return
    eng = Engine(a.workers)
    fs = [float(x) for x in np.geomspace(100, 2700, 6 if QUICK else 70)]
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
    ax.set_xticks([108, 146, 200, 300, 435, 600, 868, 1270, 2425])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlim(100, 2700); ax.set_ylim(1, 6); ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    ax.set_xlabel("MHz"); ax.set_ylabel("SWR"); ax.set_title("VHF/UHF SWR, free space, top designs")
    fig.tight_layout(); fig.savefig(os.path.join(paths.root, "final_vhf_swr.png"), dpi=120); plt.close(fig)
    best = recs[0]; d = norm(best["params"])
    spans = {"6m": (49.5, 52.5), "10m": (28.0, 29.7), "CB80": (26.5, 27.5)}
    hfjobs = []
    for b, (lo, hi) in spans.items():
        whips = best["raw"]["hf"][b].get("whips") or {ckey(COIL_C): best["raw"]["hf"][b]["whip"]}
        for cs in CSELF_SCORE:
            w = whips.get(ckey(cs), best["raw"]["hf"][b]["whip"])
            for f in np.linspace(lo, hi, 4 if QUICK else 15):
                hfjobs.append(((b, ckey(cs), float(f)), ("hfpoint", d, b, float(f), w, cs)))
    pts = {}
    def on2(tag, job, res):
        pts[tag] = res
    eng.run_jobs([(t, j) for t, j in hfjobs], on2)
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (b, (lo, hi)) in zip(axs, spans.items()):
        for j, cs in enumerate(CSELF_SCORE):
            xs = sorted(f for (bb, cc, f) in pts if bb == b and cc == ckey(cs))
            w = (best["raw"]["hf"][b].get("whips") or {}).get(ckey(cs), best["raw"]["hf"][b]["whip"])
            ax.plot(xs, [pts[(b, ckey(cs), f)] if pts[(b, ckey(cs), f)] else np.nan for f in xs],
                    marker="o", ls=("-", "--")[j % 2], ms=3, label=f"coil {ckey(cs)}, whip {w:.2f} m")
        ax.axvspan(min(HF_BANDS[b]["freqs"]), max(HF_BANDS[b]["freqs"]), color="0.88", zorder=0)
        ax.set_ylim(1, 3); ax.grid(alpha=0.3); ax.legend(fontsize=7)
        ax.set_title(f"{b} (coax {COAX_RUNS[0]:g} m, whip re-tuned per coil)"); ax.set_xlabel("MHz")
    axs[0].set_ylabel("SWR")
    fig.tight_layout(); fig.savefig(os.path.join(paths.root, "final_hf_swr.png"), dpi=120); plt.close(fig)
    eng.close()
    print(f"Saved {paths.root}/final_vhf_swr.png and {paths.root}/final_hf_swr.png")

def cmd_verify(a, paths):
    cache = load_log(paths)
    rk = ranked(cache)
    if not rk:
        print("No results yet."); return
    recs, seen = [], set()
    for s, parts, r in rk:
        if r["key"] not in seen:
            seen.add(r["key"]); recs.append(r)
        if len(recs) >= a.top:
            break
    stage = recs[0]["stage"]
    centres = [sp["hz"][0] for b, sp in VHF_BANDS.items() if band_active(b, stage)]
    jobs = []
    for i, r in enumerate(recs):
        d = norm(r["params"]); wcb = r["raw"]["hf"]["CB80"]["whip"]
        for f in centres:
            jobs.append(((i, "whip", f), ("vhf", d, f, "pat", wcb)))
            for cs in (0.5e-12, 2.0e-12):
                jobs.append(((i, f"C{cs * 1e12:.1f}", f), ("vhf", d, f, "z", 0.0, cs)))
            jobs.append(((i, "ground", f), ("ground", d, f, wcb)))
        for cs in (0.5e-12, 2.0e-12):
            for b in HF_BANDS:
                jobs.append(((i, f"C{cs * 1e12:.1f}", b), ("hf", d, b, cs)))
    print(f"verify: {len(jobs)} sims for the top {len(recs)} designs ...", flush=True)
    out = {}
    def on(tag, job, res):
        out[tag] = res
    eng = Engine(a.workers); eng.run_jobs(jobs, on); eng.close()
    g = lambda t, k: (out.get(t) or {}).get(k, float("nan")) if isinstance(out.get(t), dict) else float("nan")
    for i, r in enumerate(recs):
        print(f"\n#{i + 1} (score {r['score']:.3f})  " + ", ".join(f"{k}={r['params'][k]}" for k in ("kA", "kB")))
        print(f"  {'':10s} " + " ".join(f"{int(f):>16d}" for f in centres))
        nom = [r["raw"]["vhf"][fkey(f)] for f in centres]
        print(f"  {'scored':10s} " + " ".join(f"{n_['swr']:6.2f}/{n_.get('hz', float('nan')):+5.1f} dBi".rjust(16) for n_ in nom)
              + "   (worst case of the scored coil range)")
        print(f"  {'whip up':10s} " + " ".join(f"{g((i, 'whip', f), 'swr'):6.2f}/{g((i, 'whip', f), 'hz'):+5.1f} dBi".rjust(16) for f in centres))
        for cs in ("C0.5", "C2.0"):
            print(f"  {'coil ' + cs[1:] + 'pF':10s} " + " ".join(f"{g((i, cs, f), 'swr'):6.2f}".rjust(16) for f in centres))
        print(f"  {'on tripod':10s} " + " ".join(f"{g((i, 'ground', f), 'swr'):5.2f} {g((i, 'ground', f), 'g_0_5'):+4.1f}/{g((i, 'ground', f), 'g_5_15'):+4.1f}".rjust(16) for f in centres))
        print("             (on tripod = over ground at the feed height, whip at CB length, tripod + coax;"
              " gain = avg dBi at 0.5-5 / 5-15 deg)")
        hfn = r["raw"]["hf"]
        line = "  HF        " + " | ".join(f"{b}: {hfn[b]['swr']:.2f}" for b in HF_BANDS)
        for cs in ("C0.5", "C2.0"):
            line += f"\n  HF coil {cs[1:]}pF " + " | ".join(f"{b}: {g((i, cs, b), 'swr'):.2f} (whip {g((i, cs, b), 'whip'):.2f})" for b in HF_BANDS)
        print(line)

def parse_set(items):
    d = {}
    for it in items or []:
        k, v = it.split("=", 1)
        if k not in PARAMS:
            raise SystemExit(f"unknown parameter {k}; choose from {list(PARAMS)}")
        d[k] = float(v)
    return d

def cmd_check(a, paths):
    d = norm(dict(SEEDS[1], **parse_set(a.set)))
    print("design:", {k: d[k] for k in PARAMS})
    m = misfit(d)
    if m:
        print(f"not buildable (hub limits exceeded x{m:.2f})"); return
    eng = Engine(a.workers)
    (sc, raw), = eng.evaluate([d], a.stage, 0, {}, log=False)
    eng.close()
    s, parts = score_raw(raw, a.stage)
    sm = summary(raw, a.stage)
    print(f"score {s:.4f}   " + "  ".join(f"{k} {v:+.3f}" for k, v in parts.items()))
    print("  " + "  ".join(f"{b} {sm[b]:.2f}" for b in CSV_BANDS if b in sm) + f"  RX worst {sm.get('RXworst', 0):.2f}")
    print(dims_text(d, raw))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "pause", "resume", "stop", "throttle", "status", "report", "verify", "check"])
    ap.add_argument("value", nargs="?", default=None, help="for throttle: number of cores (0 = all)")
    ap.add_argument("--dir", default=None)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--set", action="append", help="check: name=value")
    ap.add_argument("--stage", type=int, default=2, help="check: score as stage 1 or 2")
    a = ap.parse_args()
    if a.quick:
        apply_quick()
    paths = Paths(a.dir or ("results_quick" if a.quick else "results_final"))
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
    elif a.cmd == "report":
        cmd_report(a, paths)
    elif a.cmd == "verify":
        if a.top == 15:
            a.top = 3
        cmd_verify(a, paths)
    elif a.cmd == "check":
        cmd_check(a, paths)

if __name__ == "__main__":
    main()
