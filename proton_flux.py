#!/usr/bin/env python3
"""
proton_flux.py
==============
ชั้นข้อมูลของ integral proton flux — ทำให้คลัง GOES_proton_flux_integral/ ต่อเนื่อง
ตั้งแต่ 1998 ถึงปัจจุบัน

ปัญหาที่โมดูลนี้แก้
-------------------
particle/{G08..G12,Primary,Secondary}/ เป็นไฟล์ข้อความ SWPC 5 นาที ที่ให้ integral
proton flux มาแล้ว (P>1, P>5, P>10, P>30, P>50, P>100 MeV) แต่ SWPC เลิกผลิตไฟล์ชุดนี้
เมื่อ 2020-03-09 ตอนเปลี่ยนไปใช้ GOES-R เป็นดาวเทียมหลัก

ข้อมูลที่มาแทนคือ GOES16/ กับ GOES18/ ซึ่งเป็น netCDF ของ SGPS (Solar and Galactic
Proton Sensor) และ *ไม่มีช่อง integral ที่ต้องการเลย* มีแต่

    * flux เชิงอนุพันธ์ 13 ช่อง ครอบ 1.02 - 390 MeV  หน่วย protons/(cm² sr keV s)
    * integral ช่องเดียวที่ > 500 MeV

จึงต้องประกอบสเปกตรัมแล้วอินทิเกรตเอง ดู integral_above()

คำเตือนสำคัญ
------------
ตัวเลข P>1 .. P>100 ที่โมดูลนี้เขียนออกมาเป็น **ค่าที่คำนวณจากสเปกตรัมอนุพันธ์**
ไม่ใช่ผลิตภัณฑ์ที่ NOAA เผยแพร่ ค่าปลายพลังงานสูง (>50, >100) เสถียรมาก แต่ >1 กับ >5
ขึ้นกับวิธีอินทิเกรตค่อนข้างแรง เพราะสเปกตรัมชันและช่องพลังงานมีช่องว่างระหว่างแถบ

Usage
-----
  python proton_flux.py --selftest
  python proton_flux.py --convert                  # netCDF -> ข้อความรูปแบบ Primary
  python proton_flux.py --convert --sat GOES18 --workers 8
  python proton_flux.py --build-series             # cache รายชั่วโมงสำหรับหน้าเว็บ
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import glob
import os
import re
import sys
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

ROOT = "GOES_proton_flux_integral"
PARTICLE = os.path.join(ROOT, "particle")
CACHE_DIR = "data_cache"
HOURLY_CACHE = os.path.join(CACHE_DIR, "proton_hourly.npz")

MISSING = -1.00e+05                    # ค่าที่ SWPC ใช้แทนข้อมูลหาย

# ช่อง integral ของรูปแบบ SWPC (หน่วย MeV)
PROTON_THRESHOLDS = (1, 5, 10, 30, 50, 100)
PCOLS = [f"p{t}" for t in PROTON_THRESHOLDS]
ECOLS = ["e1", "e2", "e3"]
ALLCOLS = PCOLS + ECOLS

# SGPS มีช่อง integral ที่ >500 MeV ให้มาตรง ๆ ใช้เป็นหางของการอินทิเกรต
E_INT = 500.0

# ตำแหน่งวงโคจร (ใช้เขียนใน header ให้เหมือนไฟล์เดิม)
SAT_LOCATION = {"GOES16": "W075", "GOES18": "W137"}
SAT_TAG = {"GOES16": "G16", "GOES18": "G18"}

# ลำดับความน่าเชื่อถือของ product เมื่อวันเดียวกันมีหลายไฟล์
PRODUCT_RANK = {"sci_sgps-l2-avg5m": 3, "sci_sgps-l2-avg1m": 2, "dn_sgps-l2-avg1m": 1}


# --------------------------------------------------------------------------- #
# 1. อ่านรูปแบบข้อความ SWPC 5 นาที
# --------------------------------------------------------------------------- #
def read_swpc_5m(path: str) -> pd.DataFrame:
    """อ่านไฟล์ *_part_5m.txt -> DataFrame (time, p1..p100, e1..e3)

    ใช้ได้กับทุกดาวเทียมในคลัง — ห้ามยึดป้ายคอลัมน์อิเล็กตรอนเป็นหลัก เพราะไฟล์เก่า
    เขียน "E>0.6" ส่วน Primary เขียน "E>0.8" แต่ตำแหน่งคอลัมน์เหมือนกันหมด
    """
    rows = []
    with open(path, "r", encoding="latin-1") as f:
        for ln in f:
            if not ln.strip() or ln[0] in ":#":
                continue
            parts = ln.split()
            if len(parts) < 15 or not parts[0].isdigit():
                continue
            try:
                y, mo, d = int(parts[0]), int(parts[1]), int(parts[2])
                hhmm = parts[3]
                t = datetime(y, mo, d, int(hhmm[:2]), int(hhmm[2:]))
                vals = [float(v) for v in parts[6:15]]
            except (ValueError, IndexError):
                continue
            rows.append([t] + vals)

    df = pd.DataFrame(rows, columns=["time"] + ALLCOLS)
    if not df.empty:
        df[ALLCOLS] = df[ALLCOLS].where(df[ALLCOLS] > MISSING / 2, np.nan)
    return df


def write_swpc_5m(df: pd.DataFrame, path: str, satellite: str,
                  location: str, derived: bool = True) -> None:
    """เขียน DataFrame กลับเป็นรูปแบบข้อความเดียวกับ particle/Primary/"""
    day = pd.to_datetime(df["time"].iloc[0]).date()
    tag = SAT_TAG.get(satellite, satellite)
    name = f"{day:%Y%m%d}_{tag}part_5m.txt"
    sat_label = satellite.replace("GOES", "GOES-")

    head = [
        f":Data_list: {name}",
        f":Created: {datetime.now(timezone.utc):%Y %b %d %H%M} UTC",
        "# Prepared from NOAA/NCEI GOES-R SGPS L2 files by proton_flux.py",
        "# Label: P > 1 = Particles at >1 Mev",
        "# Label: P > 5 = Particles at >5 Mev",
        "# Label: P >10 = Particles at >10 Mev",
        "# Label: P >30 = Particles at >30 Mev",
        "# Label: P >50 = Particles at >50 Mev",
        "# Label: P>100 = Particles at >100 Mev",
        "# Label: E>0.8 = Electrons at >0.8 Mev",
        "# Label: E>2.0 = Electrons at >2.0 Mev",
        "# Label: E>4.0 = Electrons at >4.0 Mev",
        "# Units: Particles = Protons/cm2-s-sr",
        "# Units: Electrons = Electrons/cm2-s-sr",
        f"# Source: {sat_label}",
        f"# Location: {location}",
        f"# Missing data: {MISSING:.2e}",
        "#",
    ]
    if derived:
        head += [
            "# NOTE: integral proton fluxes below were COMPUTED from the SGPS",
            "#       differential spectrum (piecewise power-law integration),",
            "#       they are not a NOAA-published integral product. The >100 MeV",
            "#       end is robust; >1 and >5 MeV depend on the method used.",
            "# NOTE: SGPS measures protons only - electron columns are all missing.",
            "#",
        ]
    head += [
        f"#                      5-minute  {sat_label} Solar Particle and Electron Flux",
        "#",
        "#                 Modified Seconds",
        "# UTC Date  Time   Julian  of the",
        "# YR MO DA  HHMM    Day     Day     P > 1     P > 5     P >10     P >30"
        "     P >50     P>100     E>0.8     E>2.0     E>4.0",
        "#" + "-" * 121,
    ]

    # ห้ามใช้ df.iterrows() ตรงนี้ — แถวมีทั้ง datetime และ float ปนกัน pandas จึง
    # ยุบแถวเป็น object แล้วอนุมานชนิดใหม่ ทำให้ NaN ในคอลัมน์ตัวเลขกลายเป็น NaT
    # (โผล่เฉพาะวันที่มีข้อมูลขาด) แยก array ออกมาเลยทั้งเร็วกว่าและชนิดไม่เพี้ยน
    times = pd.to_datetime(df["time"]).to_numpy()
    values = df[ALLCOLS].to_numpy(dtype=float)

    lines = []
    for tv, row in zip(times, values):
        t = pd.Timestamp(tv)
        mjd = int(_mjd(t.date()))
        sod = t.hour * 3600 + t.minute * 60
        vals = "".join(f"{_fmt(v):>10s}" for v in row)
        lines.append(f"{t:%Y %m %d}  {t:%H%M}   {mjd:5d} {sod:6d} {vals}")

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="latin-1", newline="\n") as f:
        f.write("\n".join(head + lines) + "\n")


def _fmt(v: float) -> str:
    return f"{MISSING:.2e}" if not np.isfinite(v) else f"{v:.2e}"


def _mjd(d: date) -> int:
    """Modified Julian Day — สูตรเดียวกับที่ไฟล์ต้นฉบับใช้"""
    return d.toordinal() - date(1858, 11, 17).toordinal()


# --------------------------------------------------------------------------- #
# 2. อ่าน SGPS L2 netCDF
# --------------------------------------------------------------------------- #
def read_sgps_l2(path: str) -> dict | None:
    """netCDF ของ SGPS L2 -> dict ที่ normalize หน่วยและชื่อ dim แล้ว

    product ทั้งสามแบบ (dn_avg1m / sci_avg1m / sci_avg5m ทุกเวอร์ชัน) ใช้ตัวแปรชุด
    เดียวกัน ต่างแค่ dn_* ตั้งชื่อ dim เวลาเป็น record_number แทน time

    หน่วยในไฟล์เป็น keV และ per-keV แต่รูปแบบ SWPC คิดเป็น MeV จึงแปลงให้ตรงกันตรงนี้
    ที่เดียว โค้ดที่เหลือคิดเป็น MeV ล้วน
    """
    import xarray as xr

    with xr.open_dataset(path, decode_timedelta=False) as ds:
        if "AvgDiffProtonFlux" not in ds:
            return None
        tdim = "time" if "time" in ds.dims else "record_number"

        if "time" in ds.variables:
            t = pd.to_datetime(ds["time"].values)
        else:                                   # dn_* เก็บเวลาไว้คนละที่
            base = pd.Timestamp(re.search(r"_d(\d{8})_", path).group(1))
            n = ds.sizes[tdim]
            t = base + pd.to_timedelta(np.arange(n) * (1440 // n), unit="m")

        diff = ds["AvgDiffProtonFlux"].values * 1000.0        # /keV -> /MeV
        e_eff = ds["DiffProtonEffectiveEnergy"].values / 1000.0
        e_lo = ds["DiffProtonLowerEnergy"].values / 1000.0
        i500 = ds["AvgIntProtonFlux"].values
    return {"time": t, "diff": diff, "e_eff": e_eff, "e_lo": e_lo, "int500": i500}


# --------------------------------------------------------------------------- #
# 3. differential -> integral   (หัวใจของโมดูล)
# --------------------------------------------------------------------------- #
def integral_above(e_eff: np.ndarray, diff: np.ndarray,
                   thresholds=PROTON_THRESHOLDS,
                   tail: np.ndarray | float = 0.0) -> np.ndarray:
    """อินทิเกรตสเปกตรัมอนุพันธ์ตั้งแต่ threshold ขึ้นไป

    e_eff : (nch,)        พลังงานยังผลของแต่ละช่อง [MeV] เรียงจากน้อยไปมาก
    diff  : (nt, nch)     dJ/dE [protons/(cm² sr MeV s)]
    tail  : (nt,) หรือ scalar   flux integral ที่ > E_INT ซึ่งเซนเซอร์ให้มาตรง ๆ
    คืนค่า (nt, len(thresholds)) หน่วย protons/(cm² sr s)

    วิธี: ต่อจุด (E_k, J_k) ด้วย power law ทีละช่วงใน log-log
        J(E) = J_k (E/E_k)^(-γ)   โดย γ = ln(J_k/J_k+1) / ln(E_k+1/E_k)
    แล้วอินทิเกรตเชิงวิเคราะห์ ∫_a^b J dE — ไม่ใช้ trapezoid เพราะสเปกตรัมโปรตอน
    เป็น power law โดยธรรมชาติ การประมาณเชิงเส้นบนช่วงที่กว้างหลายเท่าตัวจะเกินจริงมาก

    ต้องต่อ "หัว" กับ "หาง" ให้ครบด้วย ไม่งั้นพลังงานหายไปเงียบ ๆ:
      หัว  threshold ต่ำกว่าจุดแรกของสเปกตรัม (>1 MeV แต่ e_eff[0] = 1.377 MeV)
           -> ยืด power law ของช่วงแรกลงมาถึง threshold  มิฉะนั้น F(>1) ต่ำกว่าจริง
           ราว 3 เท่า เพราะสเปกตรัมชันมากตรงปลายพลังงานต่ำ
      หาง  ช่องอนุพันธ์สุดท้ายจบที่ ~323 MeV แต่ช่อง integral เริ่มที่ 500 MeV
           -> ยืด power law ของช่วงสุดท้ายไปถึง E_INT แล้วบวกช่อง integral
    """
    e_eff = np.asarray(e_eff, dtype=float)
    diff = np.atleast_2d(np.asarray(diff, dtype=float))
    nt = diff.shape[0]
    tail = np.broadcast_to(np.asarray(tail, dtype=float), (nt,))

    out = np.full((nt, len(thresholds)), np.nan)
    valid = np.isfinite(diff) & (diff > 0)

    # จุดที่ต่อ power law ได้ต้องมีทั้งสองปลาย ประมวลผลทีละคู่ช่องแบบ vector ตามเวลา
    for j, eth in enumerate(thresholds):
        total = np.zeros(nt)
        seen = np.zeros(nt, dtype=bool)

        # หัว: threshold อยู่ต่ำกว่าจุดแรกของสเปกตรัม -> ยืดช่วงแรกลงมา
        if eth < e_eff[0]:
            a, b = e_eff[0], e_eff[1]
            ok = valid[:, 0] & valid[:, 1]
            if ok.any():
                j1, j2 = diff[ok, 0], diff[ok, 1]
                g = -np.log(j2 / j1) / np.log(b / a)
                c = j1 * a ** g
                near1 = np.abs(g - 1.0) < 1e-9
                total[ok] += np.where(near1,
                                      c * np.log(a / eth),
                                      c * (a ** (1 - g) - eth ** (1 - g))
                                      / np.where(near1, 1.0, 1 - g))
                seen |= ok

        for k in range(len(e_eff) - 1):
            a, b = e_eff[k], e_eff[k + 1]
            if b <= eth:
                continue
            lo = max(a, eth)
            ok = valid[:, k] & valid[:, k + 1]
            if not ok.any():
                continue
            j1, j2 = diff[ok, k], diff[ok, k + 1]
            g = -np.log(j2 / j1) / np.log(b / a)
            c = j1 * a ** g
            near1 = np.abs(g - 1.0) < 1e-9
            seg = np.where(near1,
                           c * np.log(b / lo),
                           c * (b ** (1 - g) - lo ** (1 - g)) / np.where(near1, 1.0, 1 - g))
            total[ok] += seg
            seen |= ok

        # หาง: ต่อ power law ของช่วงสุดท้ายไปถึง E_INT แล้วบวกช่อง integral ของเซนเซอร์
        a, b = e_eff[-2], e_eff[-1]
        ok = valid[:, -2] & valid[:, -1] & (E_INT > b)
        if ok.any():
            j1, j2 = diff[ok, -2], diff[ok, -1]
            g = -np.log(j2 / j1) / np.log(b / a)
            c = j1 * a ** g
            lo = max(b, eth)
            near1 = np.abs(g - 1.0) < 1e-9
            total[ok] += np.where(near1,
                                  c * np.log(E_INT / lo),
                                  c * (E_INT ** (1 - g) - lo ** (1 - g))
                                  / np.where(near1, 1.0, 1 - g))
        total += np.where(np.isfinite(tail), tail, 0.0)
        out[:, j] = np.where(seen, total, np.nan)

    return out


def sgps_to_integral(rec: dict) -> pd.DataFrame:
    """SGPS record -> DataFrame integral flux (เฉลี่ยสองเซนเซอร์แล้ว)

    เฉลี่ย *หลัง* อินทิเกรตของแต่ละเซนเซอร์ ไม่ใช่เฉลี่ยสเปกตรัมก่อน เพราะการอินทิเกรต
    power law ไม่เป็นเชิงเส้น เฉลี่ยก่อนจะได้คนละค่า
    """
    diff, e_eff, i500 = rec["diff"], rec["e_eff"], rec["int500"]
    per_sensor = []
    for s in range(diff.shape[1]):
        per_sensor.append(integral_above(e_eff[s], diff[:, s, :], tail=i500[:, s]))
    stack = np.stack(per_sensor)                       # (nsensor, nt, nthr)

    with np.errstate(invalid="ignore"):
        mean = np.nanmean(stack, axis=0)
    mean[np.all(~np.isfinite(stack), axis=0)] = np.nan

    df = pd.DataFrame(mean, columns=PCOLS)
    df.insert(0, "time", rec["time"])
    for c in ECOLS:                                    # SGPS ไม่มีอิเล็กตรอน
        df[c] = np.nan
    return df


def enforce_monotonic(df: pd.DataFrame) -> pd.DataFrame:
    """F(>1) >= F(>5) >= ... ต้องจริงเสมอตามนิยามของ integral flux

    ความไม่เป็นโมโนโทนเกิดได้จาก noise ในช่องที่นับได้น้อย ตรงนี้บีบจากพลังงานสูงลงต่ำ
    ให้สอดคล้องนิยาม แทนที่จะปล่อยค่าที่ขัดแย้งในตัวเองออกไป
    """
    d = df.copy()
    vals = d[PCOLS].to_numpy(float, copy=True)   # pandas 3 คืน view แบบอ่านอย่างเดียว
    for k in range(len(PCOLS) - 2, -1, -1):
        hi = vals[:, k + 1]
        both = np.isfinite(vals[:, k]) & np.isfinite(hi)
        vals[both, k] = np.maximum(vals[both, k], hi[both])
    d[PCOLS] = vals
    return d


# --------------------------------------------------------------------------- #
# 4. batch convert
# --------------------------------------------------------------------------- #
def _grid_5min(df: pd.DataFrame, day: date) -> pd.DataFrame:
    """บังคับลงกริด 5 นาที 288 แถวของวันนั้น (avg1m ต้องยุบลงมา)"""
    idx = pd.date_range(pd.Timestamp(day), periods=288, freq="5min")
    d = df.set_index(pd.to_datetime(df["time"])).drop(columns="time")
    d = d.resample("5min").mean().reindex(idx)
    d.index.name = "time"
    return d.reset_index()


def discover_nc(sat: str) -> dict[str, str]:
    """{YYYYMMDD: path} เลือกไฟล์ที่ดีที่สุดของแต่ละวัน

    คลังนี้ถูก mirror มา จึงมี index.html ปนอยู่ 258 ไฟล์ และวันเดียวกันมีได้หลาย
    product/เวอร์ชัน — ยึด glob *.nc เท่านั้น แล้วจัดอันดับตาม PRODUCT_RANK + เวอร์ชัน

    product ที่ไม่อยู่ใน PRODUCT_RANK ถูกตัดทิ้งตั้งแต่ตรงนี้ โดยเฉพาะ
    ops_seis-l1b-sgps ซึ่งเป็นข้อมูลรายวินาทีคนละโครงสร้าง (T1/T2/T3 แยกช่อง ไม่มี
    AvgDiffProtonFlux) ถ้าปล่อยผ่านมาจะกลายเป็นวันที่ "แปลงไม่ได้" แบบเงียบ ๆ
    """
    best: dict[str, tuple] = {}
    for p in glob.glob(os.path.join(ROOT, sat, "*", "*", "*.nc")):
        m = re.match(r"(.+?)_g\d+_d(\d{8})_v([\d-]+)\.nc$", os.path.basename(p))
        if not m:
            continue
        prod, day, ver = m.group(1), m.group(2), m.group(3)
        if prod not in PRODUCT_RANK:
            continue
        key = (PRODUCT_RANK[prod], tuple(int(x) for x in ver.split("-")))
        if day not in best or key > best[day][0]:
            best[day] = (key, p)
    return {d: v[1] for d, v in sorted(best.items())}


def convert_day(args: tuple[str, str, str, bool]) -> tuple[str, str]:
    sat, day, src, overwrite = args
    out = os.path.join(PARTICLE, sat, day[:4], f"{day}_{SAT_TAG[sat]}part_5m.txt")
    if os.path.exists(out) and not overwrite:
        return day, "skip"
    try:
        rec = read_sgps_l2(src)
        if rec is None:
            return day, "no-vars"
        df = enforce_monotonic(sgps_to_integral(rec))
        df = _grid_5min(df, datetime.strptime(day, "%Y%m%d").date())
        if df[PCOLS].notna().to_numpy().sum() == 0:
            return day, "empty"
        write_swpc_5m(df, out, sat, SAT_LOCATION[sat])
        return day, "ok"
    except Exception as e:                                   # noqa: BLE001
        return day, f"error: {type(e).__name__}: {e}"


def convert(sats=("GOES16", "GOES18"), workers: int = 6,
            overwrite: bool = False) -> None:
    for sat in sats:
        days = discover_nc(sat)
        print(f"\n=== {sat}: {len(days)} วัน ===")
        jobs = [(sat, d, p, overwrite) for d, p in days.items()]
        tally: dict[str, int] = {}
        with cf.ProcessPoolExecutor(max_workers=workers) as ex:
            for i, (day, status) in enumerate(ex.map(convert_day, jobs, chunksize=4), 1):
                key = status if status in ("ok", "skip", "empty", "no-vars") else "error"
                tally[key] = tally.get(key, 0) + 1
                if key == "error" and tally[key] <= 3:
                    print(f"  ! {day}: {status}")
                if i % 250 == 0:
                    print(f"  ... {i}/{len(jobs)}  {tally}")
        print(f"  {sat} เสร็จ: {tally}")


# --------------------------------------------------------------------------- #
# 5. อนุกรมรายชั่วโมงสำหรับฝังในหน้าเว็บ
# --------------------------------------------------------------------------- #
# เฉพาะสองช่องนี้ที่ฝังลงหน้าเว็บ — ไฟล์ข้อความยังเขียนครบทั้งหกช่องตามรูปแบบ
# Primary เหมือนเดิม ตรงนี้แค่เลือกว่าจะพล็อตอะไร
SERIES_CHANNELS = ["p1", "p10", "p50", "p100"]
LOG_LO, LOG_HI = -3.0, 5.0          # ช่วง log10(flux) ที่ quantize ลง uint8

# ลำดับความน่าเชื่อถือ: Primary เป็นผลิตภัณฑ์ทางการของ SWPC จึงมาก่อน GOES-R ที่เรา
# คำนวณเอง ส่วน G10/G11/G12 ใช้เติมยุคก่อน 2010
SOURCE_ORDER = ["Primary", "GOES16", "GOES18", "G11", "G10", "G12", "G08", "G09"]


def _quantize(v: np.ndarray, lo: float = LOG_LO, hi: float = LOG_HI) -> np.ndarray:
    """log10(flux) -> uint8 (0 = ไม่มีข้อมูล, 1..255 เชิงเส้นในช่วง [lo, hi])

    แยกออกจาก build_hourly_series เพื่อให้ selftest ประกอบอนุกรมสังเคราะห์ด้วยสูตร
    เดียวกันเป๊ะ ๆ แทนที่จะเขียนสูตร quantize ซ้ำสองที่แล้วเสี่ยงเพี้ยนไม่ตรงกัน
    """
    v = np.asarray(v, dtype=float)
    ok = np.isfinite(v) & (v > 0)
    lg = np.clip(np.log10(np.where(ok, v, 1.0)), lo, hi)
    q = 1 + np.round((lg - lo) / (hi - lo) * 254).astype(int)
    return np.where(ok, np.clip(q, 1, 255), 0).astype(np.uint8)


def _iter_source_files(sub: str):
    yield from sorted(glob.glob(os.path.join(PARTICLE, sub, "*", "*_5m.txt")))


def build_hourly_series(quiet: bool = False) -> dict:
    """รวมทุกแหล่ง -> อนุกรมรายชั่วโมงช่องเดียวกันหมด แล้ว quantize

    ย่อด้วย **ค่าสูงสุดในชั่วโมง** ไม่ใช่ค่าเฉลี่ย — ยอด SEP คือสิ่งที่ต้องเห็น
    ค่าเฉลี่ยจะกลบพีคแคบ ๆ ที่กินเวลาไม่กี่สิบนาที

    combine_first ทำให้แหล่งที่มาก่อนใน SOURCE_ORDER ชนะเสมอ และแหล่งถัดไปเข้ามา
    เติมเฉพาะชั่วโมงที่ยังว่าง — Primary เป็นผลิตภัณฑ์ทางการของ SWPC จึงมาก่อน
    GOES16/18 ที่เราคำนวณเอง
    """
    # ยุบเป็นรายชั่วโมง *ทีละแหล่ง* แล้วค่อยซ้อนกันตามลำดับความน่าเชื่อถือ
    # ถ้าเอา 4.7 ล้านแถวจากทุกแหล่งมากองรวมกันก่อนจะกินหน่วยความจำหลายร้อย MB
    # โดยไม่ได้อะไรเพิ่ม เพราะสุดท้ายก็ยุบเหลือ ~246,000 ชั่วโมงอยู่ดี
    hourly = None
    for sub in SOURCE_ORDER:
        files = list(_iter_source_files(sub))
        if not files:
            continue
        got = [df for df in (read_swpc_5m(p) for p in files) if not df.empty]
        if not got:
            continue
        d = pd.concat(got, ignore_index=True)
        d["hour"] = pd.to_datetime(d["time"]).dt.floor("h")
        h = d.groupby("hour")[SERIES_CHANNELS].max()
        hourly = h if hourly is None else hourly.combine_first(h)
        if not quiet:
            print(f"  {sub:9s} {len(files):5d} ไฟล์  {len(h):7d} ชั่วโมง  "
                  f"{h.index.min():%Y-%m-%d} .. {h.index.max():%Y-%m-%d}")

    if hourly is None:
        raise RuntimeError("ไม่พบไฟล์ particle เลย — รัน --convert ก่อนหรือตรวจ path")

    t0 = hourly.index.min().floor("D")
    t1 = hourly.index.max().ceil("D")
    grid = pd.date_range(t0, t1, freq="h")
    hourly = hourly.reindex(grid)

    # quantize: 0 = ไม่มีข้อมูล, 1..255 = log10(flux) เชิงเส้นในช่วง [LOG_LO, LOG_HI]
    enc = np.zeros((len(SERIES_CHANNELS), len(grid)), dtype=np.uint8)
    for i, c in enumerate(SERIES_CHANNELS):
        enc[i] = _quantize(hourly[c].to_numpy(float))

    if not quiet:
        cov = (enc[0] > 0).mean()
        print(f"\n  รวม {len(grid):,} ชั่วโมง  {t0:%Y-%m-%d} .. {t1:%Y-%m-%d}  "
              f"มีข้อมูล {100*cov:.1f}%  ({enc.nbytes/1e6:.2f} MB ก่อน base64)")

    return {"t0": t0.strftime("%Y-%m-%dT%H:%M:%SZ"), "step_min": 60,
            "n": len(grid), "channels": SERIES_CHANNELS,
            "log_lo": LOG_LO, "log_hi": LOG_HI, "enc": enc}


def save_hourly(series: dict, path: str = HOURLY_CACHE) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(
        path, enc=series["enc"], t0=series["t0"], step_min=series["step_min"],
        n=series["n"], channels=np.array(series["channels"]),
        log_lo=series["log_lo"], log_hi=series["log_hi"])
    return path


def load_hourly(path: str = HOURLY_CACHE) -> dict | None:
    if not os.path.exists(path):
        return None
    z = np.load(path, allow_pickle=False)
    return {"t0": str(z["t0"]), "step_min": int(z["step_min"]), "n": int(z["n"]),
            "channels": [str(c) for c in z["channels"]],
            "log_lo": float(z["log_lo"]), "log_hi": float(z["log_hi"]),
            "enc": z["enc"]}


def decode_hourly(series: dict, channel: str | int = "p10") -> pd.Series:
    """ถอดรหัส uint8 กลับเป็น flux จริง [pfu] -> pd.Series index เป็นเวลา

    ตรรกะเดียวกับ decodeFlux() ใน JS ของ interactive_map.py — มีไว้ที่เดียวใน
    ฝั่ง Python เพื่อไม่ให้ plot_overview/find_top_events ต้องเขียนสูตร quantize
    ซ้ำเอง (เคยมีปัญหาแก้ที่หนึ่งแล้วอีกที่หนึ่งเพี้ยนมาแล้วกับสีธีมใน JS)

    channel รับได้ทั้งชื่อ ("p10") และ index — ใช้ชื่อดีกว่าเพราะลำดับช่องใน cache
    เปลี่ยนได้ (ตอนนี้คือ p1,p10,p50,p100) การ hardcode index จะพังเงียบ ๆ ถ้าลำดับ
    เปลี่ยน
    """
    if isinstance(channel, str):
        channel = series["channels"].index(channel)
    t0 = pd.Timestamp(series["t0"].replace("Z", ""))
    idx = pd.date_range(t0, periods=series["n"], freq=f'{series["step_min"]}min')
    q = series["enc"][channel].astype(float)
    lo, hi = series["log_lo"], series["log_hi"]
    v = np.where(q > 0, 10 ** (lo + (q - 1) / 254 * (hi - lo)), np.nan)
    return pd.Series(v, index=idx)


# --------------------------------------------------------------------------- #
# 6. เกณฑ์ flare + สัญญาณ CME/SEP
# --------------------------------------------------------------------------- #
# นี่คือเกณฑ์ที่อนุมานจาก proton flux ไม่ใช่การตรวจจับ CME จากภาพ coronagraph —
# คลังนี้ไม่มีข้อมูลภาพ จึงใช้สัญญาณ SEP ที่ตามหลัง flare เป็นตัวแทน
#
# หน้าต่างพีคตั้งใจให้เท่ากับ CHART_AFTER_H ใน interactive_map.py (กราฟตอน hover
# แสดง +60 ชม.เหมือนกัน) เพื่อให้ป้าย "มี CME" ตรงกับสิ่งที่ผู้ใช้เห็นตอนชี้เมาส์ที่
# จุดนั้นพอดี — ถ้าแก้ค่าใดค่าหนึ่งต้องแก้อีกฝั่งให้ตรงกันด้วย
CME_P10_PFU = 1.5             # >10 MeV ต้องพีคถึงระดับนี้ในหน้าต่างหลัง flare (เกณฑ์สัมบูรณ์)
CME_BG_DAYS = 30               # จำนวนวันย้อนหลังที่ใช้คำนวณค่าเฉลี่ยพื้นฐานของ >100 MeV
CME_PEAK_WINDOW_H = 60         # ชั่วโมงหลัง flare ที่มองหาพีคทั้งสองช่อง


def flag_cme(times, hourly: dict) -> np.ndarray:
    """คืน bool array ต่อ timestamp — True ถ้า flare ดวงนั้นเข้าเกณฑ์ CME/SEP

    เกณฑ์ต้องผ่านทั้งคู่ (AND):
      1. >10 MeV พีคในหน้าต่าง [t, t+CME_PEAK_WINDOW_H ชม.] >= CME_P10_PFU pfu
      2. >100 MeV พีคในหน้าต่างเดียวกัน สูงกว่าค่าเฉลี่ย >100 MeV ของ CME_BG_DAYS วัน
         ก่อนหน้า t (ไม่รวมชั่วโมงของ t เอง)

    ทำไมต้องมีเกณฑ์ (2) ด้วย: (1) อย่างเดียว false-positive ได้ง่ายช่วงที่กิจกรรม
    สุริยะสูงจน background ของ >10 MeV ลอยเกิน 1.5 pfu อยู่แล้วโดยไม่มีเหตุการณ์ใหม่
    จริง >100 MeV หายากกว่ามาก เกณฑ์สัมพัทธ์กับฐานของมันเองจึงคัดเฉพาะการเพิ่มขึ้นจริง

    ใช้ rolling window vectorize ทั้งอนุกรมครั้งเดียว ไม่ลูปทีละ flare — เร็วกว่ามาก
    เมื่อมี flare หลักหมื่นดวง แล้วค่อย index เข้าตำแหน่งของแต่ละ flare ทีหลัง
    """
    p10 = decode_hourly(hourly, "p10")
    p100 = decode_hourly(hourly, "p100")
    step_min = hourly["step_min"]
    peak_steps = max(1, round(CME_PEAK_WINDOW_H * 60 / step_min))
    bg_steps = max(1, round(CME_BG_DAYS * 24 * 60 / step_min))

    # หน้าต่างมองไปข้างหน้า [i, i+peak_steps]: rolling().max() ปกติมองย้อนหลัง จึง
    # เลื่อน (shift) ผลลัพธ์ขึ้นมา peak_steps แถวให้กลายเป็นมองไปข้างหน้าแทน
    p10_fwd = p10.rolling(peak_steps + 1, min_periods=1).max().shift(-peak_steps).to_numpy()
    p100_fwd = p100.rolling(peak_steps + 1, min_periods=1).max().shift(-peak_steps).to_numpy()
    # ฐานย้อนหลัง [i-bg_steps, i-1]: rolling มองย้อนหลังอยู่แล้ว เลื่อนลง 1 แถวเพื่อ
    # ไม่ให้ชั่วโมงของ i เองปนเข้าไปในค่าเฉลี่ยฐาน
    p100_bg = p100.rolling(bg_steps, min_periods=1).mean().shift(1).to_numpy()

    t0 = p10.index[0]
    n = len(p10)
    delta_min = (pd.to_datetime(pd.Series(times)) - t0) / pd.Timedelta(minutes=1)
    pos = np.floor(delta_min.to_numpy() / step_min)
    pos = np.where(np.isfinite(pos), pos, -1).astype("int64")     # NaT -> นอกช่วง

    out = np.zeros(len(pos), dtype=bool)
    ok = (pos >= 0) & (pos < n)
    p = pos[ok]
    a, b, c = p10_fwd[p], p100_fwd[p], p100_bg[p]
    good = np.isfinite(a) & np.isfinite(b) & np.isfinite(c)
    res = np.zeros(len(p), dtype=bool)
    res[good] = (a[good] >= CME_P10_PFU) & (b[good] > c[good])
    out[ok] = res
    return out


# --------------------------------------------------------------------------- #
# selftest
# --------------------------------------------------------------------------- #
def selftest() -> int:
    # ---- integral_above บน power law ที่รู้คำตอบเชิงวิเคราะห์ ---------------- #
    # J(E) = A E^-g  =>  ∫_Eth^inf = A/(g-1) * Eth^(1-g)
    A, g = 1000.0, 3.0
    E = np.geomspace(1.0, 400.0, 13)
    J = A * E ** (-g)
    # หางจริงเหนือ E_INT ป้อนเข้าไปเป็น tail เพื่อให้เทียบกับสูตรอนันต์ได้
    tail = A / (g - 1) * E_INT ** (1 - g)
    got = integral_above(E, J[None, :], thresholds=(1, 10, 100), tail=tail)[0]
    want = np.array([A / (g - 1) * t ** (1 - g) for t in (1, 10, 100)])
    assert np.allclose(got, want, rtol=1e-6), (got, want)

    # ---- โมโนโทนตามนิยาม ---------------------------------------------------- #
    out = integral_above(E, J[None, :], tail=tail)[0]
    assert np.all(np.diff(out) <= 1e-12), out

    # ---- ช่วงที่ threshold ตกกลางช่อง ต้องเริ่มอินทิเกรตที่ threshold -------- #
    part = integral_above(E, J[None, :], thresholds=(2.5,), tail=tail)[0, 0]
    assert abs(part - A / (g - 1) * 2.5 ** (1 - g)) / part < 1e-6, part

    # ---- threshold ต่ำกว่าจุดแรกของสเปกตรัม ต้องยืด power law ลงมา ---------- #
    # เคสจริง: SGPS มีจุดต่ำสุดที่ 1.377 MeV แต่ต้องรายงาน F(>1) ถ้าไม่ต่อหัว
    # ค่าจะขาดไปราว 3 เท่าเพราะสเปกตรัมชันมากตรงนั้น
    Ehi = np.geomspace(1.377, 400.0, 13)
    Jhi = A * Ehi ** (-g)
    head = integral_above(Ehi, Jhi[None, :], thresholds=(1.0,), tail=tail)[0, 0]
    assert abs(head - A / (g - 1) * 1.0 ** (1 - g)) / head < 1e-6, head

    # ---- MJD ตรงกับที่ไฟล์ต้นฉบับเขียนไว้ ----------------------------------- #
    assert _mjd(date(2017, 1, 1)) == 57754, _mjd(date(2017, 1, 1))

    # ---- อ่านไฟล์จริงได้ทั้งรุ่น E>0.6 และ E>0.8 ---------------------------- #
    for sub, label in (("G09", "E>0.6"), ("Primary", "E>0.8")):
        fs = list(_iter_source_files(sub))
        if not fs:
            continue
        df = read_swpc_5m(fs[0])
        assert len(df) > 200, (sub, len(df))
        assert df["p10"].notna().any(), sub
        assert list(df.columns) == ["time"] + ALLCOLS

    # ---- round-trip เขียนแล้วอ่านกลับได้ค่าเดิม ----------------------------- #
    import tempfile
    idx = pd.date_range("2024-05-14", periods=288, freq="5min")
    demo = pd.DataFrame({"time": idx})
    for i, c in enumerate(PCOLS):
        demo[c] = np.geomspace(10 ** (2 - i), 10 ** (1 - i), 288)
    for c in ECOLS:
        demo[c] = np.nan
    # แถวที่ข้อมูลโปรตอนขาดต้องเขียนออกมาเป็น missing ได้ ไม่ใช่ระเบิด — เคยพลาด
    # ตรงนี้เพราะ iterrows() แปลง NaN เป็น NaT เมื่อแถวมีคอลัมน์ datetime ปนอยู่
    demo.loc[15, PCOLS] = np.nan
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "x.txt")
        write_swpc_5m(demo, p, "GOES18", "W137")
        back = read_swpc_5m(p)
        assert len(back) == 288, len(back)
        assert back.loc[15, PCOLS].isna().all(), back.loc[15, PCOLS].tolist()
        assert back["p10"].notna().sum() == 287
        ok = back.index != 15
        assert np.allclose(back.loc[ok, "p10"], demo.loc[ok, "p10"], rtol=1e-2)
        assert back[ECOLS].isna().all().all()          # อิเล็กตรอนต้องเป็น missing
        assert (back["time"].to_numpy() == idx.to_numpy()).all()

    # ---- decode_hourly ต้อง round-trip กับ quantize ของ build_hourly_series -- #
    fake = {"t0": "2024-01-01T00:00:00Z", "step_min": 60, "n": 3,
            "channels": ["p1", "p10"], "log_lo": -3.0, "log_hi": 5.0,
            "enc": np.array([[128, 128, 128], [0, 1, 255]], dtype=np.uint8)}
    s = decode_hourly(fake, "p10")                 # index 1 -> [0, 1, 255]
    assert len(s) == 3 and s.index[0] == pd.Timestamp("2024-01-01")
    assert np.isnan(s.iloc[0])                     # q=0 -> ไม่มีข้อมูล
    assert abs(s.iloc[1] - 10 ** -3.0) < 1e-9       # q=1 -> ขอบล่าง log_lo พอดี
    assert abs(s.iloc[2] - 10 ** 5.0) < 1e-6        # q=255 -> ขอบบน log_hi พอดี
    assert decode_hourly(fake, 1).equals(decode_hourly(fake, "p10"))  # ชื่อ = index

    # ---- enforce_monotonic ต้องดันค่าที่ขัดนิยามขึ้น ------------------------ #
    bad = pd.DataFrame([[1.0, 2.0, 0.5, 0.4, 0.3, 0.2]], columns=PCOLS)
    fixed = enforce_monotonic(bad)
    assert list(fixed.iloc[0]) == [2.0, 2.0, 0.5, 0.4, 0.3, 0.2], list(fixed.iloc[0])

    # ---- flag_cme: ต้องผ่านทั้งเกณฑ์สัมบูรณ์ (>10 MeV) และเกณฑ์สัมพัทธ์ (>100 MeV) - #
    bg_h, peak_h = CME_BG_DAYS * 24, CME_PEAK_WINDOW_H
    n_series = bg_h + peak_h + 40                    # เผื่อขอบทั้งสองด้าน
    onset = bg_h + 10                                 # ฐาน 30 วันเต็มอยู่ก่อนจุดนี้
    t0 = pd.Timestamp("2024-01-01")
    flare_t = [t0 + pd.Timedelta(hours=onset)]

    def _mk_series(p10v, p100v):
        enc = np.stack([_quantize(p10v), _quantize(p100v)])
        return {"t0": t0.strftime("%Y-%m-%dT%H:%M:%SZ"), "step_min": 60,
                "n": len(p10v), "channels": ["p10", "p100"],
                "log_lo": LOG_LO, "log_hi": LOG_HI, "enc": enc}

    base10 = np.full(n_series, 0.05)                  # background เงียบทั้งสองช่อง
    base100 = np.full(n_series, 0.01)

    a10, a100 = base10.copy(), base100.copy()          # A: ผ่านทั้งสองเกณฑ์
    a10[onset], a100[onset] = 5.0, 1.0
    assert flag_cme(flare_t, _mk_series(a10, a100))[0] == True

    b10, b100 = base10.copy(), base100.copy()           # B: >10 MeV ผ่าน แต่ >100 MeV
    b10[onset] = 5.0                                     #    เท่า background เดิม
    assert flag_cme(flare_t, _mk_series(b10, b100))[0] == False

    c10, c100 = base10.copy(), base100.copy()            # C: >100 MeV ขยับ แต่ >10 MeV
    c10[onset], c100[onset] = 0.5, 1.0                    #    ไม่ถึง 1.5 pfu
    assert flag_cme(flare_t, _mk_series(c10, c100))[0] == False

    d = _mk_series(base10, base100)                       # D: เวลานอกช่วงข้อมูล -> False
    assert flag_cme([t0 - pd.Timedelta(days=5)], d)[0] == False
    assert flag_cme([t0 + pd.Timedelta(hours=n_series + 100)], d)[0] == False

    print("selftest: OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[4])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--convert", action="store_true")
    ap.add_argument("--build-series", action="store_true")
    ap.add_argument("--sat", nargs="+", default=["GOES16", "GOES18"],
                    choices=["GOES16", "GOES18"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if args.convert:
        convert(args.sat, workers=args.workers, overwrite=args.overwrite)
    if args.build_series:
        print("=== รวมอนุกรมรายชั่วโมง ===")
        s = build_hourly_series()
        p = save_hourly(s)
        print(f"  -> {p}  ({os.path.getsize(p)/1e6:.2f} MB)")
    if not (args.convert or args.build_series):
        ap.print_help()
    return 0


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
