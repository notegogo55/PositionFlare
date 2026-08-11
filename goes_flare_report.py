#!/usr/bin/env python3
"""
goes_flare_report.py
====================
ชั้นข้อมูล (data layer) ของ catalogue X-class flare — อ่านจากแหล่งเดียวคือ

    NOAA/NCEI  GOES XRS L2 Flare Report (science quality)
    https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/
        goes/multi/l2/data/xrsf-l2-flrpt_science/csv/

ทำไมต้องแหล่งนี้แหล่งเดียว
--------------------------
ไฟล์ชุดนี้ NOAA รวม "GOES XRS flare summary" กับ "SWPC solar event reports"
มาไว้ในตารางเดียว หนึ่งแถวต่อหนึ่ง flare ครอบคลุม 1995-01-03 ถึงปัจจุบัน
จึงได้เปรียบ pipeline เดิม (ที่ต้องเย็บ legacy report + SWPC รายวันเข้าด้วยกัน)
สี่เรื่อง:

1. พิกัดมาพร้อมข้อมูล — ไม่ต้อง join แถว XRA กับ FLA เองอีกแล้ว และมี
   *สองแหล่งอิสระ* ให้เลือก
     * flare_loc_swpc_hgs  จากภาคพื้น (SOON/Culgoora) หรือ SXI/AIA — มีตลอด 1995→now
     * flare_loc_xrs_hgs   จาก XRS บน GOES-16..19 เอง — มีตั้งแต่ 2017
   บวก flare_loc_xrs_hpc ที่เป็น helioprojective arcsec ตรง ๆ ซึ่งยังมีค่าแม้
   flare อยู่ริมขอบจนคำนวณ Stonyhurst ไม่ได้

2. irradiance เป็น science quality — ไม่ถูกคูณ factor 0.7 แบบที่ SWPC
   รายงาน GOES 1-15 ดู SCALE_LEGACY ด้านล่าง เรื่องนี้สำคัญมากเวลาเทียบข้าม
   cycle เพราะ SWPC เลิกคูณ 0.7 ตอนเปลี่ยนไปใช้ GOES-16 เป็นดาวเทียมหลัก
   (~2020) แปลว่า magnitude ของ cycle 23/24 กับ cycle 25 ใน catalogue เก่า
   *อยู่คนละสเกล* — เทียบกันตรง ๆ ไม่ได้

3. peak_saturated เป็น flag จริงจากผู้ผลิตข้อมูล ไม่ต้องเดาจาก integrated flux
   ว่า event ไหนโดนตัดยอด

4. 33 requests (ปีละไฟล์) แทน ~11,000 requests รายวัน

Usage (เป็น library)
--------------------
    import goes_flare_report as gfr
    cat = gfr.build_catalog(scale="science")      # X-flare ทุก cycle
    cat = gfr.build_catalog(scale="legacy")       # สเกลเดิมแบบ GOES 1-15
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import urllib.error
import urllib.request
import warnings
from datetime import date

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# source
# --------------------------------------------------------------------------- #
BASE_URL = ("https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/"
            "goes/multi/l2/data/xrsf-l2-flrpt_science/csv/")
CACHE_DIR = "data_cache"

# sci_xrsf-l2-flrpt_geo_y2024_v1-0-1.csv  — เวอร์ชันในชื่อไฟล์เปลี่ยนได้
# จึงอ่านจาก directory index แทนการ hard-code
YEAR_FILE_RE = re.compile(
    r'href="(sci_xrsf-l2-flrpt_geo_y(\d{4})_v[0-9.\-]+\.csv)"', re.I)


# --------------------------------------------------------------------------- #
# flux scales
# --------------------------------------------------------------------------- #
# SWPC รายงาน GOES 1-15 โดยคูณช่อง 1-8 Å ด้วย 0.7 (ค่าคาลิเบรตยุคเก่า) ตัวเลข
# ที่คุ้นตาในเปเปอร์/ข่าวจึงเป็นสเกลนี้ทั้งหมด เช่น 28 ต.ค. 2003 = "X17.2"
# ส่วนไฟล์ science quality เก็บ irradiance จริง -> event เดียวกันเป็น X24.6
#
#   science : ค่าจริง เป็นสเกลเดียวกันหมดตั้งแต่ 1995 ถึงปัจจุบัน  <- ใช้วิเคราะห์
#   legacy  : science x 0.7 = "GOES 1-15 equivalent" <- ใช้เทียบกับวรรณกรรมเก่า
#
# ทั้งสองสเกลสม่ำเสมอในตัวเอง แต่ "ค่าที่ประกาศจริง ณ เวลานั้น" ไม่ได้ตรงกับสเกล
# ใดสเกลหนึ่งตลอด: ยุค GOES 1-15 ตรงกับ legacy ส่วนหลังปี 2020 (GOES-16 เป็นดวง
# หลัก) SWPC เลิกคูณ 0.7 จึงตรงกับ science แทน — คิดจะแปลงกลับเป็น "ค่าที่ประกาศ"
# อัตโนมัติไม่ได้ เพราะบางเหตุการณ์ค่าที่ประกาศมาจากดาวเทียมคนละดวงกับแถวนี้
# (6 ก.ย. 2017 SWPC ประกาศ X9.3 จาก GOES-15 แต่แถวนี้เป็น GOES-16 = X14.6)
SCALE_LEGACY = 0.7
SCALES = {"science": 1.0, "legacy": SCALE_LEGACY}

CLASS_FLOOR = {"A": 1e-8, "B": 1e-7, "C": 1e-6, "M": 1e-5, "X": 1e-4}
CLASS_ORDER = ["A", "B", "C", "M", "X"]

# --------------------------------------------------------------------------- #
# solar cycles — ใช้ smoothed sunspot minima ของ NOAA/SWPC
# ขอบเขตไม่ทับกัน เพื่อให้ทุก event ถูกนับครั้งเดียว
# --------------------------------------------------------------------------- #
TODAY = date.today()
CYCLES: dict[int, tuple[date, date]] = {
    23: (date(1996, 5, 1), date(2008, 11, 30)),
    24: (date(2008, 12, 1), date(2019, 11, 30)),
    25: (date(2019, 12, 1), TODAY),
}

# --------------------------------------------------------------------------- #
# position sources — เรียงตามลำดับที่เลือกใช้
# --------------------------------------------------------------------------- #
# ทำไม SWPC มาก่อน XRS: SWPC/ภาคพื้นมีตลอดทั้งสาม cycle ส่วน XRS เพิ่งมีปี 2017
# ถ้าให้ XRS มาก่อน cycle 25 จะถูกวัดด้วยเครื่องมือคนละตัวกับ cycle 23/24
# ซึ่งทำให้ bias เชิงระบบปนเข้ามาในการเทียบข้าม cycle พอดี
# ลำดับใน dict นี้คือลำดับที่ resolve_position() ไล่เลือก
POS_TIER_LABEL = {
    "SWPC": "ภาคพื้น / SXI / AIA  (SWPC event report)",
    "XRS": "GOES-16..19 XRS  (Stonyhurst)",
    "XRS-HPC": "GOES-16..19 XRS  (helioprojective, ขอบจาน)",
    "AR": "อนุมานจาก flare ดวงอื่นใน AR เดียวกัน",
}

# tier 4 — ไฟล์เดียวกันนี้มี flare ทุก class ราว 81,000 ดวง ในจำนวนนั้นมีทั้ง
# พิกัดและเลข AR อยู่ ~27,000 ดวง ถ้า X-flare ดวงหนึ่งไม่มีพิกัดแต่รู้ว่ามาจาก
# AR ไหน ก็ยืมพิกัดของ flare ดวงอื่นจาก AR เดียวกันมาหมุนตามการหมุนของดวง
# อาทิตย์ให้ตรงเวลาได้ — ยังคงเป็นข้อมูลจากแหล่งเดิมทั้งหมด ไม่ได้ดึงจากที่อื่น
AR_FILL_WINDOW_DAYS = 1.0     # กว้างกว่านี้ได้ ref เพิ่มแทบไม่กี่ดวง แต่ error โต
AR_FILL_MAX_REFS = 5          # ใช้ ref ที่ใกล้เวลาที่สุดกี่ดวง แล้วเอา median

# NOAA เริ่มใช้เลข active region 5 หลักที่ region 10000 เมื่อ 14 มิ.ย. 2002
# ไฟล์ต้นทางเก็บแค่ 4 หลักท้าย (valid_max = 9999) จึงต้องเติม 10000 คืนเอง
# ไม่งั้น AR 10486 ของ Halloween 2003 จะกลายเป็น "AR486"
AR_ROLLOVER = date(2002, 6, 14)


# --------------------------------------------------------------------------- #
# download + cache
# --------------------------------------------------------------------------- #
def _fetch(url: str, timeout: int = 180) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "solar-research/2.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def discover_year_files(offline: bool = False) -> dict[int, str]:
    """อ่าน directory index -> {ปี: ชื่อไฟล์}

    ถ้า offline (หรือต่อเน็ตไม่ได้) ให้ย้อนกลับไปดูว่ามีอะไรอยู่ใน cache แล้วบ้าง
    """
    if not offline:
        try:
            html = _fetch(BASE_URL, timeout=60).decode("utf-8", "replace")
            found = {int(y): name for name, y in YEAR_FILE_RE.findall(html)}
            if found:
                return dict(sorted(found.items()))
            print("  ! directory index ไม่มีไฟล์รายปี -> ใช้ cache แทน")
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            print(f"  ! เข้าถึง index ไม่ได้ ({e}) -> ใช้ cache แทน")

    cached = {}
    if os.path.isdir(CACHE_DIR):
        for name in sorted(os.listdir(CACHE_DIR)):
            m = YEAR_FILE_RE.search(f'href="{name}"')
            if m:
                cached[int(m.group(2))] = name
    if not cached:
        raise RuntimeError(
            "ไม่มีทั้งการเชื่อมต่อและไฟล์ใน cache — รันครั้งแรกต้องต่อเน็ต")
    return dict(sorted(cached.items()))


def _local_path(name: str) -> str:
    return os.path.join(CACHE_DIR, name)


def ensure_year_file(year: int, name: str, refresh_current: bool = True) -> str:
    """โหลดไฟล์รายปีถ้ายังไม่มีใน cache; ปีปัจจุบันโหลดใหม่เสมอ (ข้อมูลยังเดินอยู่)"""
    path = _local_path(name)
    stale = refresh_current and year >= TODAY.year
    if os.path.exists(path) and not stale:
        return path
    os.makedirs(CACHE_DIR, exist_ok=True)
    data = _fetch(BASE_URL + name)
    with open(path, "wb") as f:
        f.write(data)
    return path


# --------------------------------------------------------------------------- #
# load
# --------------------------------------------------------------------------- #
USECOLS = [
    "time", "start_time", "end_time", "flare_id", "xrsb_irrad", "flare_class",
    "xrsb_irrad_source", "background_irrad", "integrated_irrad_peak",
    "integrated_irrad_end", "flare_loc_swpc_hgs_lon", "flare_loc_swpc_hgs_lat",
    "flare_loc_swpc_source", "flare_loc_xrs_hgs_lon", "flare_loc_xrs_hgs_lat",
    "flare_loc_xrs_hpc_x", "flare_loc_xrs_hpc_y", "flare_loc_xrs_source",
    "sequential_flare_num", "event_id_swpc", "peak_saturated", "active_region",
]


def load_raw(years: list[int] | None = None, offline: bool = False,
             quiet: bool = False) -> pd.DataFrame:
    """โหลดไฟล์รายปีทั้งหมด -> DataFrame ดิบ (ทุก class ตั้งแต่ A ถึง X)"""
    files = discover_year_files(offline=offline)
    if years is not None:
        files = {y: n for y, n in files.items() if y in years}
    if not files:
        raise RuntimeError("ไม่มีไฟล์ตรงกับปีที่ขอ")

    frames = []
    for y, name in files.items():
        path = ensure_year_file(y, name, refresh_current=not offline)
        df = pd.read_csv(path, usecols=lambda c: c in USECOLS, low_memory=False)
        frames.append(df)
        if not quiet:
            print(f"  {y}: {len(df):6d} flares   ({name})")

    raw = pd.concat(frames, ignore_index=True)
    # ไฟล์บางปีมีแถวซ้ำที่ start_time หาย — เก็บแถวที่ข้อมูลครบกว่า
    raw = (raw.assign(_n=raw.notna().sum(axis=1))
              .sort_values("_n", ascending=False)
              .drop_duplicates(subset=["flare_id", "time"], keep="first")
              .drop(columns="_n"))
    return raw.sort_values("time").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# classification
# --------------------------------------------------------------------------- #
def classify(flux: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """irradiance 1-8 Å [W/m2] -> (ตัวอักษร class, magnitude)

    X ไม่มีเพดานบน: 2.48e-3 W/m2 -> ('X', 24.8)
    """
    flux = np.asarray(flux, dtype=float)
    letter = np.full(flux.shape, "", dtype="<U1")
    mag = np.full(flux.shape, np.nan)
    for c in CLASS_ORDER:
        floor = CLASS_FLOOR[c]
        hit = flux >= floor
        letter[hit] = c
        mag[hit] = flux[hit] / floor
    # ต่ำกว่า A1.0 ยังนับเป็น A (magnitude < 1)
    sub = np.isfinite(flux) & (flux < CLASS_FLOOR["A"]) & (flux > 0)
    letter[sub] = "A"
    mag[sub] = flux[sub] / CLASS_FLOOR["A"]
    return letter, mag


def class_label(letter: str, mag: float) -> str:
    """('X', 24.88) -> 'X24.8';  ('C', 9.99) -> 'C9.9'

    ตัดทศนิยมทิ้ง ไม่ปัดขึ้น — ตรงกับที่ NOAA เขียนในคอลัมน์ flare_class เอง
    (15 เม.ย. 2001 flux 2.067e-3 ต้นทางเขียน X20.6 ไม่ใช่ X20.7) และจำเป็นด้วย
    เพราะปัดขึ้นจะได้ "C10.0" ซึ่งไม่มีอยู่จริง — เกิน C9.9 ไปแล้วคือ M1.0
    """
    if not letter or not np.isfinite(mag):
        return "?"
    # ต้อง round ก่อน floor: flux 1.600e-07 หาร 1e-7 ได้ 1.5999999999999999
    # ในเลขทศนิยมฐานสอง ถ้า floor ตรง ๆ จะกลายเป็น B1.5
    return f"{letter}{np.floor(np.round(mag, 6) * 10 + 1e-6) / 10:.1f}"


def full_active_region(ar, day) -> np.ndarray:
    """เลข AR 4 หลักจากไฟล์ -> เลข NOAA เต็ม (486 ที่ปี 2003 -> 10486)"""
    ar = np.asarray(ar, dtype=float)
    after = pd.to_datetime(pd.Series(day)).dt.date.to_numpy() >= AR_ROLLOVER
    return np.where(np.isfinite(ar) & after & (ar < 10000), ar + 10000, ar)


def format_position(lat: float, lon: float) -> str:
    """(-19.0, 83.0) -> 'S19W83' — รูปแบบเดียวกับที่ SWPC พิมพ์"""
    if not (np.isfinite(lat) and np.isfinite(lon)):
        return ""
    return (f"{'N' if lat >= 0 else 'S'}{abs(round(lat)):02.0f}"
            f"{'W' if lon >= 0 else 'E'}{abs(round(lon)):02.0f}")


# --------------------------------------------------------------------------- #
# position resolution
# --------------------------------------------------------------------------- #
def resolve_position(df: pd.DataFrame) -> pd.DataFrame:
    """เลือกพิกัดจากแหล่งที่ดีที่สุดที่มี แล้วบันทึกว่าเอามาจากไหน

    tier 1  SWPC HGS   — ภาคพื้น/SXI/AIA มีครบทั้งสาม cycle จึงเป็นฐานหลัก
    tier 2  XRS HGS    — GOES-16..19 เติมช่วงที่ภาคพื้นไม่ได้สังเกต (2017→)
    tier 3  XRS HPC    — เหลือแค่ arcsec เพราะ flare อยู่ริมขอบ; ฉายกลับเป็น
                         Stonyhurst ใน add_coordinates()
    """
    d = df.copy()
    d["lat"] = np.nan
    d["lon"] = np.nan
    d["pos_source"] = ""

    def take(mask, lon_col, lat_col, tag):
        m = mask & d["lon"].isna()
        d.loc[m, "lon"] = d.loc[m, lon_col]
        d.loc[m, "lat"] = d.loc[m, lat_col]
        d.loc[m, "pos_source"] = tag

    take(d["flare_loc_swpc_hgs_lon"].notna() & d["flare_loc_swpc_hgs_lat"].notna(),
         "flare_loc_swpc_hgs_lon", "flare_loc_swpc_hgs_lat", "SWPC")
    take(d["flare_loc_xrs_hgs_lon"].notna() & d["flare_loc_xrs_hgs_lat"].notna(),
         "flare_loc_xrs_hgs_lon", "flare_loc_xrs_hgs_lat", "XRS")

    # tier 3 ทำเครื่องหมายไว้ก่อน ค่าจริงต้องรอ transform ใน add_coordinates()
    hpc_only = (d["lon"].isna() & d["flare_loc_xrs_hpc_x"].notna()
                & d["flare_loc_xrs_hpc_y"].notna())
    d.loc[hpc_only, "pos_source"] = "XRS-HPC"

    d["pos_detail"] = np.where(
        d["pos_source"] == "SWPC", d["flare_loc_swpc_source"].astype("string").fillna(""),
        np.where(d["pos_source"].str.startswith("XRS"),
                 d["flare_loc_xrs_source"].astype("string").fillna(""), ""))
    return d


def build_reference_positions(raw_pos: pd.DataFrame) -> pd.DataFrame:
    """flare ทุก class ที่มีทั้งพิกัดตรง ๆ และเลข AR — จุดอ้างอิงของ tier 4"""
    ok = (raw_pos["lon"].notna() & raw_pos["lat"].notna()
          & raw_pos["active_region"].notna())
    r = raw_pos.loc[ok, ["time", "active_region", "lon", "lat"]].copy()
    r["_t"] = pd.to_datetime(r["time"])
    r["_ar"] = full_active_region(r["active_region"], r["_t"])
    return r[["_ar", "_t", "lon", "lat"]].sort_values("_t").reset_index(drop=True)


def fill_from_active_region(df: pd.DataFrame, refs: pd.DataFrame,
                            window_days: float = AR_FILL_WINDOW_DAYS,
                            max_refs: int = AR_FILL_MAX_REFS) -> pd.DataFrame:
    """tier 4 — เติมพิกัดของ flare ที่ไม่มีใครรายงานตำแหน่ง จาก AR เดียวกัน

    ใช้ RotatedSunFrame ของ sunpy หมุน ref แต่ละดวงไปยังเวลาของ event (โมเดล
    differential rotation ของ Howard) แล้วเอา median เพื่อกันจุดผิดเดี่ยว ๆ
    ค่า pos_dt_days บอกว่า ref ที่ไกลสุดห่างจาก event กี่วัน ใช้กรองภายหลังได้

    ความแม่นยำ (วัดแบบ hold-out: เอา X-flare 294 ดวงที่ *มี* พิกัดจริงมาปิดพิกัด
    ทิ้งแล้วให้ tier นี้เดาใหม่ โดยตัดตัวมันเองออกจากชุด ref)

        ระยะคลาดเคลื่อนบนผิวทรงกลม  median 3.0°   p90 9°   p95 12.5°

    หางที่เหลืออีกราว 2% พลาดหนัก (สูงสุด ~170°) เพราะไฟล์ต้นทางติดเลข AR ไม่ตรง
    กันเองระหว่างสองเหตุการณ์ ไม่ใช่ความคลาดจากการหมุน — ทดลองใส่เกณฑ์ "ถ้า ref
    กระจายกว้างให้ทิ้ง" แล้วพบว่าตัดของดีทิ้ง 25% โดยจับ outlier ไม่ได้ จึงไม่ใส่
    แถวกลุ่มนี้ติด pos_source = "AR" ไว้ทุกแถว กรองออกได้ถ้าต้องการเฉพาะค่าที่วัดตรง
    """
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astropy.time import Time
    from sunpy.coordinates import (HeliographicStonyhurst, RotatedSunFrame,
                                   transform_with_sun_center)

    d = df.copy()
    if "pos_dt_days" not in d.columns:
        d["pos_dt_days"] = np.nan
    need = (d["pos_source"] == "") & d["_ar"].notna()
    if not need.any() or refs.empty:
        return d

    by_ar = dict(tuple(refs.groupby("_ar")))
    tgt_idx, ref_lon, ref_lat, t_ref, t_evt = [], [], [], [], []
    dt_max: dict[int, float] = {}

    for i, row in d.loc[need].iterrows():
        g = by_ar.get(row["_ar"])
        if g is None:
            continue
        dt = (g["_t"] - row["time_peak"]).abs().dt.total_seconds() / 86400.0
        g = g[dt <= window_days]
        if g.empty:
            continue
        g = g.assign(_dt=dt).nsmallest(max_refs, "_dt")
        tgt_idx += [i] * len(g)
        ref_lon += g["lon"].tolist()
        ref_lat += g["lat"].tolist()
        t_ref += g["_t"].tolist()
        t_evt += [row["time_peak"]] * len(g)
        dt_max[i] = float(g["_dt"].max())

    if not tgt_idx:
        return d

    def as_time(seq):
        return Time(pd.to_datetime(pd.Series(seq)).dt.strftime("%Y-%m-%dT%H:%M:%S")
                    .tolist(), format="isot", scale="utc")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = SkyCoord(np.array(ref_lon) * u.deg, np.array(ref_lat) * u.deg,
                        frame=HeliographicStonyhurst(obstime=as_time(t_ref)))
        with transform_with_sun_center():
            rot = SkyCoord(RotatedSunFrame(base=base, rotated_time=as_time(t_evt)))
            hgs = rot.transform_to(HeliographicStonyhurst(obstime=as_time(t_evt)))

    est = (pd.DataFrame({"i": tgt_idx,
                         "lon": hgs.lon.to_value(u.deg),
                         "lat": hgs.lat.to_value(u.deg)})
           .groupby("i").median())
    d.loc[est.index, "lon"] = est["lon"]
    d.loc[est.index, "lat"] = est["lat"]
    d.loc[est.index, "pos_source"] = "AR"
    d.loc[est.index, "pos_detail"] = [f"AR{int(a)}" for a in d.loc[est.index, "_ar"]]
    d.loc[est.index, "pos_dt_days"] = [round(dt_max[i], 2) for i in est.index]
    return d


# --------------------------------------------------------------------------- #
# coordinates — ให้ sunpy/astropy รับผิดชอบทั้งหมด
# --------------------------------------------------------------------------- #
def add_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """เติม (1) พิกัด tier-3 ที่ฉายจาก helioprojective  (2) carr_lon

    Carrington longitude ยึดกับตัวดวงอาทิตย์เอง ไม่ใช่กับผู้สังเกต จึงเป็นระบบ
    เดียวที่ตอบได้ว่า X-flare กระจุกที่ลองจิจูดเดิมซ้ำ ๆ ไหม (active longitude)
    """
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astropy.time import Time
    from sunpy.coordinates import (HeliographicCarrington, HeliographicStonyhurst,
                                   Helioprojective, sun)

    d = df.copy()
    d["carr_lon"] = np.nan
    if d.empty:
        return d

    def astro_time(sub: pd.DataFrame) -> Time:
        return Time(pd.to_datetime(sub["time_peak"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
                    .tolist(), format="isot", scale="utc")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # ---- tier 3: helioprojective arcsec -> Stonyhurst ------------------- #
        hpc = d["pos_source"] == "XRS-HPC"
        if hpc.any():
            sub = d.loc[hpc]
            t = astro_time(sub)
            x = sub["flare_loc_xrs_hpc_x"].to_numpy(float)
            y = sub["flare_loc_xrs_hpc_y"].to_numpy(float)
            # event กลุ่มนี้อยู่ที่ขอบพอดี วัดได้ r เกิน R_sun นิดหน่อยจาก noise
            # ถ้าไม่ดึงกลับเข้ามา เส้นสายตาจะไม่ตัดผิวดวงอาทิตย์ -> ได้ NaN
            r = np.hypot(x, y)
            r_sun = sun.angular_radius(t).to_value(u.arcsec)
            shrink = np.minimum(1.0, 0.9995 * r_sun / np.where(r > 0, r, 1.0))
            c = SkyCoord(x * shrink * u.arcsec, y * shrink * u.arcsec,
                         frame=Helioprojective(observer="earth", obstime=t))
            hgs = c.make_3d().transform_to(HeliographicStonyhurst(obstime=t))
            d.loc[hpc, "lon"] = hgs.lon.to_value(u.deg)
            d.loc[hpc, "lat"] = hgs.lat.to_value(u.deg)

        # ---- ดึงจุดที่หลุดหลังขอบจานกลับมาที่ขอบ ---------------------------- #
        # tier 3 วัดจุดศูนย์กลางการเปล่งแสงซึ่งลอยเหนือ photosphere ได้จริง ส่วน
        # tier 4 หมุน ref ข้ามขอบไปเมื่อ AR กำลังจะลับขอบพอดี ทั้งสองกรณีได้
        # |lon| > 90 ซึ่ง orthographic จะ "พับ" กลับมาวางในจานราวกับอยู่ด้านหน้า
        # (W97.7 ไปตกที่เดียวกับ W82.3) — ผิดชัด ๆ จึงหนีบไว้ที่ขอบแล้วติดธงบอก
        d["pos_at_limb"] = False
        beyond = d["lon"].notna() & (d["lon"].abs() > 90.0)
        if beyond.any():
            d.loc[beyond, "pos_at_limb"] = True
            d.loc[beyond, "lon"] = np.sign(d.loc[beyond, "lon"]) * 90.0

        # ---- Carrington ----------------------------------------------------- #
        ok = d["lat"].notna() & d["lon"].notna()
        if ok.any():
            sub = d.loc[ok]
            t = astro_time(sub)
            sty = SkyCoord(sub["lon"].to_numpy(float) * u.deg,
                           sub["lat"].to_numpy(float) * u.deg,
                           frame=HeliographicStonyhurst(obstime=t))
            carr = sty.transform_to(HeliographicCarrington(observer="earth", obstime=t))
            d.loc[ok, "carr_lon"] = carr.lon.to_value(u.deg) % 360.0

    return d


# --------------------------------------------------------------------------- #
# catalogue assembly
# --------------------------------------------------------------------------- #
OUT_COLS = [
    "cycle", "class_letter", "date", "time_peak", "time_start", "time_end",
    "goes_class", "magnitude", "goes_class_sci", "magnitude_sci",
    "goes_class_legacy", "magnitude_legacy", "saturated",
    "position", "lat", "lon", "carr_lon", "pos_source", "pos_detail", "pos_dt_days",
    "pos_at_limb",
    "active_region", "active_region_swpc",
    "satellite", "xrsb_irrad", "background_irrad",
    "integrated_irrad_peak", "integrated_irrad_end", "flare_id",
]


def build_catalog(scale: str = "science", cls: str = "X",
                  years: list[int] | None = None, offline: bool = False,
                  ar_fill: bool = True, quiet: bool = False) -> pd.DataFrame:
    """สร้าง catalogue พร้อมใช้ — 1 แถวต่อ 1 flare ที่ผ่านเกณฑ์ class

    scale   : "science" = irradiance จริง (ค่าเริ่มต้น สเกลเดียวกันทั้ง 1995→now)
              "legacy"  = คูณ 0.7 ให้ตรงกับตัวเลขที่ SWPC เคยรายงานยุค GOES 1-15
    ar_fill : เปิด tier 4 (อนุมานพิกัดจาก AR เดียวกัน) — ดู fill_from_active_region
    """
    if scale not in SCALES:
        raise ValueError(f"scale ต้องเป็น {set(SCALES)}")

    raw = load_raw(years=years, offline=offline, quiet=quiet)
    # แก้พิกัดทั้งตารางก่อนกรอง class เพราะ flare ระดับ C/M ที่มีพิกัดคือชุด
    # อ้างอิงของ tier 4 (X-flare ที่ไม่มีใครรายงานตำแหน่งมีแค่ไม่กี่สิบดวง)
    raw = resolve_position(raw)
    refs = build_reference_positions(raw) if ar_fill else pd.DataFrame()

    # คำนวณทั้งสองสเกลเสมอ แล้วค่อยเลือกว่าอันไหนคือ "magnitude" ที่ใช้คัดกรอง
    # CSV จึงอธิบายตัวเองได้ ไม่ต้องจำว่ารันด้วย --scale อะไร
    flux = raw["xrsb_irrad"].to_numpy(float)
    sci_letter, sci_mag = classify(flux)
    leg_letter, leg_mag = classify(flux * SCALE_LEGACY)
    sel = (sci_letter, sci_mag) if scale == "science" else (leg_letter, leg_mag)

    d = raw.assign(_letter=sel[0], _mag=sel[1],
                   _sci_l=sci_letter, _sci_m=sci_mag,
                   _leg_l=leg_letter, _leg_m=leg_mag)
    keep = CLASS_ORDER[CLASS_ORDER.index(cls):]
    d = d[d["_letter"].isin(keep)].reset_index(drop=True)
    if d.empty:
        return pd.DataFrame(columns=OUT_COLS)

    d["time_peak"] = pd.to_datetime(d["time"])
    d["time_start"] = pd.to_datetime(d["start_time"])
    d["time_end"] = pd.to_datetime(d["end_time"])
    d["date"] = d["time_peak"].dt.date
    d["_ar"] = full_active_region(d["active_region"], d["date"])

    if ar_fill:
        d = fill_from_active_region(d, refs)
    d = add_coordinates(d)

    for pre, (lc, mc) in {"": ("_letter", "_mag"), "_sci": ("_sci_l", "_sci_m"),
                          "_legacy": ("_leg_l", "_leg_m")}.items():
        d[f"goes_class{pre}"] = [class_label(l, m) for l, m in zip(d[lc], d[mc])]
        d[f"magnitude{pre}"] = d[mc].astype(float)

    d["class_letter"] = d["_letter"]
    d["position"] = [format_position(a, o) for a, o in zip(d["lat"], d["lon"])]
    d["active_region_swpc"] = d["active_region"]
    d["active_region"] = d["_ar"]
    d["saturated"] = d["peak_saturated"].fillna(0).astype(int).astype(bool)
    if "pos_dt_days" not in d.columns:
        d["pos_dt_days"] = np.nan
    d["satellite"] = d["xrsb_irrad_source"].astype("string").fillna("")
    d["cycle"] = assign_cycle(d["date"])

    return d[OUT_COLS].sort_values("time_peak").reset_index(drop=True)


def assign_cycle(days) -> pd.Series:
    """map วันที่ -> หมายเลข cycle (นอกช่วงที่นิยามไว้ = NaN)"""
    s = pd.Series(pd.to_datetime(pd.Series(days).astype("datetime64[ns]")))
    out = pd.Series(np.nan, index=s.index)
    for c, (d0, d1) in CYCLES.items():
        m = (s >= pd.Timestamp(d0)) & (s <= pd.Timestamp(d1))
        out[m] = c
    return out


def cycle_slice(cat: pd.DataFrame, cycle: int, cls: str | None = None) -> pd.DataFrame:
    """ตัดเอาเฉพาะ cycle (และ class ถ้าระบุ) — build_catalog คืนมาทุก class ที่ >= cls
    เพราะโหลด/แปลงพิกัดรอบเดียวแล้วค่อยแบ่งทีหลังเร็วกว่าเรียกซ้ำทีละ class
    """
    d = cat[cat["cycle"] == cycle]
    if cls is not None:
        d = d[d["class_letter"] == cls]
    return d.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# selftest
# --------------------------------------------------------------------------- #
SAMPLE = """time,start_time,end_time,flare_id,xrsb_irrad,flare_class,xrsb_irrad_source,background_irrad,integrated_irrad_peak,integrated_irrad_end,flare_loc_swpc_hgs_lon,flare_loc_swpc_hgs_lat,flare_loc_swpc_source,flare_loc_xrs_hgs_lon,flare_loc_xrs_hgs_lat,flare_loc_xrs_hpc_x,flare_loc_xrs_hpc_y,flare_loc_xrs_source,sequential_flare_num,event_id_swpc,peak_saturated,active_region
2003-11-04 19:52:00,2003-11-04 19:29:00,2003-11-04 20:06:00,200311041929,2.488e-03,X24.8,GOES-12,1.0e-06,1.0e-01,1.5e-01,83.0,-19.0,HOL,,,,,,1,7550,1,486.0
2024-05-14 16:51:00,2024-05-14 16:37:00,2024-05-14 17:14:00,202405141637,8.6e-04,X8.6,GOES-16,1.0e-06,1.0e-02,2.0e-02,,,,,,876.0074,-369.49094,GOES-16_XRS,1,1050,0,3664.0
2011-08-09 08:04:00,2011-08-09 07:48:00,2011-08-09 08:08:00,201108090748,1.010e-03,X10.1,GOES-15,1.0e-06,2.0e-02,3.0e-02,69.0,17.0,LEA,,,,,,1,2340,0,1263.0
2008-01-01 00:26:00,2008-01-01 00:03:00,2008-01-01 00:38:00,200801010003,1.003e-06,C1.0,GOES-11,1.6e-07,9.9e-04,1.4e-03,,,,,,,,,2,8420,0,
2001-04-02 21:51:00,2001-04-02 21:32:00,2001-04-02 22:03:00,200104022132,2.445e-03,X24.4,GOES-8,1.0e-06,9.0e-02,1.3e-01,,,,,,,,,1,3900,1,9393.0
"""


def selftest() -> int:
    import io

    # ---- classify: X ไม่มีเพดาน, ขอบเขต class ตรงตามนิยาม ------------------- #
    letter, mag = classify(np.array([2.488e-3, 1.0e-4, 9.99e-5, 1.0e-6, 5e-9]))
    assert list(letter) == ["X", "X", "M", "C", "A"], letter
    assert abs(mag[0] - 24.88) < 1e-6 and abs(mag[1] - 1.0) < 1e-9
    assert abs(mag[2] - 9.99) < 1e-6

    # ตัดทศนิยม ไม่ปัดขึ้น — ตรงกับคอลัมน์ flare_class ของต้นทาง 98.8% ส่วนที่
    # ต่างเป็น 0.1 ในหลักท้าย (artifact ทศนิยมฐานสองฝั่ง NOAA) กับ flux ที่
    # ตกขอบ class พอดีอีก 2 แถว ซึ่งตามนิยาม 1.0e-6 คือ C1.0 ไม่ใช่ B9.9
    assert class_label("X", 24.88) == "X24.8"
    assert class_label("C", 9.99) == "C9.9"          # ห้ามโผล่ "C10.0"
    assert class_label("B", 1.6e-7 / 1e-7) == "B1.6"  # ห้ามหล่นเป็น B1.5
    assert class_label("M", 1.0) == "M1.0"

    # AR rollover: 2003 -> เติม 10000, ก่อน 14 มิ.ย. 2002 -> ไม่แตะ
    ar = full_active_region([486.0, 9393.0, np.nan, 3664.0],
                            ["2003-11-04", "2001-04-02", "2008-01-01", "2024-05-14"])
    assert ar[0] == 10486 and ar[1] == 9393 and np.isnan(ar[2]) and ar[3] == 13664, ar

    assert format_position(-19.0, 83.0) == "S19W83"
    assert format_position(4.0, -73.0) == "N04E73"
    assert format_position(np.nan, 5.0) == ""

    # ---- pipeline บนตัวอย่างจากไฟล์จริง ------------------------------------- #
    raw = pd.read_csv(io.StringIO(SAMPLE))
    d = resolve_position(raw)
    assert list(d["pos_source"]) == ["SWPC", "XRS-HPC", "SWPC", "", ""], list(d["pos_source"])
    assert d["lon"].iloc[0] == 83.0 and d["lat"].iloc[0] == -19.0
    assert d["pos_detail"].iloc[0] == "HOL"

    d["time_peak"] = pd.to_datetime(d["time"])
    d["date"] = d["time_peak"].dt.date
    d["_ar"] = full_active_region(d["active_region"], d["date"])

    # ---- tier 4: ยืมพิกัดจาก AR เดียวกันแล้วหมุนตามเวลา --------------------- #
    # แถวที่ 4 (C1.0 ปี 2008) ไม่มีทั้งพิกัดและ AR -> ต้องยังว่างอยู่
    # ใส่ ref ให้ AR 9393 ของ 2 เม.ย. 2001 ซึ่งเป็น X24.4 ที่ไม่มีใครรายงานตำแหน่ง
    refs = pd.DataFrame({
        "_ar": [9393.0, 9393.0],
        "_t": pd.to_datetime(["2001-04-02 18:00:00", "2001-04-03 06:00:00"]),
        "lon": [60.0, 66.6], "lat": [-17.0, -17.0]})
    d = fill_from_active_region(d, refs)
    assert d["pos_source"].iloc[4] == "AR", list(d["pos_source"])
    assert d["pos_source"].iloc[3] == ""            # ไม่มี AR -> เติมไม่ได้
    # ref ทั้งสองดวงเป็นจุดเดียวกันที่หมุนไปแล้ว (13.2°/วัน) หมุนกลับต้องมาบรรจบ
    assert 61.0 < d["lon"].iloc[4] < 64.0, d["lon"].iloc[4]
    assert abs(d["lat"].iloc[4] + 17.0) < 0.5, d["lat"].iloc[4]
    assert d["pos_dt_days"].iloc[4] == 0.34, d["pos_dt_days"].iloc[4]

    d = add_coordinates(d)
    # 14 พ.ค. 2024 X8.6 อยู่ที่ขอบตะวันตกพอดี (AR 3664 กำลังหมุนพ้นขอบ)
    assert 85.0 < d["lon"].iloc[1] < 90.0, d["lon"].iloc[1]
    # NaN <= 90 เป็น False จึงต้อง dropna ก่อน ไม่งั้นแถวที่ไม่มีพิกัดทำให้ตก
    assert d["lon"].dropna().abs().le(90.0).all(), "หลุดหลังขอบต้องถูกหนีบไว้ที่ขอบ"
    assert -26.0 < d["lat"].iloc[1] < -20.0, d["lat"].iloc[1]
    assert np.isfinite(d["carr_lon"].iloc[0]) and 0 <= d["carr_lon"].iloc[0] < 360

    # ---- สองสเกลต่างกัน 0.7 เป๊ะ -------------------------------------------- #
    _, m_sci = classify(np.array([2.488e-3]))
    _, m_leg = classify(np.array([2.488e-3]) * SCALE_LEGACY)
    assert abs(m_leg[0] / m_sci[0] - 0.7) < 1e-9
    # 28 ต.ค. 2003: science X24.6 <-> ตัวเลขที่คุ้นเคย X17.2
    _, m = classify(np.array([2.461e-3]) * SCALE_LEGACY)
    assert abs(m[0] - 17.2) < 0.05, m

    print("selftest: OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[2])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--scale", default="science", choices=sorted(SCALES))
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    cat = build_catalog(scale=args.scale, offline=args.offline)
    print(cat.groupby("cycle").size().to_string())
    return 0


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
