#!/usr/bin/env python3
"""
interactive_map.py
==================
เขียนหน้า HTML แบบ interactive หนึ่งไฟล์ — จานสุริยะที่ชี้จุดแล้วขึ้นรายละเอียด
ของ flare ดวงนั้น (วัน-เวลา, class, flux, ตำแหน่ง, AR, ที่มาของพิกัด)

ทำไมไม่ใช้ plotly / bokeh
-------------------------
จุดที่ต้องวาดมีสามหมื่นกว่าดวง ถ้าให้แต่ละดวงเป็น element ของ SVG/DOM หน้าจะ
หนืดตั้งแต่โหลด ที่นี่จึงวาดลง <canvas> ตัวเดียวแล้วทำ hit-test เองด้วย spatial
hash — ได้ไฟล์เดียวจบ ไม่ต้องพึ่ง CDN และเปิดจากเครื่องได้เลย

ทำไมจุดถึงวางตรงกับรูป static
-----------------------------
ใช้ orthographic projection บนสมมติฐาน B0 = 0 ชุดเดียวกับ solar_plots.py
(x = cos(lat)·sin(lon), y = sin(lat)) รูปนิ่งกับหน้า interactive จึงอ่านคู่กันได้
ตรง ๆ ไม่ใช่คนละระบบพิกัด
"""

from __future__ import annotations

import base64
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

from goes_flare_report import POS_TIER_LABEL, SCALE_LEGACY
from proton_flux import CME_BG_DAYS, CME_P10_PFU

EPOCH = datetime(1995, 1, 1)

# --------------------------------------------------------------------------- #
# กราฟ proton flux
# --------------------------------------------------------------------------- #
# ช่องพลังงานเป็นตัวแปร *เชิงลำดับ* (>10 ครอบ >50 ครอบ >100) ไม่ใช่หมวดหมู่
# จึงใช้ ramp ไล่เข้มของสีเดียว ไม่ใช่ hue คนละสีแบบ CLASS_STYLE — และต้องไม่ไป
# ชนความหมายของสีที่ใช้แทน flare class อยู่แล้วในหน้าเดียวกัน
# สีเก็บเป็นชื่อ CSS var ไม่ใช่ค่าตรง เพราะ ramp ต้องกลับทิศเมื่อสลับธีม:
# ธีมสว่างไล่จากอ่อนไปเข้ม ธีมมืดไล่จากเข้มไปสว่าง — ยึดหลักเดียวกันคือ
# "ยิ่งพลังงานสูง ยิ่งตัดกับพื้นหลังมาก" ถ้าใช้สีตายตัว เส้น >100 MeV สีกรม
# จะจมหายไปกับพื้นมืด
PROTON_STYLE = [
    {"key": "p1", "label": "> 1 MeV", "var": "--pline0", "width": 2.4},
    {"key": "p10", "label": "> 10 MeV", "var": "--pline1", "width": 2.2},
    {"key": "p50", "label": "> 50 MeV", "var": "--pline2", "width": 2.0},
    {"key": "p100", "label": "> 100 MeV", "var": "--pline3", "width": 1.8},
]
# ระดับ NOAA S-scale วัดที่ช่อง >10 MeV หน่วย pfu (protons/cm²-s-sr)
S_SCALE = [(10, "S1"), (100, "S2"), (1000, "S3"), (10000, "S4"), (100000, "S5")]

# หน้าต่างเวลาที่แสดงรอบ flare — SEP มักขึ้นภายในไม่กี่ชั่วโมงหลัง flare แล้วค่อย ๆ
# ลดลงเป็นวัน จึงเผื่อก่อนไว้น้อยและหลังไว้มาก
CHART_BEFORE_H, CHART_AFTER_H = 12, 60

# --------------------------------------------------------------------------- #
# สีตาม class
# --------------------------------------------------------------------------- #
# ที่นี่ "class" เป็นตัวแปรเชิงหมวดหมู่หลักของภาพ (ไม่ใช่ซีรีส์เดียวแบบรูปนิ่ง)
# จึงใช้ categorical palette ที่ผ่าน all-pairs check ทั้งสามช่องได้เต็ม ๆ —
# ส้มในหน้านี้ไม่ได้ถูกสงวนไว้เป็น data-quality flag เหมือนใน solar_plots.py
#
# ความรุนแรงเป็นลำดับ (C < M < X) แต่ hue ไม่สื่อลำดับ จึงให้ "ขนาดจุด" เป็นตัว
# บอกลำดับแทน แล้วปล่อยให้ hue ทำหน้าที่แยกหมวดอย่างเดียว
# ขนาดต้องถ่วงกับ "จำนวน" ด้วย ไม่ใช่ดูแต่ลำดับความรุนแรง: C มี 26,000 ดวง
# M มี 4,700 ดวง ถ้าให้ M ใหญ่กว่ามาก ๆ แล้ววาดทับทีหลัง ภาพจะออกมาเหมือน M
# เยอะกว่า C ซึ่งตรงข้ามกับความจริง จึงบีบช่วงขนาดให้แคบลงและลด alpha ของ M
CLASS_STYLE = {
    "C": {"color": "#2a78d6", "r": 1.5, "alpha": 0.50},
    "M": {"color": "#eb6834", "r": 2.1, "alpha": 0.62},
    "X": {"color": "#1baf7a", "r": 3.8, "alpha": 0.95},
}
CLASS_ORDER = ["C", "M", "X"]          # X วาดท้ายสุด = อยู่บนสุด

# --------------------------------------------------------------------------- #
# design tokens ของหน้าเว็บ
# --------------------------------------------------------------------------- #
# ค่าธีมมืดต้องไปโผล่ใน CSS สองที่ (@media prefers-color-scheme กับ
# [data-theme="dark"] ที่ปุ่มสลับตั้งให้) จึงประกาศไว้ที่เดียวแล้วให้ Python
# แจกลงทั้งสองบล็อก — เคยแก้มือแล้วอัปเดตไม่ครบทั้งคู่ สีเลยเพี้ยนต่างกัน
LIGHT_VARS = {
    "surface": "#fcfcfb", "panel": "#ffffff", "primary": "#0b0b0b",
    "secondary": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
    "rule": "#c3c2b7", "disk-in": "#fbf3e8", "disk-out": "#e9d3b4",
    "gridline": "rgba(255,255,255,.62)",
    "pline0": "#a4ccf4", "pline1": "#7cb0ee", "pline2": "#4283d3", "pline3": "#124687",
    "shadow": "0 1px 3px rgba(0,0,0,.07),0 6px 20px rgba(0,0,0,.05)",
}
DARK_VARS = {
    "surface": "#15161a", "panel": "#1c1e23", "primary": "#f2f2f0",
    "secondary": "#b9b8b3", "muted": "#86857f", "grid": "#2c2e34",
    "rule": "#3a3d45", "disk-in": "#4a3d30", "disk-out": "#2e251c",
    # grid บนจานมืดต้องจางกว่ามาก ไม่งั้นเส้นขาวแย่งสายตาไปจากจุดข้อมูล
    "gridline": "rgba(255,255,255,.20)",
    "pline0": "#2f6fb5", "pline1": "#4e8fd4", "pline2": "#86b5ee", "pline3": "#b8d2f6",
    "shadow": "0 1px 3px rgba(0,0,0,.5),0 6px 20px rgba(0,0,0,.4)",
}


def _vars(d: dict) -> str:
    return "\n".join(f"    --{k}:{v};" for k, v in d.items())


def _theme_css() -> str:
    return (":root{\n" + _vars(LIGHT_VARS) + "\n  }\n"
            "  @media (prefers-color-scheme:dark){\n"
            "    :root:not([data-theme=\"light\"]){\n" + _vars(DARK_VARS) + "\n    }\n  }\n"
            "  :root[data-theme=\"dark\"]{\n" + _vars(DARK_VARS) + "\n  }\n"
            "  :root[data-theme=\"light\"]{\n" + _vars(LIGHT_VARS) + "\n  }")


def _payload(cat: pd.DataFrame, cme: np.ndarray | None = None) -> dict:
    """บีบ catalogue ให้เป็น array ขนานกัน — JSON เล็กกว่า array-of-object ~3 เท่า

    cme (ถ้ามี) ต้องเป็น array บูลีนยาวเท่า cat เรียงแถวเดียวกับ cat ทุกประการ —
    เป็นผลจาก proton_flux.flag_cme() ที่ make_report.py คำนวณไว้ก่อนเรียกฟังก์ชันนี้
    แนบเป็นคอลัมน์ตั้งแต่ก่อน dropna/filter/sort เพื่อให้ค่าติดไปกับแถวของมันถูกต้อง
    แม้ลำดับ/จำนวนแถวจะเปลี่ยนไปหลังจากนั้น
    """
    d = cat.copy()
    d["_cme"] = cme if cme is not None else False
    d = d.dropna(subset=["lat", "lon"])
    d = d[d["class_letter"].isin(CLASS_ORDER)]
    d = d.sort_values(["class_letter", "time_peak"],
                      key=lambda s: s.map(CLASS_ORDER.index) if s.name == "class_letter" else s)

    t = pd.to_datetime(d["time_peak"])
    minutes = ((t - pd.Timestamp(EPOCH)).dt.total_seconds() // 60).astype("int64")
    tiers = sorted(POS_TIER_LABEL)

    return {
        "cyc": d["cycle"].astype(int).tolist(),
        "cls": [CLASS_ORDER.index(c) for c in d["class_letter"]],
        "lat": [round(float(v), 2) for v in d["lat"]],
        "lon": [round(float(v), 2) for v in d["lon"]],
        # ตัดทศนิยมแบบเดียวกับ class_label() ไม่ใช่ปัด — ปัดขึ้นจะได้ "C10.0"
        # ซึ่งไม่มีอยู่จริง (เกิน C9.9 ไปแล้วคือ M1.0)
        "mag": [float(np.floor(np.round(v, 6) * 10 + 1e-6) / 10)
                for v in d["magnitude"]],
        "flux": [float(f"{v:.3g}") for v in d["xrsb_irrad"]],
        "t": minutes.tolist(),
        "ar": [0 if not np.isfinite(v) else int(v) for v in d["active_region"]],
        "src": [tiers.index(s) if s in tiers else -1 for s in d["pos_source"]],
        "sat": [int(bool(v)) for v in d["saturated"]],
        "limb": [int(bool(v)) for v in d.get("pos_at_limb", pd.Series(False, index=d.index))],
        "tierNames": [POS_TIER_LABEL[k] for k in tiers],
        "classes": CLASS_ORDER,
        "style": [CLASS_STYLE[c] for c in CLASS_ORDER],
        "cme": [int(bool(v)) for v in d["_cme"]],
        "cmeAvailable": cme is not None,
    }


def _proton_payload(series: dict | None) -> dict | None:
    """อนุกรมรายชั่วโมงจาก proton_flux.build_hourly_series -> ก้อนสำหรับฝังใน JS

    ส่งเป็น base64 ของ uint8 ไม่ใช่ JSON array ของตัวเลข — 246,000 ชั่วโมง × 3 ช่อง
    ถ้าเขียนเป็น "123," จะกินราว 3 MB แต่ base64 เหลือ 1 MB
    """
    if series is None:
        return None
    enc = np.asarray(series["enc"], dtype=np.uint8)

    # cache เป็นไฟล์แยก จึงค้างจากรอบก่อนที่ใช้ช่องคนละชุดได้ ถ้าไม่ตรวจตรงนี้
    # enc[i] จะไปหยิบแถวผิดช่องแบบเงียบ ๆ (เช่น เอา >50 มาแสดงว่าเป็น >100)
    want = [c["key"] for c in PROTON_STYLE]
    have = list(series.get("channels", []))
    if have != want or enc.shape[0] != len(want):
        raise ValueError(
            f"cache โปรตอนมีช่อง {have} (enc {enc.shape}) แต่โค้ดต้องการ {want} — "
            "สร้างใหม่ด้วย: python proton_flux.py --build-series")

    return {
        "t0": series["t0"], "stepMin": int(series["step_min"]), "n": int(series["n"]),
        "logLo": float(series["log_lo"]), "logHi": float(series["log_hi"]),
        "channels": [c["key"] for c in PROTON_STYLE],
        "style": PROTON_STYLE, "sScale": S_SCALE,
        "beforeH": CHART_BEFORE_H, "afterH": CHART_AFTER_H,
        "b64": [base64.b64encode(enc[i].tobytes()).decode("ascii")
                for i, _ in enumerate(PROTON_STYLE)],
    }


def write_html(cat: pd.DataFrame, outfile: str, scale: str = "science",
               proton_series: dict | None = None,
               cme_flags: np.ndarray | None = None) -> str:
    data = _payload(cat, cme_flags)
    data["proton"] = _proton_payload(proton_series)
    n_all = len(cat)
    n_pos = len(data["lat"])
    note = ("magnitude เป็น irradiance จริง (science quality) — ยุค GOES 1-15 "
            f"SWPC รายงานค่านี้ × {SCALE_LEGACY}" if scale == "science" else
            f"magnitude เป็นสเกลรายงานยุค GOES 1-15 (irradiance จริง × {SCALE_LEGACY})")
    cme_note = (
        f'ตัวกรอง "CME" คำนวณจาก proton flux ไม่ใช่การตรวจพบ CME โดยตรง '
        f"(คลังนี้ไม่มีข้อมูลภาพ coronagraph) — flare เข้าเกณฑ์เมื่อ >10 MeV พีคภายใน "
        f"{CHART_AFTER_H} ชม. ≥ {CME_P10_PFU:g} pfu <b>และ</b> >100 MeV พีคในช่วงเดียวกัน "
        f"สูงกว่าค่าเฉลี่ย {CME_BG_DAYS} วันก่อนหน้า<br>"
        if cme_flags is not None else "")

    html = (_TEMPLATE
            .replace("__DATA__", json.dumps(data, separators=(",", ":")))
            .replace("__NPOS__", f"{n_pos:,}")
            .replace("__NALL__", f"{n_all:,}")
            .replace("__SCALENOTE__", note)
            .replace("__CMENOTE__", cme_note)
            .replace("__THEMECSS__", _theme_css())
            .replace("__BUILT__", datetime.now().strftime("%Y-%m-%d %H:%M")))

    os.makedirs(os.path.dirname(os.path.abspath(outfile)), exist_ok=True)
    with open(outfile, "w", encoding="utf-8") as f:
        f.write(html)
    return outfile


# --------------------------------------------------------------------------- #
# template
# --------------------------------------------------------------------------- #
_TEMPLATE = r"""<!doctype html>
<html lang="th">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ตำแหน่ง solar flare บนจานสุริยะ — GOES XRS</title>
<style>
  __THEMECSS__
  *{box-sizing:border-box}
  body{
    margin:0; background:var(--surface); color:var(--primary);
    font-family:"Leelawadee UI",Tahoma,"Noto Sans Thai",Sarabun,system-ui,sans-serif;
    font-size:15px; line-height:1.5;
  }
  .wrap{max-width:1180px; margin:0 auto; padding:28px 20px 48px}
  h1{font-size:22px; font-weight:600; margin:0 0 6px; letter-spacing:-.01em}
  .sub{color:var(--secondary); font-size:13.5px; margin:0 0 22px}
  .bar{
    display:flex; flex-wrap:wrap; gap:22px 30px; align-items:flex-end;
    padding:0 0 16px; border-bottom:1px solid var(--grid); margin-bottom:20px;
  }
  .grp{display:flex; flex-direction:column; gap:7px}
  .lab{font-size:11px; color:var(--muted); letter-spacing:.02em}
  .seg{display:flex; gap:4px}
  button{
    font:inherit; font-size:13px; color:var(--secondary); background:transparent;
    border:1px solid var(--rule); border-radius:7px; padding:5px 12px;
    cursor:pointer; transition:.12s;
  }
  button:hover{border-color:var(--secondary); color:var(--primary)}
  button[aria-pressed="true"]{
    background:var(--primary); color:var(--surface); border-color:var(--primary);
  }
  button.cls[aria-pressed="true"]{
    background:var(--c); border-color:var(--c); color:#fff;
  }
  button.cls{border-left:4px solid var(--c)}
  button.cls[aria-pressed="false"]{opacity:.5}
  /* stage ต้องกว้างเท่า canvas พอดี เพราะ tooltip วางตำแหน่งด้วยพิกัดที่วัดจาก
     canvas — ถ้า stage กว้างกว่าแล้ว canvas ถูกจัดกลาง ป้ายจะเลื่อนไปครึ่งหนึ่ง */
  .stage{position:relative; width:100%; max-width:760px; margin:0 auto}
  canvas{display:block; width:100%; cursor:crosshair; touch-action:none}
  #tip{
    position:absolute; pointer-events:none; opacity:0; transition:opacity .1s;
    background:var(--panel); border:1px solid var(--rule); border-radius:9px;
    box-shadow:var(--shadow); padding:10px 12px; font-size:12.5px; min-width:210px;
    z-index:5;
  }
  #tip .hd{font-weight:600; font-size:14px; margin-bottom:5px; display:flex;
           align-items:center; gap:7px}
  #tip .dot{width:9px; height:9px; border-radius:50%; flex:0 0 auto}
  #tip table{border-collapse:collapse; width:100%}
  #tip td{padding:1.5px 0; vertical-align:top}
  #tip td:first-child{color:var(--muted); padding-right:12px; white-space:nowrap}
  #tip td:last-child{color:var(--secondary); text-align:right;
                     font-variant-numeric:tabular-nums}
  .counts{display:flex; flex-wrap:wrap; gap:18px; margin:16px 0 0; font-size:13px;
          color:var(--secondary); justify-content:center}
  .counts b{color:var(--primary); font-variant-numeric:tabular-nums}
  .swatch{display:inline-block; width:10px; height:10px; border-radius:50%;
          margin-right:6px; vertical-align:-1px}
  footer{margin-top:26px; padding-top:14px; border-top:1px solid var(--grid);
         color:var(--muted); font-size:11.5px; line-height:1.7}
  .hint{color:var(--muted); font-size:12px; text-align:center; margin-top:10px}
  #pwrap{margin:26px auto 0; max-width:900px; padding-top:20px;
         border-top:1px solid var(--grid)}
  .phead{display:flex; flex-wrap:wrap; gap:10px 24px; align-items:baseline;
         justify-content:space-between; margin-bottom:12px}
  .phead h2{font-size:15px; font-weight:600; margin:0}
  #pcap{margin:3px 0 0; color:var(--secondary); font-size:12.5px;
        font-variant-numeric:tabular-nums}
  .plegend{display:flex; gap:16px; font-size:12px; color:var(--secondary)}
  .plegend i{display:inline-block; width:14px; height:3px; border-radius:2px;
             margin-right:6px; vertical-align:3px}
  #pchart{display:block; width:100%; height:auto}
</style>
</head>
<body>
<div class="wrap">
  <h1>ตำแหน่ง solar flare บนจานสุริยะ</h1>
  <p class="sub">
    __NPOS__ จาก __NALL__ เหตุการณ์ที่ระบุพิกัด heliographic ได้ &nbsp;·&nbsp;
    Solar Cycle 23 / 24 / 25 &nbsp;·&nbsp; ชี้ที่จุดเพื่อดูรายละเอียด
  </p>

  <div class="bar">
    <div class="grp">
      <span class="lab">SOLAR CYCLE</span>
      <div class="seg" id="cycSeg"></div>
    </div>
    <div class="grp">
      <span class="lab">CLASS</span>
      <div class="seg" id="clsSeg"></div>
    </div>
    <div class="grp" id="cmeGrp" hidden>
      <span class="lab">CME</span>
      <div class="seg" id="cmeSeg"></div>
    </div>
    <div class="grp">
      <span class="lab">ธีม</span>
      <div class="seg"><button id="themeBtn" aria-pressed="false">สลับสว่าง/มืด</button></div>
    </div>
  </div>

  <div class="stage">
    <canvas id="cv"></canvas>
    <div id="tip"></div>
  </div>
  <div class="counts" id="counts"></div>
  <p class="hint">E = ขอบตะวันออก (ซ้าย) · W = ขอบตะวันตก (ขวา) · grid Stonyhurst ทุก 15°</p>

  <section id="pwrap" hidden>
    <div class="phead">
      <div>
        <h2>Proton flux รอบเวลาที่เกิด flare</h2>
        <p id="pcap">ชี้ที่จุด flare เพื่อดูกราฟ · คลิกเพื่อตรึงไว้</p>
      </div>
      <div class="plegend" id="plegend"></div>
    </div>
    <canvas id="pchart"></canvas>
  </section>

  <footer>
    ที่มา NOAA/NCEI GOES XRS L2 Flare Report (science quality) — xrsf-l2-flrpt<br>
    __SCALENOTE__<br>
    __CMENOTE__
    จุดถูกฉายแบบ orthographic บนสมมติฐาน B<sub>0</sub> = 0 ชุดเดียวกับรูปนิ่งใน out/
    จึงวางทับกันได้ · เหตุการณ์ที่ไม่มีพิกัดส่วนใหญ่เกิดหลังขอบจาน · สร้างเมื่อ __BUILT__
  </footer>
</div>

<script>
const D = __DATA__;
const N = D.lat.length;
const EPOCH = Date.UTC(1995, 0, 1);
const CYCLES = [23, 24, 25];

/* ---- สถานะตัวกรอง ---------------------------------------------------- */
const state = { cyc: new Set(CYCLES), cls: new Set([0, 1, 2]), cme: false };

/* ---- projection: orthographic, B0 = 0 (ตรงกับ solar_plots.py) -------- */
const RAD = Math.PI / 180;
function proj(latDeg, lonDeg) {
  const la = latDeg * RAD, lo = lonDeg * RAD;
  return [Math.cos(la) * Math.sin(lo), Math.sin(la)];   // หน่วย R_sun
}

/* ---- canvas ---------------------------------------------------------- */
const cv = document.getElementById('cv');
const ctx = cv.getContext('2d');
const base = document.createElement('canvas');   // ชั้นนิ่ง: จาน + grid + จุด
const bctx = base.getContext('2d');
const tip = document.getElementById('tip');
let W = 0, H = 0, CX = 0, CY = 0, R = 0, DPR = 1;
const FOV = 2.25;               // ความกว้างภาพ = 2.25 R_sun (เท่ารูปนิ่ง)
const TAU = Math.PI * 2;

function layout() {
  const cssW = Math.max(280, Math.min(760, cv.parentElement.clientWidth));
  const cssH = cssW;
  DPR = window.devicePixelRatio || 1;
  for (const c of [cv, base]) {
    c.width = Math.round(cssW * DPR);
    c.height = Math.round(cssH * DPR);
  }
  cv.style.width = cssW + 'px';
  cv.style.height = cssH + 'px';
  ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
  bctx.setTransform(DPR, 0, 0, DPR, 0, 0);
  W = cssW; H = cssH; CX = cssW / 2; CY = cssH / 2;
  R = cssW / FOV;               // ภาพกว้าง FOV เท่าของรัศมี -> 1 R_sun = W/FOV px
}

function css(v) { return getComputedStyle(document.documentElement)
                    .getPropertyValue(v).trim(); }

function drawDisk(g2) {
  g2.clearRect(0, 0, W, H);
  const g = g2.createRadialGradient(CX, CY, 0, CX, CY, R);
  g.addColorStop(0, css('--disk-in'));
  g.addColorStop(1, css('--disk-out'));
  g2.fillStyle = g;
  g2.beginPath(); g2.arc(CX, CY, R, 0, TAU); g2.fill();

  /* grid Stonyhurst ทุก 15° */
  g2.save();
  g2.strokeStyle = css('--gridline');
  g2.lineWidth = 0.8;
  for (let lon = -75; lon <= 75; lon += 15) {          // เส้นเมริเดียน
    g2.beginPath();
    for (let lat = -90; lat <= 90; lat += 2) {
      const [x, y] = proj(lat, lon);
      const px = CX + x * R, py = CY - y * R;
      lat === -90 ? g2.moveTo(px, py) : g2.lineTo(px, py);
    }
    g2.stroke();
  }
  for (let lat = -75; lat <= 75; lat += 15) {          // เส้นขนาน
    g2.beginPath();
    for (let lon = -90; lon <= 90; lon += 2) {
      const [x, y] = proj(lat, lon);
      const px = CX + x * R, py = CY - y * R;
      lon === -90 ? g2.moveTo(px, py) : g2.lineTo(px, py);
    }
    g2.stroke();
  }
  g2.restore();

  /* ขอบจาน + ทิศ */
  g2.strokeStyle = css('--rule'); g2.lineWidth = 1;
  g2.beginPath(); g2.arc(CX, CY, R, 0, TAU); g2.stroke();
  g2.fillStyle = css('--secondary');
  g2.font = '600 13px system-ui, sans-serif';
  g2.textAlign = 'center'; g2.textBaseline = 'middle';
  const pad = R * 1.07;
  g2.fillText('N', CX, CY - pad); g2.fillText('S', CX, CY + pad);
  g2.fillText('E', CX - pad, CY); g2.fillText('W', CX + pad, CY);
}

/* ---- spatial hash สำหรับ hit-test ------------------------------------ */
let vis = [];                 // ดัชนีของจุดที่ผ่านตัวกรอง (เรียงตาม class)
let sx = null, sy = null;     // พิกัดจอของทุกจุด
const CELL = 12;
let hash = new Map();

function project() {
  sx = new Float32Array(N); sy = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const [x, y] = proj(D.lat[i], D.lon[i]);
    sx[i] = CX + x * R; sy[i] = CY - y * R;
  }
}

function rebuild() {
  vis = [];
  for (let i = 0; i < N; i++)
    if (state.cyc.has(D.cyc[i]) && state.cls.has(D.cls[i]) &&
        (!state.cme || D.cme[i])) vis.push(i);
  /* วาดตาม class จากน้อยไปมาก -> X อยู่บนสุด (payload เรียงมาแล้ว) */
  hash = new Map();
  for (const i of vis) {
    const k = (Math.floor(sx[i] / CELL) << 12) ^ Math.floor(sy[i] / CELL);
    let b = hash.get(k); if (!b) hash.set(k, b = []); b.push(i);
  }
}

/* ชั้นนิ่งวาดครั้งเดียวต่อการเปลี่ยนตัวกรอง แล้ว hover แค่ blit ทับ —
   ถ้าวาดจุดสามหมื่นใหม่ทุกครั้งที่เมาส์ขยับ จะกระตุก
   จุดทั้งกลุ่มรวมเป็น path เดียวต่อ class (fill ครั้งเดียว) ไม่ใช่ fill ทีละดวง */
function paintBase() {
  drawDisk(bctx);
  let k = 0;
  while (k < vis.length) {
    const cls = D.cls[vis[k]], st = D.style[cls];
    bctx.fillStyle = st.color; bctx.globalAlpha = st.alpha;
    bctx.beginPath();
    while (k < vis.length && D.cls[vis[k]] === cls) {
      const i = vis[k++];
      bctx.moveTo(sx[i] + st.r, sy[i]);
      bctx.arc(sx[i], sy[i], st.r, 0, TAU);
    }
    bctx.fill();
  }
  bctx.globalAlpha = 1;
  counts();
}

function render() {
  ctx.clearRect(0, 0, W, H);
  ctx.drawImage(base, 0, 0, W, H);
  if (hoverIdx >= 0) {                          // เน้นจุดที่ชี้อยู่
    const st = D.style[D.cls[hoverIdx]];
    ctx.beginPath();
    ctx.arc(sx[hoverIdx], sy[hoverIdx], st.r + 4.5, 0, TAU);
    ctx.strokeStyle = css('--primary'); ctx.lineWidth = 1.6; ctx.stroke();
    // วงประรอบนอก เห็นเฉพาะจุดที่ชี้อยู่ตอนนี้และเข้าเกณฑ์ CME — เดิมเคยตีวงบนจุด
    // ทุกจุดที่เข้าเกณฑ์ค้างไว้ตลอด (~17% ของทั้งหมด) แต่ลายตาเกินไปเมื่อดูภาพรวม
    // จึงย้ายมาไว้ในชั้น hover นี้ ให้ขึ้นเฉพาะตอนชี้แทน
    if (D.cmeAvailable && D.cme[hoverIdx]) {
      ctx.beginPath();
      ctx.arc(sx[hoverIdx], sy[hoverIdx], st.r + 7.5, 0, TAU);
      ctx.setLineDash([2, 2]); ctx.lineWidth = 1; ctx.stroke();
      ctx.setLineDash([]);
    }
  }
}

function counts() {
  const n = [0, 0, 0];
  for (const i of vis) n[D.cls[i]]++;
  document.getElementById('counts').innerHTML = D.classes.map((c, k) =>
    `<span><i class="swatch" style="background:${D.style[k].color}"></i>` +
    `${c}-class <b>${n[k].toLocaleString('th-TH')}</b></span>`).join('');
}

/* ---- กราฟ proton flux -------------------------------------------------- */
const P = D.proton;
const pw = document.getElementById('pwrap');
const pc = document.getElementById('pchart');
const pcap = document.getElementById('pcap');
let pctx = null, PW = 0, PH = 0, pseries = null, pinned = -1;

if (P) {
  pw.hidden = false;
  pctx = pc.getContext('2d');
  /* base64 -> Uint8Array ต่อช่อง (0 = ไม่มีข้อมูล, 1..255 = log10 flux) */
  pseries = P.b64.map(s => {
    const bin = atob(s), a = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) a[i] = bin.charCodeAt(i);
    return a;
  });
  P.t0ms = Date.parse(P.t0);
  document.getElementById('plegend').innerHTML = P.style.map(s =>
    `<span><i style="background:var(${s.var})"></i>${s.label}</span>`).join('');
}

function fmtPfu(v) {
  // flux กินช่วงหลายสิบเท่า ใช้ toPrecision อย่างเดียวจะได้ "1.49e+3" ซึ่งอ่านยาก
  if (!isFinite(v)) return '—';
  if (v >= 1000) return Math.round(v).toLocaleString('en-US');
  if (v >= 10) return v.toFixed(1);
  return v.toPrecision(2);
}

function decodeFlux(ch, idx) {
  const q = pseries[ch][idx];
  if (q === 0) return NaN;                       // 0 สงวนไว้เป็น "ไม่มีข้อมูล"
  return Math.pow(10, P.logLo + (q - 1) / 254 * (P.logHi - P.logLo));
}

function fluxAt(tms, ch) {
  if (!P) return NaN;
  const i = Math.floor((tms - P.t0ms) / (P.stepMin * 60000));
  return (i < 0 || i >= P.n) ? NaN : decodeFlux(ch, i);
}

function layoutChart() {
  if (!P) return;
  const cssW = Math.max(280, pw.clientWidth);
  const cssH = Math.round(Math.min(300, Math.max(190, cssW * 0.34)));
  pc.width = Math.round(cssW * DPR); pc.height = Math.round(cssH * DPR);
  pc.style.height = cssH + 'px';
  pctx.setTransform(DPR, 0, 0, DPR, 0, 0);
  PW = cssW; PH = cssH;
}

const PAD = { l: 52, r: 30, t: 14, b: 26 };   // r เผื่อป้าย S-scale ที่อยู่นอกกรอบขวา
const YLO = -2, YHI = 5;                          // log10 pfu

function drawChart(i) {
  if (!P) return;
  const g = pctx;
  g.clearRect(0, 0, PW, PH);
  const x0 = PAD.l, x1 = PW - PAD.r, y0 = PAD.t, y1 = PH - PAD.b;
  const X = h => x0 + (h + P.beforeH) / (P.beforeH + P.afterH) * (x1 - x0);
  const Y = lg => y1 - (lg - YLO) / (YHI - YLO) * (y1 - y0);

  /* กริดแนวนอนทีละ decade + ป้าย S-scale ที่ขอบขวา */
  g.font = '10px system-ui, sans-serif'; g.textBaseline = 'middle';
  for (let lg = YLO; lg <= YHI; lg++) {
    const y = Y(lg);
    g.strokeStyle = css('--grid'); g.lineWidth = 1;
    g.beginPath(); g.moveTo(x0, y); g.lineTo(x1, y); g.stroke();
    g.fillStyle = css('--muted'); g.textAlign = 'right';
    g.fillText('1e' + lg, x0 - 7, y);
    const s = P.sScale.find(v => Math.abs(Math.log10(v[0]) - lg) < 1e-9);
    if (s) { g.textAlign = 'left'; g.fillStyle = css('--muted');
             g.fillText(s[1], x1 + 3, y); }
  }
  /* เส้นเวลาทุก 12 ชม. */
  g.textAlign = 'center'; g.textBaseline = 'top';
  for (let h = -P.beforeH; h <= P.afterH; h += 12) {
    const x = X(h);
    g.strokeStyle = css('--grid'); g.beginPath();
    g.moveTo(x, y0); g.lineTo(x, y1); g.stroke();
    g.fillStyle = css('--muted');
    g.fillText((h > 0 ? '+' : '') + h + ' ชม.', x, y1 + 6);
  }

  if (i < 0) { pcap.textContent = 'ชี้ที่จุด flare เพื่อดูกราฟ · คลิกเพื่อตรึงไว้'; return; }

  const tms = EPOCH + D.t[i] * 60000;
  const base = Math.floor((tms - P.t0ms) / (P.stepMin * 60000));
  let any = false;
  P.style.forEach((st, ch) => {
    g.strokeStyle = css(st.var); g.lineWidth = st.width;
    g.lineJoin = 'round'; g.lineCap = 'round';
    g.beginPath();
    let pen = false;
    for (let h = -P.beforeH; h <= P.afterH; h++) {
      const k = base + h;
      const v = (k < 0 || k >= P.n) ? NaN : decodeFlux(ch, k);
      if (!isFinite(v)) { pen = false; continue; }     // เว้นช่องที่ข้อมูลขาด
      any = true;
      const x = X(h), y = Y(Math.max(YLO, Math.min(YHI, Math.log10(v))));
      pen ? g.lineTo(x, y) : g.moveTo(x, y);
      pen = true;
    }
    g.stroke();
  });

  /* เส้นเวลาที่ flare พีค */
  const xf = X(0);
  g.strokeStyle = css('--primary'); g.lineWidth = 1.2;
  g.beginPath(); g.moveTo(xf, y0); g.lineTo(xf, y1); g.stroke();
  g.fillStyle = css('--primary'); g.textAlign = 'center'; g.textBaseline = 'bottom';
  g.font = '600 10px system-ui, sans-serif';
  g.fillText(D.classes[D.cls[i]] + D.mag[i].toFixed(1), xf, y0 - 1);

  // ค่า "ขณะนั้น" อย่างเดียวไม่พอ — SEP มาถึงโลกช้ากว่าแสงเป็นชั่วโมง ค่าที่เวลา
  // flare จึงมักยังเป็น background อยู่ ต้องบอกยอดสูงสุดในหน้าต่างด้วยถึงจะเห็นว่า
  // flare ดวงนี้ตามมาด้วยพายุรังสีหรือเปล่า
  const now = fluxAt(tms, 0);
  let peak = NaN;
  for (let h = 0; h <= P.afterH; h++) {
    const k = base + h;
    if (k < 0 || k >= P.n) continue;
    const v = decodeFlux(0, k);
    if (isFinite(v) && !(v <= peak)) peak = v;
  }
  const sLevel = P.sScale.filter(s => peak >= s[0]).pop();
  pcap.innerHTML = any
    ? `${fmtTime(D.t[i])} &nbsp;·&nbsp; >10 MeV ขณะนั้น ` +
      `<b>${fmtPfu(now)}</b> pfu` +
      (isFinite(peak) ? ` &nbsp;·&nbsp; สูงสุดใน +${P.afterH} ชม. ` +
        `<b>${fmtPfu(peak)}</b> pfu` +
        (sLevel ? ` (${sLevel[1]})` : '') : '') +
      (pinned >= 0 ? ' &nbsp;·&nbsp; ตรึงไว้ (คลิกซ้ำเพื่อปลด)' : '')
    : `${fmtTime(D.t[i])} &nbsp;·&nbsp; ไม่มีข้อมูลโปรตอนช่วงนี้`;
}

/* ---- hover ------------------------------------------------------------ */
let hoverIdx = -1;
function pick(mx, my) {
  let best = -1, bestD = 81;                    // รัศมีจับ 9px
  const gx = Math.floor(mx / CELL), gy = Math.floor(my / CELL);
  for (let a = -1; a <= 1; a++) for (let b = -1; b <= 1; b++) {
    const bucket = hash.get(((gx + a) << 12) ^ (gy + b));
    if (!bucket) continue;
    for (const i of bucket) {
      const dx = sx[i] - mx, dy = sy[i] - my, d = dx * dx + dy * dy;
      /* class ใหญ่กว่าชนะเมื่อระยะใกล้เคียงกัน — X ที่วาดทับอยู่ต้องหยิบได้ */
      const w = d - D.cls[i] * 6;
      if (w < bestD) { bestD = w; best = i; }
    }
  }
  return best;
}

const MON = ['ม.ค.','ก.พ.','มี.ค.','เม.ย.','พ.ค.','มิ.ย.',
             'ก.ค.','ส.ค.','ก.ย.','ต.ค.','พ.ย.','ธ.ค.'];
function fmtTime(min) {
  const d = new Date(EPOCH + min * 60000);
  const p = (v) => String(v).padStart(2, '0');
  return `${d.getUTCDate()} ${MON[d.getUTCMonth()]} ${d.getUTCFullYear()} ` +
         `${p(d.getUTCHours())}:${p(d.getUTCMinutes())} UT`;
}
function fmtPos(lat, lon) {
  const a = Math.round(Math.abs(lat)), o = Math.round(Math.abs(lon));
  return (lat >= 0 ? 'N' : 'S') + String(a).padStart(2, '0') +
         (lon >= 0 ? 'W' : 'E') + String(o).padStart(2, '0');
}

function showTip(i, mx, my) {
  const cls = D.classes[D.cls[i]], st = D.style[D.cls[i]];
  const rows = [
    ['เวลาที่พีค', fmtTime(D.t[i])],
    ['flux 1-8 Å', D.flux[i].toExponential(2) + ' W/m²'],
    ['ตำแหน่ง', fmtPos(D.lat[i], D.lon[i])],
    ['lat / lon', `${D.lat[i].toFixed(1)}° / ${D.lon[i].toFixed(1)}°`],
    ['active region', D.ar[i] ? 'AR' + D.ar[i] : '—'],
    ['ที่มาพิกัด', D.src[i] >= 0 ? D.tierNames[D.src[i]] : '—'],
    ['solar cycle', D.cyc[i]],
  ];
  if (P) {
    const v = fluxAt(EPOCH + D.t[i] * 60000, 0);
    rows.push(['proton >10 MeV', isFinite(v) ? fmtPfu(v) + ' pfu' : 'ไม่มีข้อมูล']);
  }
  if (D.cmeAvailable && D.cme[i]) rows.push(['CME', 'เข้าเกณฑ์ ✓ (ดูเกณฑ์ท้ายหน้า)']);
  if (D.sat[i]) rows.push(['หมายเหตุ', 'เซนเซอร์อิ่มตัว — ค่าจริงสูงกว่านี้']);
  if (D.limb[i]) rows.push(['หมายเหตุ', 'ค่าที่คำนวณได้เลยขอบจาน — หนีบไว้ที่ขอบ']);
  tip.innerHTML =
    `<div class="hd"><span class="dot" style="background:${st.color}"></span>` +
    `${cls}${D.mag[i].toFixed(1)}</div><table>` +
    rows.map(r => `<tr><td>${r[0]}</td><td>${r[1]}</td></tr>`).join('') +
    `</table>`;
  tip.style.opacity = 1;
  const rect = cv.getBoundingClientRect();
  const tw = tip.offsetWidth, th = tip.offsetHeight;
  let left = mx + 16, top = my - th / 2;
  if (left + tw > rect.width) left = mx - tw - 16;
  top = Math.max(0, Math.min(rect.height - th, top));
  tip.style.left = left + 'px';
  tip.style.top = top + 'px';
}

cv.addEventListener('pointermove', (e) => {
  const rect = cv.getBoundingClientRect();
  const mx = e.clientX - rect.left, my = e.clientY - rect.top;
  const i = pick(mx, my);
  if (i !== hoverIdx) {
    hoverIdx = i; render();
    if (pinned < 0) drawChart(i);         // ตรึงอยู่ก็อย่าให้ hover มาทับ
  }
  if (i >= 0) showTip(i, mx, my); else tip.style.opacity = 0;
});
cv.addEventListener('pointerleave', () => {
  hoverIdx = -1; tip.style.opacity = 0; render();
  if (pinned < 0) drawChart(-1);
});
/* คลิกเพื่อตรึงกราฟ — เลื่อนเมาส์ออกแล้วกราฟหายทันทีจะอ่านตัวเลขไม่ทัน */
cv.addEventListener('click', (e) => {
  const rect = cv.getBoundingClientRect();
  const i = pick(e.clientX - rect.left, e.clientY - rect.top);
  if (i < 0) return;
  pinned = (pinned === i) ? -1 : i;
  drawChart(pinned >= 0 ? pinned : i);
});

/* ---- controls --------------------------------------------------------- */
function seg(host, items, isOn, toggle) {
  host.innerHTML = '';
  items.forEach((it) => {
    const b = document.createElement('button');
    b.textContent = it.label;
    if (it.color) { b.className = 'cls'; b.style.setProperty('--c', it.color); }
    b.setAttribute('aria-pressed', isOn(it.key));
    b.onclick = () => { toggle(it.key); refresh(); };
    host.appendChild(b);
  });
}
function refresh() {
  seg(document.getElementById('cycSeg'),
      CYCLES.map(c => ({ key: c, label: 'Cycle ' + c })),
      k => state.cyc.has(k),
      k => { state.cyc.has(k) ? state.cyc.delete(k) : state.cyc.add(k);
             if (!state.cyc.size) state.cyc.add(k); });
  seg(document.getElementById('clsSeg'),
      D.classes.map((c, k) => ({ key: k, label: c + '-class',
                                 color: D.style[k].color })),
      k => state.cls.has(k),
      k => { state.cls.has(k) ? state.cls.delete(k) : state.cls.add(k);
             if (!state.cls.size) state.cls.add(k); });
  if (D.cmeAvailable) {
    document.getElementById('cmeGrp').hidden = false;
    const nCme = D.cme.reduce((a, b) => a + b, 0);
    seg(document.getElementById('cmeSeg'),
        [{ key: 1, label: `เฉพาะที่มี CME (${nCme.toLocaleString('th-TH')})` }],
        () => state.cme,
        () => { state.cme = !state.cme; });
  }
  rebuild(); paintBase(); render();
}

const themeBtn = document.getElementById('themeBtn');
themeBtn.onclick = () => {
  const cur = document.documentElement.getAttribute('data-theme');
  const dark = cur ? cur === 'dark'
    : matchMedia('(prefers-color-scheme: dark)').matches;
  document.documentElement.setAttribute('data-theme', dark ? 'light' : 'dark');
  themeBtn.setAttribute('aria-pressed', String(!dark));
  paintBase(); render(); drawChart(pinned >= 0 ? pinned : hoverIdx);
};

function boot() { layout(); project(); refresh(); layoutChart(); drawChart(-1); }
let rz; window.addEventListener('resize', () => {
  clearTimeout(rz);
  rz = setTimeout(() => {
    layout(); project(); rebuild(); paintBase(); render();
    layoutChart(); drawChart(pinned >= 0 ? pinned : hoverIdx);
  }, 120);
});
boot();
</script>
</body>
</html>
"""
