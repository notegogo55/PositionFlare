#!/usr/bin/env python3
"""
proton_plots.py
===============
ภาพนิ่ง (PNG) ของ integral proton flux — คู่ขนานกับกราฟ interactive ใน
flares_interactive.html สำหรับกรณีที่ต้องการไฟล์ภาพเปิดได้โดยไม่ต้องใช้
เบราว์เซอร์ (แนบอีเมล พิมพ์ วางในเอกสาร ฯลฯ)

แนวคิดใกล้เคียงกับ notebook สำรวจข้อมูล SGPS ที่ดึง flux เชิงอนุพันธ์มาต่อเป็น
integral ทีละช่องพลังงานแล้ว plot เทียบเวลา (log scale, หลายเส้นสี) — ต่างกันตรงที่
โมดูลนี้ใช้ pipeline ที่ตรวจสอบแล้วของ proton_flux.py (อินทิเกรตวิเคราะห์ +
ต่อเนื่องจาก particle/Primary ตั้งแต่ปี 1998) แทนการอ่าน netCDF ดิบทีละเดือน จึง
พล็อตข้ามทั้งภารกิจได้ ไม่ใช่แค่เดือนเดียว และไม่ต้องแยก East/West เพราะ
build_hourly_series() เฉลี่ยสองเซนเซอร์ไว้แล้วตามที่ตกลงกันไว้ตอนออกแบบ pipeline

สองรูปแบบหลัก
--------------
plot_event()     กราฟช่วงสั้น (วัน) รอบเหตุการณ์หนึ่ง อ่านจากไฟล์ 5 นาทีโดยตรง
                  เพื่อความละเอียดสูง ใส่เส้น marker ของ flare ที่ทำให้เกิดได้
plot_overview()  กราฟทั้งภารกิจ (ปี) จาก cache รายชั่วโมง ย่อเป็นค่าสูงสุดรายวัน
                  ให้เห็นภาพรวมว่า solar cycle ไหนมี SEP ถี่/แรงกว่ากัน

find_top_events() หาเหตุการณ์ที่แรงที่สุด N อันดับจากข้อมูลเอง (ไม่ต้อง hardcode
                  วันที่ทางประวัติศาสตร์ ซึ่งอาจผิดพลาดหรือตกหล่นได้)

Usage
-----
  python proton_plots.py --overview out/proton_overview.png
  python proton_plots.py --event 2024-05-09 2024-05-16 out/may2024.png
  python proton_plots.py --top-events 6 --outdir out/proton_events
  python proton_plots.py --selftest
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import proton_flux as pf
from solar_plots import BASELINE, GRIDLINE, MUTED, PRIMARY, SECONDARY, SURFACE, apply_style

# --------------------------------------------------------------------------- #
# design tokens
# --------------------------------------------------------------------------- #
# integral flux เป็นตัวแปรเชิงลำดับ (P>1 ครอบ P>10 ครอบ P>50 ครอบ P>100) เหมือนกับ
# ช่องพลังงานในกราฟ interactive จึงใช้หลักเดียวกัน: ramp ไล่เข้มของสีเดียว ไม่ใช่
# hue คนละสีต่อเส้น — ยิ่งพลังงานสูง เส้นยิ่งเข้ม
CHANNEL_RAMP = {"p1": "#a4ccf4", "p10": "#5598e7", "p30": "#2a78d6",
                "p50": "#1c5cab", "p100": "#124687"}
CHANNEL_LABEL = {"p1": "> 1 MeV", "p5": "> 5 MeV", "p10": "> 10 MeV",
                 "p30": "> 30 MeV", "p50": "> 50 MeV", "p100": "> 100 MeV"}
CHANNEL_WIDTH = {"p1": 2.2, "p5": 1.9, "p10": 1.9, "p30": 1.6, "p50": 1.6, "p100": 1.3}

# ระดับ NOAA S-scale วัดที่ช่อง >10 MeV หน่วย pfu (protons/cm²-s-sr)
S_SCALE = [(10, "S1"), (100, "S2"), (1_000, "S3"), (10_000, "S4"), (100_000, "S5")]

SOURCE_NOTE = ("ที่มา particle/{Primary,GOES16,GOES18,G08..G12}/  —  ช่อง P>1..P>100 "
              "MeV ของ GOES16/18 คำนวณจากสเปกตรัมอนุพันธ์ SGPS (ดู proton_flux.py), "
              "ยุคก่อนหน้าเป็นผลิตภัณฑ์ SWPC โดยตรง")

# หน้าต่างเวลาเริ่มต้นของ plot_event เมื่อเรียกจาก find_top_events — SEP พุ่งเร็ว
# ภายในไม่กี่ชั่วโมงแล้วซาลงเป็นวัน จึงเผื่อก่อนพีคน้อยและหลังพีคมาก
EVENT_BEFORE = pd.Timedelta(days=1)
EVENT_AFTER = pd.Timedelta(days=5)


def _file_for_day(sub: str, day: pd.Timestamp) -> str | None:
    pat = os.path.join(pf.PARTICLE, sub, f"{day:%Y}", f"{day:%Y%m%d}_*_5m.txt")
    hits = glob.glob(pat)
    return hits[0] if hits else None


def load_range(t0, t1) -> pd.DataFrame:
    """รวมไฟล์ 5 นาทีทุกแหล่งในช่วง [t0, t1] -> DataFrame (time, p1..p100)

    เลือกแหล่งของแต่ละวันตามลำดับเดียวกับ build_hourly_series() (SOURCE_ORDER)
    เพื่อให้ plot เดี่ยวกับ cache รายชั่วโมงสอดคล้องกัน ไม่ใช่คนละที่มา
    """
    t0, t1 = pd.Timestamp(t0), pd.Timestamp(t1)
    days = pd.date_range(t0.normalize(), t1.normalize(), freq="D")
    got: dict[str, pd.DataFrame] = {}
    for sub in pf.SOURCE_ORDER:
        for d in days:
            key = d.strftime("%Y%m%d")
            if key in got:
                continue
            p = _file_for_day(sub, d)
            if p:
                df = pf.read_swpc_5m(p)
                if not df.empty:
                    got[key] = df
    if not got:
        return pd.DataFrame(columns=["time"] + pf.PCOLS)
    allf = pd.concat(got.values(), ignore_index=True).sort_values("time")
    m = (allf["time"] >= t0) & (allf["time"] <= t1)
    return allf.loc[m, ["time"] + pf.PCOLS].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# plot 1: event zoom
# --------------------------------------------------------------------------- #
def plot_event(t0, t1, outfile: str, title: str | None = None,
               flares: list[tuple] | None = None,
               channels: list[str] = ("p1", "p10", "p50", "p100")) -> bool:
    """กราฟ integral flux ความละเอียด 5 นาที ช่วง [t0, t1]

    flares: รายการ (timestamp, label) ที่จะตีเส้นแนวตั้งกำกับไว้ — ใช้เชื่อมว่า
    flare ดวงไหนตามมาด้วยพายุโปรตอน คืน False ถ้าช่วงนี้ไม่มีข้อมูลเลย (ไม่เขียนไฟล์)
    """
    t0, t1 = pd.Timestamp(t0), pd.Timestamp(t1)
    df = load_range(t0, t1)
    if df.empty or df[list(channels)].notna().to_numpy().sum() == 0:
        return False

    apply_style()
    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    fig.subplots_adjust(left=0.075, right=0.90, top=0.86, bottom=0.13)

    for c in channels:
        ax.plot(df["time"], df[c], color=CHANNEL_RAMP.get(c, SECONDARY),
                linewidth=CHANNEL_WIDTH.get(c, 1.5), label=CHANNEL_LABEL.get(c, c),
                solid_joinstyle="round", solid_capstyle="round")

    ax.set_yscale("log")
    ax.set_ylim(1e-2, 1e5)
    ax.set_ylabel("Integral proton flux  [pfu = protons/(cm² sr s)]")
    ax.grid(True, which="major", axis="y", color=GRIDLINE, linewidth=0.8)
    ax.grid(True, which="major", axis="x", color=GRIDLINE, linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(length=0, colors=SECONDARY)

    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=9))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))

    # แกนขวา = ระดับ NOAA S-scale เทียบตรงกับสเกลซ้าย ให้อ่านความรุนแรงได้ทันที
    axr = ax.twinx()
    axr.set_yscale("log")
    axr.set_ylim(ax.get_ylim())
    axr.set_yticks([s[0] for s in S_SCALE])
    axr.set_yticklabels([s[1] for s in S_SCALE])
    axr.tick_params(length=0, colors=MUTED)
    for side in ("top", "left", "bottom"):
        axr.spines[side].set_visible(False)
    axr.spines["right"].set_color(BASELINE)

    if flares:
        # เรียงตามเวลาแล้วสลับความสูงป้ายเป็น 3 ชั้น — ช่วงที่มี flare ถี่ (เช่น
        # Halloween 2003 ที่มี X-flare หลายดวงในไม่กี่วัน) ป้ายแนวเดียวจะทับกัน
        # จนอ่านไม่ออก คำนวณตำแหน่ง y เป็นพิกัดข้อมูลจริงจาก ylim ที่ตั้งไว้แล้ว
        # (ไม่ใช้ xaxis_transform/axes-fraction เพราะ matplotlib รุ่นนี้มีบั๊กเวลา
        # ผสมกับแกน y log-scale — BlendedGenericTransform ไปเรียก log() บนค่า
        # fraction ที่ไม่ใช่ scale เดียวกัน)
        shown = sorted({(pd.Timestamp(t), label) for t, label in flares})
        ylo, yhi = ax.get_ylim()
        tiers_y = [10 ** (np.log10(yhi) - d) for d in (0.15, 0.70, 1.25)]
        for i, (t, label) in enumerate(shown):
            if not (df["time"].min() <= t <= df["time"].max()):
                continue
            ax.axvline(t, color=PRIMARY, linewidth=1.0, alpha=0.45, zorder=1)
            ax.annotate(label, (t, tiers_y[i % len(tiers_y)]),
                        xytext=(3, 0), textcoords="offset points",
                        color=PRIMARY, fontsize=8.5, fontweight="semibold",
                        ha="left", va="top")

    leg = ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.10), ncol=len(channels),
                    frameon=False, fontsize=9, handlelength=1.6, columnspacing=1.3)
    for txt in leg.get_texts():
        txt.set_color(SECONDARY)

    ttl = title or f"Integral proton flux — {t0:%d %b %Y} to {t1:%d %b %Y}"
    fig.text(0.075, 0.955, ttl, color=PRIMARY, fontsize=15,
             fontweight="semibold", va="center")
    fig.text(0.075, 0.915, SOURCE_NOTE, color=MUTED, fontsize=8.5, va="center")

    fig.savefig(outfile, dpi=170, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return True


# --------------------------------------------------------------------------- #
# plot 2: full-mission overview
# --------------------------------------------------------------------------- #
def plot_overview(outfile: str, hourly: dict | None = None,
                  channels: list[str] = ("p10", "p100")) -> None:
    """กราฟทั้งภารกิจจาก cache รายชั่วโมง ย่อเป็นค่าสูงสุดรายวัน

    ย่อด้วยค่าสูงสุด ไม่ใช่ค่าเฉลี่ย ด้วยเหตุผลเดียวกับตอนสร้าง cache — เส้นกราฟ
    หลายสิบปีถ้าเฉลี่ยจะกลบพายุ SEP ที่กินเวลาแค่ 2-3 วันจนหายไปเลย
    """
    hourly = hourly if hourly is not None else pf.load_hourly()
    if hourly is None:
        raise RuntimeError(
            "ไม่พบ data_cache/proton_hourly.npz — รัน: python proton_flux.py --build-series")

    apply_style()
    fig, ax = plt.subplots(figsize=(14, 5.2))
    fig.subplots_adjust(left=0.065, right=0.925, top=0.84, bottom=0.12)

    for c in channels:
        if c not in hourly["channels"]:
            continue
        daily = decode_daily_max(hourly, c)
        ax.plot(daily.index, daily.to_numpy(), color=CHANNEL_RAMP.get(c, SECONDARY),
                linewidth=0.9, label=CHANNEL_LABEL.get(c, c),
                solid_joinstyle="round", solid_capstyle="round")

    ax.set_yscale("log")
    ax.set_ylim(1e-2, 1e5)
    ax.set_ylabel("Integral proton flux  [pfu]  (ค่าสูงสุดรายวัน)")
    ax.grid(True, which="major", axis="y", color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(length=0, colors=SECONDARY)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    axr = ax.twinx()
    axr.set_yscale("log")
    axr.set_ylim(ax.get_ylim())
    axr.set_yticks([s[0] for s in S_SCALE])
    axr.set_yticklabels([s[1] for s in S_SCALE])
    axr.tick_params(length=0, colors=MUTED)
    for side in ("top", "left", "bottom"):
        axr.spines[side].set_visible(False)
    axr.spines["right"].set_color(BASELINE)

    leg = ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.12), ncol=len(channels),
                    frameon=False, fontsize=9, handlelength=1.6, columnspacing=1.3)
    for txt in leg.get_texts():
        txt.set_color(SECONDARY)

    t0 = pd.Timestamp(hourly["t0"].replace("Z", ""))
    t1 = t0 + pd.Timedelta(hours=hourly["n"])
    fig.text(0.065, 0.955, "Integral proton flux — ภาพรวมทั้งภารกิจ",
             color=PRIMARY, fontsize=16, fontweight="semibold", va="center")
    fig.text(0.065, 0.915, f"{t0:%b %Y} – {t1:%b %Y}   ·   " + SOURCE_NOTE,
             color=MUTED, fontsize=8.5, va="center")

    fig.savefig(outfile, dpi=170, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def decode_daily_max(hourly: dict, channel: str) -> pd.Series:
    s = pf.decode_hourly(hourly, channel)
    return s.resample("D").max()


# --------------------------------------------------------------------------- #
# หาเหตุการณ์ที่แรงที่สุดจากข้อมูลเอง — ไม่ hardcode วันที่ทางประวัติศาสตร์
# --------------------------------------------------------------------------- #
def find_top_events(hourly: dict | None = None, n: int = 6,
                    min_gap_days: int = 10, min_pfu: float = 10.0,
                    channel: str = "p10") -> list[tuple[pd.Timestamp, float]]:
    """N เหตุการณ์ที่ >10 MeV แรงที่สุด เรียงตามเวลา

    บังคับระยะห่างขั้นต่ำระหว่างพีคที่เลือก มิฉะนั้นพายุลูกเดียวที่แกว่งขึ้นลง
    หลายรอบ (เช่น หาง SEP ที่ยังไม่นิ่ง) จะถูกนับเป็นหลายเหตุการณ์ซ้อนกัน
    """
    hourly = hourly if hourly is not None else pf.load_hourly()
    if hourly is None:
        raise RuntimeError(
            "ไม่พบ data_cache/proton_hourly.npz — รัน: python proton_flux.py --build-series")

    s = pf.decode_hourly(hourly, channel).dropna()
    s = s[s >= min_pfu].sort_values(ascending=False)

    picked: list[tuple[pd.Timestamp, float]] = []
    for t, v in s.items():
        if any(abs((t - u).days) < min_gap_days for u, _ in picked):
            continue
        picked.append((t, float(v)))
        if len(picked) >= n:
            break
    return sorted(picked)


# --------------------------------------------------------------------------- #
# selftest
# --------------------------------------------------------------------------- #
def selftest() -> int:
    import tempfile

    # ---- plot_event บนอนุกรมสังเคราะห์ (ไม่พึ่งไฟล์จริงในเครื่อง) ------------ #
    idx = pd.date_range("2024-05-09", periods=288 * 3, freq="5min")
    df = pd.DataFrame({"time": idx})
    for c in pf.PCOLS:
        df[c] = np.nan
    df["p1"] = np.geomspace(1, 500, len(idx))
    df["p10"] = np.geomspace(0.5, 300, len(idx))
    df["p50"] = np.geomspace(0.2, 20, len(idx))
    df["p100"] = np.geomspace(0.1, 5, len(idx))

    import unittest.mock as mock
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "event.png")
        with mock.patch("proton_plots.load_range", return_value=df):
            ok = plot_event("2024-05-09", "2024-05-12", out,
                            flares=[(pd.Timestamp("2024-05-10"), "X1.1")])
        assert ok, "plot_event ต้องคืน True เมื่อมีข้อมูล"
        assert os.path.getsize(out) > 5_000, "ไฟล์ภาพเล็กผิดปกติ — อาจวาดไม่สำเร็จ"

        # ช่วงที่ไม่มีข้อมูลเลยต้องคืน False และไม่เขียนไฟล์ทับ
        empty = pd.DataFrame(columns=["time"] + pf.PCOLS)
        out2 = os.path.join(tmp, "empty.png")
        with mock.patch("proton_plots.load_range", return_value=empty):
            ok2 = plot_event("1990-01-01", "1990-01-02", out2)
        assert ok2 is False and not os.path.exists(out2)

    # ---- find_top_events: สังเคราะห์ 3 พีคที่รู้คำตอบ ----------------------- #
    n = 24 * 400
    t0 = pd.Timestamp("2020-01-01")
    enc = np.ones((2, n), dtype=np.uint8)          # baseline ต่ำสุด (q=1)
    peaks_at = [50, 150, 250]                       # ชั่วโมงที่ตั้งพีคไว้
    for i, h in enumerate(peaks_at):
        enc[0, h:h + 5] = 200 - i * 10               # พีคลดหลั่นความแรงกัน
    fake = {"t0": t0.strftime("%Y-%m-%dT%H:%M:%SZ"), "step_min": 60, "n": n,
            "channels": ["p10", "p100"], "log_lo": -3.0, "log_hi": 5.0, "enc": enc}
    events = find_top_events(fake, n=5, min_gap_days=1, min_pfu=1.0)
    assert len(events) == 3, events
    got_hours = sorted(int((t - t0).total_seconds() // 3600) for t, _ in events)
    assert all(abs(g - w) <= 1 for g, w in zip(got_hours, peaks_at)), (got_hours, peaks_at)
    # เรียงตามเวลา ไม่ใช่ตามความแรง
    assert [t for t, _ in events] == sorted(t for t, _ in events)

    # gap ถี่เกินไปต้องถูกยุบเหลือพีคเดียว
    close = find_top_events(fake, n=5, min_gap_days=30, min_pfu=1.0)
    assert len(close) == 1, close

    print("selftest: OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[2])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--overview", metavar="OUT")
    ap.add_argument("--event", nargs=3, metavar=("START", "END", "OUT"))
    ap.add_argument("--top-events", type=int, metavar="N")
    ap.add_argument("--outdir", default="out/proton_events")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    apply_style()
    if args.overview:
        plot_overview(args.overview)
        print(f"-> {args.overview}")
    if args.event:
        t0, t1, out = args.event
        ok = plot_event(pd.Timestamp(t0), pd.Timestamp(t1), out)
        print(f"-> {out}" if ok else f"! ไม่มีข้อมูลในช่วง {t0}..{t1}")
    if args.top_events:
        hourly = pf.load_hourly()
        events = find_top_events(hourly, n=args.top_events)
        os.makedirs(args.outdir, exist_ok=True)
        print(f"เหตุการณ์ที่แรงที่สุด {len(events)} จาก {args.top_events} ที่ขอ:")
        for t, v in events:
            out = os.path.join(args.outdir, f"{t:%Y%m%d}_sep_event.png")
            title = f"SEP event {t:%d %b %Y} UT  ·  พีค >10 MeV ~{v:,.0f} pfu"
            ok = plot_event(t - EVENT_BEFORE, t + EVENT_AFTER, out, title=title)
            print(f"  {t:%Y-%m-%d %H:%M}  {v:>9,.0f} pfu  -> {out if ok else '(ไม่มีข้อมูลรายละเอียด)'}")
    if not (args.overview or args.event or args.top_events):
        ap.print_help()
    return 0


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
