#!/usr/bin/env python3
"""
make_report.py
==============
สร้าง catalogue + รูป ของ flare class C / M / X ใน Solar Cycle 23 / 24 / 25
จากแหล่งข้อมูลเดียว: NOAA/NCEI GOES XRS L2 Flare Report (science quality)

    https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/
        goes/multi/l2/data/xrsf-l2-flrpt_science/

ผลลัพธ์  ({k} = c / m / x, {c} = 23 / 24 / 25)
-------
  out/cycle{c}/{k}_flares_cycle{c}.csv           catalogue รายเหตุการณ์
  out/cycle{c}/{k}_flares_cycle{c}_disk.png      ตำแหน่งบนจานสุริยะ
  out/cycle{c}/{k}_flares_cycle{c}_overview.png  butterfly + การกระจาย
  out/cycle_comparison_{k}.png                   เทียบข้าม cycle
  out/flares_all_cycles.csv                      รวมทุก class ทุก cycle
  out/summary_all_cycles.csv                     สรุปต่อ class ต่อ cycle
  out/flares_interactive.html                    จานสุริยะ interactive + กราฟโปรตอน
  out/proton_overview.png                        integral proton flux ทั้งภารกิจ (นิ่ง)
  out/proton_events/{YYYYMMDD}_sep_event.png     SEP event ที่แรงสุด N อันดับ (นิ่ง)

C-class มีเป็นหมื่นดวงต่อ cycle เกิน DENSITY_MIN ของ solar_plots ภาพจึงสลับไป
วาดเป็นความหนาแน่นแทน scatter ให้อัตโนมัติ

ภาพ proton_* ต้องมี data_cache/proton_hourly.npz ก่อน (สร้างด้วย
python proton_flux.py --convert --build-series) ไม่งั้นจะถูกข้ามไปเฉย ๆ

Usage
-----
  python make_report.py                      # C M X × 3 cycle, สเกล science
  python make_report.py --classes X          # เฉพาะ X-class
  python make_report.py --cycles 24 25
  python make_report.py --scale legacy       # magnitude แบบที่ SWPC เคยรายงาน
  python make_report.py --offline            # ใช้เฉพาะไฟล์ที่ cache ไว้แล้ว
  python make_report.py --selftest
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

import goes_flare_report as gfr
import interactive_map
import proton_flux
import proton_plots
import solar_plots

# ให้สระ/วรรณยุกต์ไทยออกถูกต้องบน console ของ Windows
for _stream in (sys.stdout, sys.stderr):
    if _stream.encoding and _stream.encoding.lower() != "utf-8":
        _stream.reconfigure(encoding="utf-8", errors="replace")


# --------------------------------------------------------------------------- #
def report(x: pd.DataFrame, cycle: int, cls: str, scale: str) -> dict:
    d0, d1 = gfr.CYCLES[cycle]
    n_pos = int(x["lat"].notna().sum())
    print(f"\n--- Cycle {cycle}  ·  {cls}-class  ({d0} .. {d1}) ---")
    print(f"  {cls}-class flares ทั้งหมด  : {len(x)}")
    if not len(x):
        return {"cycle": cycle, "class": cls, "start": d0, "end": d1, "n_flare": 0,
                "n_with_position": 0, "pct_with_position": np.nan,
                "strongest": "", "strongest_date": "", "top_ar": np.nan}

    print(f"  ระบุตำแหน่งได้          : {n_pos}  ({100*n_pos/len(x):.1f} %)")
    print(f"  ไม่มีตำแหน่ง            : {len(x)-n_pos}  (ส่วนใหญ่หลังขอบจาน)")
    print("  ที่มาของพิกัด:")
    for tier, n in x.loc[x["pos_source"] != "", "pos_source"].value_counts().items():
        print(f"    {n:>4d}  {tier:<8s} {gfr.POS_TIER_LABEL.get(tier, '')}")

    print("  จำนวนต่อปี:")
    print("   ", x.groupby(pd.to_datetime(x["date"]).dt.year).size()
                 .to_string().replace("\n", "\n    "))

    ar = x["active_region"].value_counts()
    if not ar.empty:
        print(f"  AR ที่ผลิต {cls}-flare มากสุด:")
        for a, n in ar.head(5).items():
            print(f"    AR{int(a)}  {n} ครั้ง")

    top = x.nlargest(5, "magnitude")
    print("  แรงสุด 5 อันดับ:")
    print("   ", top[["date", "goes_class", "goes_class_legacy", "position",
                      "pos_source", "active_region", "saturated"]]
          .to_string(index=False).replace("\n", "\n    "))

    best = top.iloc[0]
    return {
        "cycle": cycle, "class": cls, "start": d0, "end": d1,
        "n_flare": len(x), "n_with_position": n_pos,
        "pct_with_position": round(100 * n_pos / len(x), 1),
        "strongest": best["goes_class"],
        "strongest_legacy": best["goes_class_legacy"],
        "strongest_date": best["date"],
        "top_ar": ar.index[0] if not ar.empty else np.nan,
        "scale": scale,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="catalogue + รูป X-class flare จาก GOES XRS L2 Flare Report")
    ap.add_argument("--cycles", nargs="+", type=int, default=[23, 24, 25],
                    choices=sorted(gfr.CYCLES))
    ap.add_argument("--classes", nargs="+", default=["C", "M", "X"],
                    choices=["C", "M", "X"], type=str.upper,
                    help="class ที่จะออกรายงาน (ค่าเริ่มต้นทั้งสาม)")
    ap.add_argument("--scale", default="science", choices=sorted(gfr.SCALES),
                    help="science = irradiance จริง (ค่าเริ่มต้น); "
                         "legacy = สเกลที่ SWPC เคยรายงานยุค GOES 1-15")
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--offline", action="store_true",
                    help="ไม่ต่อเน็ต ใช้ไฟล์ใน data_cache/ ที่โหลดไว้แล้ว")
    ap.add_argument("--no-ar-fill", dest="ar_fill", action="store_false",
                    help="ปิด tier 4 — ใช้เฉพาะพิกัดที่มีคนรายงานตรง ๆ เท่านั้น")
    ap.add_argument("--no-html", action="store_true",
                    help="ไม่ต้องสร้างหน้า interactive")
    ap.add_argument("--no-proton", action="store_true",
                    help="ไม่ต้องฝังกราฟ proton flux ในหน้า interactive")
    ap.add_argument("--no-proton-plots", action="store_true",
                    help="ไม่ต้องสร้างภาพนิ่ง (PNG) ของ proton flux")
    ap.add_argument("--proton-events", type=int, default=6, metavar="N",
                    help="จำนวนเหตุการณ์ SEP ที่แรงสุดที่จะพล็อตแยกภาพ (ค่าเริ่มต้น 6)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return gfr.selftest()

    print("=== โหลดข้อมูล GOES XRS L2 Flare Report (science quality) ===")
    # โหลด/แปลงพิกัดรอบเดียวสำหรับ class ที่ต่ำสุดที่ขอ แล้วค่อยแบ่งทีหลัง
    # (C ครอบ M กับ X อยู่แล้ว — เรียก build_catalog สาม รอบจะเสียเวลาเปล่า)
    lowest = min(args.classes, key=gfr.CLASS_ORDER.index)
    cat = gfr.build_catalog(scale=args.scale, cls=lowest, offline=args.offline,
                            ar_fill=args.ar_fill)
    # ไฟล์ต้นทางเริ่มปี 1995 จึงมี event หางของ cycle 22 ติดมาด้วย — ตัดทิ้ง
    # เพื่อให้ไฟล์รวมมีแถวเท่ากับผลบวกของสาม cycle พอดี
    outside = int(cat["cycle"].isna().sum())
    cat = cat[cat["cycle"].notna()].copy()
    cat["cycle"] = cat["cycle"].astype(int)
    per_cls = cat["class_letter"].value_counts().reindex(args.classes, fill_value=0)
    print(f"\nได้ flare {len(cat)} เหตุการณ์ใน cycle 23-25 "
          f"({cat['date'].min()} .. {cat['date'].max()})   สเกล = {args.scale}"
          + (f"   [ตัด {outside} เหตุการณ์ก่อน cycle 23 ทิ้ง]" if outside else ""))
    print("  " + "   ".join(f"{k}: {v:,}" for k, v in per_cls.items()))

    solar_plots.apply_style()
    os.makedirs(args.outdir, exist_ok=True)
    summaries = []

    for cls in args.classes:
        frames = {}
        for c in args.cycles:
            x = gfr.cycle_slice(cat, c, cls)
            sub = os.path.join(args.outdir, f"cycle{c}")
            os.makedirs(sub, exist_ok=True)
            stem = f"{cls.lower()}_flares_cycle{c}"
            x.to_csv(os.path.join(sub, f"{stem}.csv"), index=False)
            if not x.empty:
                frames[c] = x
                solar_plots.plot_disk(x, os.path.join(sub, f"{stem}_disk.png"),
                                      c, cls=cls, scale=args.scale)
                solar_plots.plot_overview(x, os.path.join(sub, f"{stem}_overview.png"),
                                          c, cls=cls, scale=args.scale)
            summaries.append(report(x, c, cls, args.scale))
            print(f"  -> {sub}/{stem}.*")

        if len(frames) > 1:
            solar_plots.plot_cycle_comparison(
                frames, os.path.join(args.outdir, f"cycle_comparison_{cls.lower()}.png"),
                cls=cls, incomplete={c for c in frames if gfr.CYCLES[c][1] >= gfr.TODAY},
                scale=args.scale)

    cat.to_csv(os.path.join(args.outdir, "flares_all_cycles.csv"), index=False)

    # หน้า interactive ครอบทุก class ที่ขอไว้ในไฟล์เดียว (กรองเอาในหน้าเว็บ)
    if not args.no_html:
        # cache โปรตอนสร้างจาก `python proton_flux.py --convert --build-series`
        # ถ้ายังไม่มีก็ข้ามไป หน้าเว็บยังใช้งานได้ปกติ แค่ไม่มีแผงกราฟ
        series = None if args.no_proton else proton_flux.load_hourly()
        if series is None and not args.no_proton:
            print("\n  ! ไม่พบ data_cache/proton_hourly.npz -> หน้า interactive จะไม่มีกราฟโปรตอน"
                  "\n    สร้างด้วย: python proton_flux.py --convert --build-series")
        sub = cat[cat["class_letter"].isin(args.classes)]
        # เกณฑ์ CME/SEP ต้องคำนวณบน sub เฟรมเดียวกับที่ส่งเข้า write_html() เป๊ะ ๆ
        # (เรียงแถวและจำนวนแถวต้องตรงกัน — ดูหมายเหตุใน interactive_map._payload)
        cme_flags = proton_flux.flag_cme(sub["time_peak"], series) if series is not None else None
        page = interactive_map.write_html(
            sub, os.path.join(args.outdir, "flares_interactive.html"),
            scale=args.scale, proton_series=series, cme_flags=cme_flags)
        extra = ""
        if series is not None:
            n_cme = int(cme_flags.sum())
            extra = f"  (มีกราฟโปรตอน · เข้าเกณฑ์ CME {n_cme:,}/{len(sub):,})"
        print(f"\nหน้า interactive: {page}  ({os.path.getsize(page)/1e6:.1f} MB){extra}")

        # ภาพนิ่ง (PNG) ของ proton flux — คู่ขนานกับกราฟ interactive สำหรับกรณีที่
        # ต้องการไฟล์เปิดได้โดยไม่ใช้เบราว์เซอร์ ใช้ cache รายชั่วโมงชุดเดียวกัน
        if series is not None and not args.no_proton_plots:
            print("\n=== ภาพนิ่ง proton flux ===")
            proton_plots.apply_style()
            ov = os.path.join(args.outdir, "proton_overview.png")
            proton_plots.plot_overview(ov, hourly=series)
            print(f"  -> {ov}")

            events = proton_plots.find_top_events(series, n=args.proton_events)
            edir = os.path.join(args.outdir, "proton_events")
            os.makedirs(edir, exist_ok=True)
            # ผูก flare X/M ที่อยู่ในหน้าต่างเวลาของแต่ละ SEP event เป็น marker บนกราฟ
            # ตอบคำถามว่า "flare ดวงไหนเป็นต้นเหตุ" โดยไม่ต้องเดา
            markable = cat[cat["class_letter"].isin(["M", "X"])]
            for t, v in events:
                t0 = t - proton_plots.EVENT_BEFORE
                t1 = t + proton_plots.EVENT_AFTER
                win = markable[(pd.to_datetime(markable["time_peak"]) >= t0)
                              & (pd.to_datetime(markable["time_peak"]) <= t)]
                # ช่วงที่ flare ถี่ (เช่น Halloween 2003) มีได้เป็นสิบดวง — เก็บแค่
                # ที่แรงสุด 5 ดวง ไม่งั้นป้ายบนกราฟจะรกจนอ่านไม่ออกแม้จะสลับชั้นแล้ว
                win = win.nlargest(5, "magnitude")
                flares = [(pd.Timestamp(r["time_peak"]), str(r["goes_class"]))
                         for _, r in win.iterrows()]
                out = os.path.join(edir, f"{t:%Y%m%d}_sep_event.png")
                title = f"SEP event {t:%d %b %Y} UT  ·  พีค >10 MeV ~{v:,.0f} pfu"
                if proton_plots.plot_event(t0, t1, out, title=title, flares=flares):
                    print(f"  -> {out}")
            print(f"  รวม {len(events)} เหตุการณ์ใน {edir}/")

    s = pd.DataFrame(summaries)
    s.to_csv(os.path.join(args.outdir, "summary_all_cycles.csv"), index=False)
    print("\n=== สรุปข้ามทุก cycle ===")
    print(s.to_string(index=False))
    if args.scale == "science":
        print(f"\nหมายเหตุ magnitude เป็น irradiance จริง สเกลเดียวกันตลอด 1995→ปัจจุบัน"
              f"\n  ยุค GOES 1-15 SWPC รายงานค่านี้ × {gfr.SCALE_LEGACY} เช่น 28 ต.ค. 2003 "
              f"X24.6 ที่นี่ = X17.2 ที่เคยประกาศ (ดูคอลัมน์ *_legacy ใน CSV)"
              f"\n  ตั้งแต่ราวปี 2020 SWPC เลิกคูณตัวประกอบนี้ ค่าของ cycle 25 จึงตรงกับ"
              f" magnitude ในตารางนี้อยู่แล้ว")
    return 0


if __name__ == "__main__":
    sys.exit(main())
