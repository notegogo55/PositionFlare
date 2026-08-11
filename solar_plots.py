#!/usr/bin/env python3
"""
solar_plots.py
==============
ชั้นภาพ (figure layer) ของ catalogue flare (C / M / X) — วาดด้วย sunpy/astropy

ทำไมต้อง sunpy
--------------
การพล็อตจานสุริยะด้วยสูตรมือ (x = cos(lat)·sin(lon), y = sin(lat)) เป็นแค่
orthographic projection เปล่า ๆ ไม่มี WCS จริง ไม่มีหน่วยเชิงมุม และต้องวาด grid
เองทีละเส้น ที่นี่จึงสร้าง `sunpy.map.Map` สังเคราะห์ขึ้นมาหนึ่งใบ (จาน
limb-darkened เปล่า ๆ) แล้วให้ sunpy/astropy รับผิดชอบทุกอย่างที่เป็นเรื่องพิกัด:

  * แกน x / y   -> WCSAxes ของ astropy หน่วย Helioprojective arcsec ของจริง
  * grid        -> GenericMap.draw_grid() (Stonyhurst)
  * ขอบดวงอาทิตย์ -> GenericMap.draw_limb()
  * ตำแหน่ง flare -> SkyCoord(HeliographicStonyhurst) -> transform -> world_to_pixel

observer ถูกวางไว้ที่ heliographic latitude 0 พอดี => B0 = 0 by construction
ภาพนี้จึงเป็น "แผนที่ Stonyhurst" ที่ event จากคนละวันวางทับกันได้อย่างมี
ความหมาย (ถ้าใช้ B0 จริงของแต่ละวัน จานจะเอียงไม่เท่ากันทุกจุด)

Design tokens
-------------
สีมาจาก validated categorical palette — slot 1 (น้ำเงิน) คือซีรีส์ข้อมูลเดียว
ของรายงาน ส่วน slot 2 (ส้ม) สงวนไว้เป็น data-quality flag เท่านั้น ผ่าน
all-pairs contrast check ทั้ง CVD และ normal vision บน light surface
"""

from __future__ import annotations

import warnings

import astropy.units as u
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sunpy.map
from astropy.coordinates import SkyCoord
from astropy.time import Time
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import MaxNLocator
from sunpy.coordinates import HeliographicStonyhurst, Helioprojective
from sunpy.map.header_helper import make_fitswcs_header
from sunpy.sun import constants as sunconst

from goes_flare_report import CYCLES, POS_TIER_LABEL, SCALE_LEGACY

# --------------------------------------------------------------------------- #
# design tokens
# --------------------------------------------------------------------------- #
SURFACE = "#fcfcfb"        # chart surface
PRIMARY = "#0b0b0b"        # primary ink
SECONDARY = "#52514e"      # secondary ink
MUTED = "#898781"          # axis / annotation ink
GRIDLINE = "#e1e0d9"       # hairline grid
BASELINE = "#c3c2b7"       # axis rule

SERIES = "#2a78d6"         # slot 1 — X-flare (ซีรีส์เดียวทั้งรายงาน)
FLAG = "#eb6834"           # slot 2 — สงวนไว้สำหรับ data-quality flag เท่านั้น
SERIES_ALT = ("#2a78d6", "#eb6834", "#1baf7a")   # ใช้เฉพาะกราฟเทียบ cycle

# sequential ramp (blue 350 -> 700) สำหรับ magnitude
# ตัดสเต็ปที่อ่อนกว่า 350 ทิ้ง เพราะบนพื้นผิวจานสุริยะจะเหลือ contrast < 2:1
MAG_RAMP = ["#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95",
            "#104281", "#0d366b"]
MAG_CMAP = LinearSegmentedColormap.from_list("xflare_blue", MAG_RAMP)

# จานสุริยะ: warm neutral chroma ต่ำ — บอกว่า "นี่คือดวงอาทิตย์" โดยไม่แย่งสายตา
DISK_CMAP = LinearSegmentedColormap.from_list("quiet_sun", ["#e9d3b4", "#fbf3e8"])
DISK_CMAP = DISK_CMAP.with_extremes(bad=(0, 0, 0, 0))

# ramp ของความหนาแน่นเริ่มอ่อนกว่า MAG_RAMP มาก เพราะเป็นพื้นที่ทึบผืนใหญ่
# ไม่ใช่ marker เม็ดเล็ก — กฎ contrast ขั้นต่ำของ mark จึงไม่ใช้กับกรณีนี้
# ถ้าเริ่มที่ #5598e7 เหมือน marker ช่องที่นับได้ 1 จะเข้มพอ ๆ กับช่องที่นับได้ 100
DENSITY_RAMP = ["#dce9fb", "#b8d2f6", "#8fb8ef", "#66a0e9", "#3987e5",
                "#2a78d6", "#1c5cab", "#124687", "#0d366b"]
DENSITY_CMAP = LinearSegmentedColormap.from_list("flare_density", DENSITY_RAMP)
DENSITY_CMAP = DENSITY_CMAP.with_extremes(bad=(0, 0, 0, 0))    # bin ว่าง = โปร่ง


def density_norm(vmax: float) -> LogNorm:
    """จำนวนต่อช่องมีหางยาว (ช่องเงียบ 1-2 ดวง vs ช่องกลาง 100+) ถ้าใช้สเกล
    เชิงเส้น ทุกช่องยกเว้นยอดจะกองอยู่ปลายอ่อนของ ramp แล้วแบนเป็นสีเดียวกันหมด
    """
    return LogNorm(vmin=1.0, vmax=max(2.0, float(vmax)))


def _density_ticks(vmax: float) -> list[int]:
    return [t for t in (1, 3, 10, 30, 100, 300, 1000) if t <= vmax] or [1]


RSUN_ANGULAR = np.arctan(sunconst.radius / (1 * u.AU)).to(u.arcsec)
FOV_RSUN = 2.25            # ความกว้างภาพจานสุริยะ หน่วย R_sun (เผื่อที่ให้ป้าย N/E/S/W)

# เกินจำนวนนี้ scatter จะกลายเป็นก้อนทึบอ่านไม่ออก (C-class มีถึง ~11,000 ดวง
# ต่อ cycle) จึงสลับไปวาดเป็นความหนาแน่นแทน — เปลี่ยนคำถามจาก "ดวงไหนอยู่ตรงไหน"
# เป็น "กระจุกตรงไหน" ซึ่งเป็นสิ่งเดียวที่ตอบได้เมื่อจุดเยอะขนาดนั้น
DENSITY_MIN = 700

SOURCE_NOTE = ("ที่มา NOAA/NCEI GOES XRS L2 Flare Report (science quality) "
               "— xrsf-l2-flrpt")

# ฟอนต์: ต้องมีสระ/วรรณยุกต์ไทยครบ แล้ว fall back ไป DejaVu สำหรับ ⊙ ≥ ∝
THAI_FONTS = ["Leelawadee UI", "Tahoma", "Noto Sans Thai", "Sarabun"]


def _font_stack() -> list[str]:
    import matplotlib.font_manager as fm
    have = {f.name for f in fm.fontManager.ttflist}
    return [f for f in THAI_FONTS if f in have] + ["DejaVu Sans"]


def apply_style() -> None:
    """rcParams กลาง — chrome จาง ๆ ไม่มีกรอบ ไม่มีเส้นประ"""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": _font_stack(),
        "axes.unicode_minus": False,
        "figure.facecolor": SURFACE,
        "figure.dpi": 110,
        "savefig.facecolor": SURFACE,
        "savefig.dpi": 170,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": BASELINE,
        "axes.linewidth": 0.8,
        "axes.labelcolor": SECONDARY,
        "axes.labelsize": 9.5,
        "axes.labelpad": 7,
        "axes.titlecolor": PRIMARY,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.color": GRIDLINE,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "xtick.color": BASELINE,
        "ytick.color": BASELINE,
        "xtick.labelcolor": SECONDARY,
        "ytick.labelcolor": SECONDARY,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "font.size": 10,
    })


# --------------------------------------------------------------------------- #
# helpers ทั่วไป
# --------------------------------------------------------------------------- #
def scale_note(scale: str) -> str:
    """คำอธิบายว่า magnitude ในภาพอยู่บนสเกลไหน — ต้องมีทุกภาพ"""
    if scale == "science":
        return ("magnitude = irradiance จริง สเกลเดียวกันตลอด 1995 ถึงปัจจุบัน  ·  "
                f"ยุค GOES 1-15 SWPC รายงานค่านี้ × {SCALE_LEGACY} "
                "(28 ต.ค. 2003 X24.6 ที่นี่ คือ X17.2 ที่เคยประกาศ)")
    return (f"magnitude = สเกลรายงานยุค GOES 1-15 (irradiance จริง × {SCALE_LEGACY})  ·  "
            "SWPC เลิกคูณตัวประกอบนี้ราวปี 2020 ค่าของ cycle 25 จึงต่ำกว่าที่ประกาศตอนนั้น")


def marker_size(mag: np.ndarray) -> np.ndarray:
    """พื้นที่ marker แปรตาม magnitude (ไม่ใช่รัศมี) — เล็กสุดยังเกิน 8px ตามสเปก"""
    return 30.0 + 34.0 * np.sqrt(np.clip(np.asarray(mag, float), 0.5, None))


def _mag_ticks(vmax: float) -> list[int]:
    return [t for t in (1, 2, 3, 5, 10, 20, 30, 50) if t <= vmax] or [1]


def class_name(cls: str) -> str:
    return f"{cls}-class"


def _rounded_bars(ax, fig, xs, heights, width, color, *, horizontal=False,
                  max_radius_px=6.0, zorder=3, alpha=1.0):
    """แท่งปลายมน–โคนตัดตรง ตามสเปก mark (rounded data-end, square at baseline)

    FancyBboxPatch มนทั้งสี่มุม จึงยืดแท่งลงไปใต้เส้นฐานเท่ารัศมีพอดี แล้วให้
    ax clip ส่วนที่ล้นทิ้ง — โคนที่มองเห็นจึงเป็นมุมฉาก

    เรขาคณิตของ FancyBboxPatch: rounding_size คือรัศมีในหน่วยแกน x ส่วนแกน y
    ถูกคูณด้วย mutation_aspect ทีหลัง ดังนั้นจะได้มุมกลมจริง (ไม่ใช่วงรี) ต้อง
    ตั้ง aspect = px ต่อหน่วย x / px ต่อหน่วย y
    """
    fig.canvas.draw()                      # ต้องรู้ขนาดจริงของ axes ก่อน
    bb = ax.get_window_extent()
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    px_per_x, px_per_y = bb.width / (x1 - x0), bb.height / (y1 - y0)
    aspect = px_per_x / px_per_y           # ใช้ค่าเดียวกันทั้งแนวตั้ง/แนวนอน

    for pos, val in zip(np.asarray(xs, float), np.asarray(heights, float)):
        if not np.isfinite(val) or val <= 0:
            continue
        if horizontal:
            r_px = min(0.4 * width * px_per_y, 0.42 * val * px_per_x, max_radius_px)
            rx = r_px / px_per_x                       # rounding_size (หน่วยแกน x)
            box = FancyBboxPatch((-rx, pos - width / 2), val + rx, width,
                                 boxstyle=f"round,pad=0,rounding_size={rx}",
                                 mutation_aspect=aspect)
        else:
            r_px = min(0.4 * width * px_per_x, 0.45 * val * px_per_y, max_radius_px)
            rx = r_px / px_per_x
            ry = rx * aspect                           # รัศมีจริงในหน่วยแกน y
            box = FancyBboxPatch((pos - width / 2, -ry), width, val + ry,
                                 boxstyle=f"round,pad=0,rounding_size={rx}",
                                 mutation_aspect=aspect)
        box.set(facecolor=color, edgecolor="none", zorder=zorder, alpha=alpha,
                clip_on=True, clip_path=None)
        ax.add_patch(box)


def _stat_tile(fig, x, y, label, value, note=""):
    """stat tile ในพิกัด figure: label เล็กจาง / ค่าใหญ่หนา / หมายเหตุบรรทัดล่าง"""
    fig.text(x, y, label, color=MUTED, fontsize=8.5, va="bottom", ha="left")
    fig.text(x, y - 0.0225, value, color=PRIMARY, fontsize=21,
             fontweight="semibold", va="top", ha="left")
    if note:
        fig.text(x, y - 0.0455, note, color=SECONDARY, fontsize=8.5,
                 va="top", ha="left")


def _quiet_axes(ax, *, grid_axis="y", count_axis=None):
    ax.set_axisbelow(True)
    ax.grid(True, axis=grid_axis, color=GRIDLINE, linewidth=0.8, linestyle="-")
    ax.tick_params(length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    # จำนวนเหตุการณ์เป็นจำนวนเต็ม — ห้ามให้ locator แจก tick แบบ 12.5
    if count_axis:
        getattr(ax, f"{count_axis}axis").set_major_locator(MaxNLocator(integer=True))


def _located(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(subset=["lat", "lon"])


# --------------------------------------------------------------------------- #
# 1) disk view — sunpy WCS
# --------------------------------------------------------------------------- #
def blank_solar_map(ref_time: Time, npix: int = 900,
                    fov: float = FOV_RSUN) -> sunpy.map.GenericMap:
    """Map สังเคราะห์: จานสุริยะ limb-darkened เปล่า ๆ พร้อม WCS ที่ถูกต้อง

    observer ถูกวางที่ HeliographicStonyhurst(lon=0, lat=0, 1 AU) จึงได้
    B0 = 0 เป๊ะ ๆ และเส้นเมริเดียนกลางตรงกับของโลก
    """
    observer = SkyCoord(0 * u.deg, 0 * u.deg, 1 * u.AU,
                        frame=HeliographicStonyhurst, obstime=ref_time)
    frame = Helioprojective(observer=observer, obstime=ref_time)
    scale = (fov * RSUN_ANGULAR / (npix * u.pix)).to(u.arcsec / u.pix)

    g = (np.arange(npix) - (npix - 1) / 2) * (fov / npix)      # หน่วย R_sun
    r = np.hypot(*np.meshgrid(g, g))
    mu = np.sqrt(np.clip(1 - r ** 2, 0, 1))                    # cos(angle)
    data = np.where(r <= 1.0, mu ** 0.45, np.nan)              # limb darkening

    header = make_fitswcs_header(
        data, SkyCoord(0 * u.arcsec, 0 * u.arcsec, frame=frame),
        scale=u.Quantity([scale, scale]),
        observatory="synthetic", instrument="Stonyhurst", telescope="B0 = 0",
    )
    return sunpy.map.Map(data, header)


def plot_disk(df: pd.DataFrame, outfile: str, cycle: int, cls: str = "X",
              scale: str = "science", grid_step: int = 15) -> None:
    d = _located(df).sort_values("magnitude")   # แรงสุดวาดทับบนสุด
    if d.empty:
        return

    ref_time = Time(f"{pd.to_datetime(d['date']).dt.year.median():.0f}-06-01")
    smap = blank_solar_map(ref_time)

    fig = plt.figure(figsize=(12.2, 9.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.36], wspace=0.05,
                          left=0.07, right=0.985, top=0.875, bottom=0.115)
    ax = fig.add_subplot(gs[0], projection=smap)
    side = fig.add_subplot(gs[1])
    side.set_axis_off()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        smap.plot(axes=ax, cmap=DISK_CMAP, norm=Normalize(vmin=0, vmax=1),
                  interpolation="bilinear", title=False, zorder=0)
        smap.draw_limb(axes=ax, color=BASELINE, linewidth=1.0, zorder=2)
        smap.draw_grid(axes=ax, grid_spacing=grid_step * u.deg, annotate=False,
                       system="stonyhurst", color="#ffffff", linewidth=0.7,
                       alpha=0.85, zorder=1)

    ax.set_facecolor(SURFACE)
    ax.set_aspect("equal", adjustable="box")  # จานต้องกลม ไม่ใช่วงรี
    ax.coords.grid(False)                     # ปิด grid ของ WCSAxes เอง

    # ---- แกน x / y : Helioprojective arcsec จาก WCS ของ sunpy -------------- #
    for cax_, lab in ((ax.coords[0], "Helioprojective longitude  $\\theta_x$  [arcsec]"
                                     "   (E $-$  /  W $+$)"),
                      (ax.coords[1], "Helioprojective latitude  $\\theta_y$  [arcsec]"
                                     "   (S $-$  /  N $+$)")):
        cax_.set_axislabel(lab, color=SECONDARY, fontsize=9.5, minpad=0.9)
        cax_.set_major_formatter("s")
        cax_.set_ticks(spacing=500 * u.arcsec, color=BASELINE, size=3, width=0.8)
        cax_.set_ticklabel(color=SECONDARY, fontsize=9)
    for spine in ax.spines.values():
        spine.set(color=BASELINE, linewidth=0.8)

    # ---- ป้ายบน grid + ทิศ N/E/S/W ----------------------------------------- #
    def to_pix(lon_deg, lat_deg):
        c = SkyCoord(lon_deg * u.deg, lat_deg * u.deg,
                     frame=HeliographicStonyhurst(obstime=ref_time))
        return smap.wcs.world_to_pixel(c.transform_to(smap.coordinate_frame))

    for lat_deg in range(-60, 61, 30):
        if lat_deg == 0:
            continue
        gx, gy = to_pix(4, lat_deg)
        ax.text(gx, gy, f"{lat_deg:+d}°", color="#ffffff", fontsize=8,
                ha="left", va="bottom", zorder=3)
    for lon_deg in (-60, -30, 30, 60):
        gx, gy = to_pix(lon_deg, 3)
        ax.text(gx, gy, f"{lon_deg:+d}°", color="#ffffff", fontsize=8,
                ha="center", va="bottom", zorder=3)

    cen = (smap.data.shape[1] - 1) / 2.0
    r_pix = cen / (FOV_RSUN / 2.0)                      # 1 R_sun เป็นพิกเซล
    for label, (ux, uy) in (("N", (0, 1)), ("S", (0, -1)),
                            ("E", (-1, 0)), ("W", (1, 0))):
        ax.text(cen + ux * r_pix * 1.06, cen + uy * r_pix * 1.06, label,
                color=SECONDARY, fontsize=11, fontweight="semibold",
                ha="center", va="center", zorder=3)

    # ---- ตำแหน่ง flare ------------------------------------------------------ #
    mag = d["magnitude"].to_numpy(float)
    sat = d.get("saturated", pd.Series(False, index=d.index)).fillna(False).to_numpy(bool)
    order = list(np.argsort(mag)[::-1][:3])
    dense = len(d) >= DENSITY_MIN

    if dense:
        # bin ในพิกัด heliographic (ไม่ใช่พิกัดพิกเซล) แล้วค่อยฉายมุมของ bin ผ่าน
        # WCS — ช่องใกล้ขอบจึงถูกบีบตามการฉายจริง ไม่ใช่ถูกบิดเพราะเลือก bin ผิดที่
        step = 5.0
        e_lon = np.arange(-90.0, 90.0 + step, step)
        e_lat = np.arange(-60.0, 60.0 + step, step)
        counts, _, _ = np.histogram2d(d["lon"].to_numpy(float),
                                      d["lat"].to_numpy(float), bins=[e_lon, e_lat])
        gl, gb = np.meshgrid(e_lon, e_lat)                 # (nlat+1, nlon+1)
        gx, gy = to_pix(gl, gb)
        dnorm = density_norm(counts.max())
        mesh = ax.pcolormesh(gx, gy, np.where(counts.T > 0, counts.T, np.nan),
                             cmap=DENSITY_CMAP, norm=dnorm, shading="flat",
                             zorder=4, linewidth=0, rasterized=True)
    else:
        px, py = to_pix(d["lon"].to_numpy(float), d["lat"].to_numpy(float))
        size = marker_size(mag)
        vmax = max(10.0, float(np.nanmax(mag)))
        norm = LogNorm(vmin=1.0, vmax=vmax)
        sc = ax.scatter(px, py, s=size, c=mag, cmap=MAG_CMAP, norm=norm,
                        edgecolors=SURFACE, linewidths=1.1, zorder=5)
        if sat.any():
            ax.scatter(px[sat], py[sat], s=size[sat] + 230, facecolors="none",
                       edgecolors=FLAG, linewidths=1.4, zorder=6)
        # 3 อันดับแรก: ติดเลขบนจาน แล้วไปแจกแจงในแผงข้าง (เลี่ยงป้ายชนกัน)
        for rank, i in enumerate(order, 1):
            ax.annotate(str(rank), (px[i], py[i]), xytext=(10, 10),
                        textcoords="offset points", color=PRIMARY, fontsize=9,
                        fontweight="semibold", ha="center", va="center", zorder=7,
                        bbox=dict(boxstyle="circle,pad=0.22", facecolor=SURFACE,
                                  edgecolor=BASELINE, linewidth=0.8))

    # ---- แผงข้าง: colorbar / ขนาด / ที่มาพิกัด / 3 อันดับแรก ---------------- #
    cax = side.inset_axes([0.13, 0.63, 0.085, 0.33])
    if dense:
        cb = fig.colorbar(mesh, cax=cax)
        dticks = _density_ticks(counts.max())
        cb.set_ticks(dticks)
        cb.set_ticklabels([str(t) for t in dticks])
        cb.ax.minorticks_off()
        side.text(0.13, 0.985, f"จำนวน {class_name(cls)} ต่อช่อง {step:.0f}°×{step:.0f}°",
                  transform=side.transAxes, color=SECONDARY, fontsize=9.5, va="bottom")
    else:
        cb = fig.colorbar(sc, cax=cax)
        ticks = _mag_ticks(vmax)
        cb.set_ticks(ticks)
        cb.set_ticklabels([f"{cls}{t}" for t in ticks])
        cb.ax.minorticks_off()                # LogNorm แถม 4×10⁰ / 6×10⁰ มาให้
        side.text(0.13, 0.985, "GOES peak class", transform=side.transAxes,
                  color=SECONDARY, fontsize=9.5, va="bottom")
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=9, colors=SECONDARY)

    if not dense:
        demo = [m for m in (1, 3, 10, 30) if m <= vmax] or [1]
        handles = [plt.scatter([], [], s=marker_size(np.array([m]))[0],
                               color=MAG_CMAP(float(norm(m))), edgecolors=SURFACE,
                               linewidths=1.1, label=f"{cls}{m}") for m in demo]
        if sat.any():
            handles.append(plt.scatter([], [], s=160, facecolors="none", edgecolors=FLAG,
                                       linewidths=1.5,
                                       label="เซนเซอร์อิ่มตัว (ค่าจริงสูงกว่านี้)"))
        leg = side.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.585),
                          labelspacing=1.45, handletextpad=1.0, borderpad=0,
                          title="ขนาด marker แปรตาม magnitude", alignment="left")
        leg.get_title().set(color=SECONDARY, fontsize=9.5)
        for txt in leg.get_texts():
            txt.set_color(SECONDARY)

    # ---- ที่มาของพิกัด (คำถามหลักของรายงานนี้) ------------------------------ #
    # แท่งยาวตามจำนวน — ผู้อ่านต้องเห็นทันทีว่า cycle ไหนพึ่งแหล่งใดเป็นหลัก
    y = 0.560 if dense else (0.300 if sat.any() else 0.340)
    side.text(0.0, y, f"พิกัดมาจาก  (n = {len(d)})", transform=side.transAxes,
              color=SECONDARY, fontsize=9.5, va="top")
    counts = d["pos_source"].value_counts()
    bar_w = 0.62                                    # ความกว้างเต็มสเกล
    for tier, n in counts.items():
        y -= 0.052
        side.text(0.0, y, POS_TIER_LABEL.get(tier, tier), transform=side.transAxes,
                  color=SECONDARY, fontsize=8.5, va="top")
        side.add_patch(plt.Rectangle((0.0, y - 0.040), bar_w * n / counts.max(), 0.013,
                                     transform=side.transAxes, facecolor=SERIES,
                                     edgecolor="none", clip_on=False))
        side.text(bar_w * n / counts.max() + 0.02, y - 0.0345, f"{n}",
                  transform=side.transAxes, color=PRIMARY, fontsize=8.5,
                  fontweight="semibold", va="center")

    # ---- แผงล่างสุดของ side ------------------------------------------------- #
    y -= 0.100
    if dense:
        # โหมดนี้ไม่มีเลขกำกับบนจาน และ "แรงที่สุด" ของ C/M จะชนเพดาน class
        # เท่ากันหมด (C9.9 สามบรรทัด) จึงบอก AR ที่ผลิตมากที่สุดแทน
        side.text(0.0, y, "AR ที่ผลิตมากที่สุด", transform=side.transAxes,
                  color=SECONDARY, fontsize=9.5, va="top")
        for a, n in df["active_region"].value_counts().head(3).items():
            y -= 0.045
            side.text(0.0, y, f"AR{int(a)}  ·  {n} ดวง", transform=side.transAxes,
                      color=SECONDARY, fontsize=8.5, va="top")
    else:
        side.text(0.0, y, "แรงที่สุดใน cycle นี้", transform=side.transAxes,
                  color=SECONDARY, fontsize=9.5, va="top")
        for rank, i in enumerate(order, 1):
            y -= 0.045
            side.text(0.014, y, f"{rank}", transform=side.transAxes, color=PRIMARY,
                      fontsize=8.5, fontweight="semibold", va="top", ha="center",
                      bbox=dict(boxstyle="circle,pad=0.22", facecolor=SURFACE,
                                edgecolor=BASELINE, linewidth=0.8))
            row = d.iloc[i]
            side.text(0.075, y, f"{row['goes_class']}  ·  "
                                f"{pd.to_datetime(row['date']):%d %b %Y}  ·  "
                                f"{row['position'] or '—'}",
                      transform=side.transAxes, color=SECONDARY, fontsize=8.5, va="top")

    # ---- title block -------------------------------------------------------- #
    n_all, n_pos = len(df), len(d)
    fig.text(0.07, 0.955,
             f"ตำแหน่ง {class_name(cls)} flare บนจานสุริยะ — Solar Cycle {cycle}",
             color=PRIMARY, fontsize=16, fontweight="semibold", va="center")
    fig.text(0.07, 0.918,
             f"{n_pos} จาก {n_all} เหตุการณ์มีพิกัด heliographic ({100*n_pos/n_all:.0f}%)"
             f"  ·  แรงสุด {d['goes_class'].iloc[int(order[0])]}"
             f"  ·  grid Stonyhurst ทุก {grid_step}°",
             color=SECONDARY, fontsize=10, va="center")
    fig.text(0.07, 0.030, SOURCE_NOTE + "   |   " + scale_note(scale),
             color=MUTED, fontsize=8, va="center")
    fig.text(0.07, 0.010,
             "แกนและ grid มาจาก WCS ของ sunpy (Helioprojective-cartesian, TAN)   |   "
             "observer อยู่ที่ heliographic latitude 0 จึงได้ B$_0$ = 0 โดยนิยาม   |   "
             "เหตุการณ์ที่ไม่มีพิกัดส่วนใหญ่เกิดหลังขอบจาน",
             color=MUTED, fontsize=8, va="center")

    fig.savefig(outfile, bbox_inches="tight", pad_inches=0.28)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 2) overview — butterfly + distributions
# --------------------------------------------------------------------------- #
def plot_overview(df: pd.DataFrame, outfile: str, cycle: int, cls: str = "X",
                  scale: str = "science") -> None:
    if df.empty:
        return
    dense = len(df) >= DENSITY_MIN
    full = df.copy()
    full["dt"] = pd.to_datetime(full["time_peak"])
    d = _located(full)

    fig = plt.figure(figsize=(13.2, 12.4))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.15, 0.85, 0.85],
                          width_ratios=[1, 1, 1, 0.62],
                          left=0.062, right=0.972, top=0.845, bottom=0.06,
                          hspace=0.50, wspace=0.34)

    # ---- stat strip (วางด้วยพิกัด figure จะได้ระยะคงที่ไม่ขึ้นกับ gridspec) -- #
    strongest = full.loc[full["magnitude"].idxmax()] if full["magnitude"].notna().any() else None
    ar = full["active_region"].value_counts()
    year_counts = full.groupby(full["dt"].dt.year).size()

    ty = 0.930
    _stat_tile(fig, 0.062, ty, f"{class_name(cls)} flares", f"{len(full)}",
               f"ทั้ง cycle {cycle}")
    _stat_tile(fig, 0.300, ty, "มีพิกัดบนจาน", f"{len(d)}",
               f"{100 * len(d) / len(full):.0f}% ของทั้งหมด")
    if strongest is not None:
        alt = (f"  ·  {strongest['goes_class_legacy']} สเกลเดิม"
               if scale == "science" else "")
        _stat_tile(fig, 0.530, ty, "แรงที่สุด", str(strongest["goes_class"]),
                   f"{pd.to_datetime(strongest['date']):%d %b %Y}"
                   + (f"  ·  AR{int(strongest['active_region'])}"
                      if np.isfinite(strongest["active_region"]) else "") + alt)
    if not ar.empty:
        _stat_tile(fig, 0.760, ty, "AR ที่ผลิตมากสุด", f"AR{int(ar.index[0])}",
                   f"{ar.iloc[0]} เหตุการณ์  ·  ปีพีค {year_counts.idxmax()}"
                   f" ({year_counts.max()} ครั้ง)")
    fig.add_artist(plt.Line2D([0.062, 0.972], [ty + 0.014] * 2,
                              color=GRIDLINE, linewidth=1.0))

    # ---- butterfly ---------------------------------------------------------- #
    ax = fig.add_subplot(gs[0, :3])
    ax.axhspan(-48, 0, color=SERIES, alpha=0.035, lw=0)
    if dense:
        # จุดหลักหมื่นทับกันจนนับไม่ได้ — วาดเป็นความหนาแน่นราย 3 เดือน × 2°
        # ซึ่งเป็นวิธีมาตรฐานของ butterfly diagram อยู่แล้ว
        tnum = mdates.date2num(d["dt"])
        t_edges = np.arange(tnum.min(), tnum.max() + 91.31, 91.31)
        b_edges = np.arange(-48, 49, 2.0)
        h, _, _ = np.histogram2d(tnum, d["lat"].to_numpy(float),
                                 bins=[t_edges, b_edges])
        bmesh = ax.pcolormesh(t_edges, b_edges, np.where(h.T > 0, h.T, np.nan),
                              cmap=DENSITY_CMAP, shading="flat", zorder=3,
                              rasterized=True)
        cb = fig.colorbar(bmesh, ax=ax, pad=0.008, fraction=0.028, aspect=20)
        cb.outline.set_visible(False)
        cb.ax.tick_params(length=0, labelsize=8.5, colors=SECONDARY)
        cb.ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        # หน่วยวางไว้ "เหนือ" แถบสี ไม่ใช่ด้านข้าง — ป้ายแนวตั้งด้านขวาจะยื่นไป
        # ทับฮิสโตแกรมละติจูดที่อยู่ติดกัน
        cb.ax.set_title("ต่อช่อง\n3 ด. × 2°", color=SECONDARY, fontsize=8,
                        pad=6, loc="left")
    else:
        ax.scatter(d["dt"], d["lat"], s=marker_size(d["magnitude"].to_numpy(float)) * 0.8,
                   color=SERIES, alpha=0.72, edgecolors=SURFACE, linewidths=0.9, zorder=3)
    ax.axhline(0, color=BASELINE, linewidth=1.0, zorder=2)
    ax.text(0.005, 0.945, "ซีกเหนือ", transform=ax.transAxes, color=MUTED,
            fontsize=8.5, va="center")
    ax.text(0.005, 0.055, "ซีกใต้", transform=ax.transAxes, color=MUTED,
            fontsize=8.5, va="center")
    ax.set_ylim(-48, 48)
    ax.set_yticks(range(-40, 41, 20))
    ax.set_ylabel("Heliographic latitude  [deg]")
    ax.set_title(f"Butterfly diagram — ละติจูดของ {cls}-flare เลื่อนเข้าหา"
                 "เส้นศูนย์สูตรเมื่อ cycle เดินไป")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    _quiet_axes(ax)

    # ---- latitude marginal -------------------------------------------------- #
    axh = fig.add_subplot(gs[0, 3], sharey=ax)
    counts, edges = np.histogram(d["lat"].dropna(), bins=np.arange(-50, 51, 5))
    axh.set_xlim(0, max(counts.max() * 1.12, 1))
    axh.axhline(0, color=BASELINE, linewidth=1.0, zorder=2)
    axh.set_xlabel("จำนวนเหตุการณ์")
    n_n, n_s = int((d["lat"] > 0).sum()), int((d["lat"] < 0).sum())
    axh.set_title(f"ตามละติจูด — N {n_n} : S {n_s}")
    _quiet_axes(axh, grid_axis="x")
    axh.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    axh.tick_params(labelleft=False)
    _rounded_bars(axh, fig, (edges[:-1] + edges[1:]) / 2, counts, 4.4, SERIES,
                  horizontal=True)

    # ---- yearly counts ------------------------------------------------------ #
    axy = fig.add_subplot(gs[1, :2])
    years = np.arange(year_counts.index.min(), year_counts.index.max() + 1)
    vals = year_counts.reindex(years, fill_value=0).to_numpy()
    axy.set_xlim(years[0] - 0.75, years[-1] + 0.75)
    axy.set_ylim(0, max(vals.max() * 1.18, 1))
    axy.set_xticks(years[::max(1, len(years) // 12)])
    axy.set_ylabel(f"จำนวน {cls}-flare")
    axy.set_title("จำนวนต่อปี")
    _quiet_axes(axy, count_axis="y")
    _rounded_bars(axy, fig, years, vals, 0.58, SERIES)
    peak = int(np.argmax(vals))
    axy.annotate(f"{vals[peak]}", (years[peak], vals[peak]), xytext=(0, 7),
                 textcoords="offset points", ha="center", color=PRIMARY,
                 fontsize=9, fontweight="semibold")

    # ---- top 10 --------------------------------------------------------------- #
    axt = fig.add_subplot(gs[1, 2:])
    if dense:
        # C/M มีเป็นหมื่นดวง 10 อันดับแรกจะไปกองที่เพดานของ class (C9.9 เท่ากันหมด)
        # แท่งยาวเท่ากันสิบแท่ง ไม่บอกอะไรเลย — ถามว่า AR ไหนผลิตเยอะสุดแทน
        prod = full["active_region"].value_counts().head(10).iloc[::-1]
        pv = prod.to_numpy(float)
        py_ = np.arange(len(prod))
        axt.set_ylim(-0.7, len(prod) - 0.3)
        axt.set_xlim(0, pv.max() * 1.22)
        axt.set_yticks(py_, [f"AR{int(a)}" for a in prod.index])
        axt.set_xlabel(f"จำนวน {cls}-flare")
        axt.set_title("10 active region ที่ผลิตมากที่สุด")
        _quiet_axes(axt, grid_axis="x", count_axis="x")
        _rounded_bars(axt, fig, py_, pv, 0.5, SERIES, horizontal=True)
        for y, v in zip(py_, pv):
            axt.text(v + pv.max() * 0.02, y, f"{int(v)}", va="center", ha="left",
                     color=PRIMARY, fontsize=8.5)
    else:
        top = full.nlargest(10, "magnitude").iloc[::-1]
        ypos = np.arange(len(top))
        tvals = top["magnitude"].to_numpy(float)
        axt.set_ylim(-0.7, len(top) - 0.3)
        axt.set_xlim(0, tvals.max() * 1.40)
        axt.set_yticks(ypos, [f"{pd.to_datetime(r['date']):%d %b %Y}"
                              for _, r in top.iterrows()])
        axt.set_xlabel("GOES peak magnitude")
        axt.set_title("10 อันดับที่แรงที่สุดของ cycle")
        _quiet_axes(axt, grid_axis="x")
        _rounded_bars(axt, fig, ypos, tvals, 0.5, SERIES, horizontal=True)
        for y, (_, r) in zip(ypos, top.iterrows()):
            tag = str(r["goes_class"]) + ("*" if r["saturated"] else "")
            if np.isfinite(r["active_region"]):
                tag += f"   AR{int(r['active_region'])}"
            axt.text(r["magnitude"] + tvals.max() * 0.028, y, tag, va="center",
                     ha="left", color=PRIMARY, fontsize=8.5)
        if top["saturated"].any():
            axt.text(0.99, 0.02, "* เซนเซอร์อิ่มตัว — ค่าจริงสูงกว่าที่แสดง",
                     transform=axt.transAxes, color=MUTED, fontsize=8, ha="right")

    # ---- central meridian distance ------------------------------------------ #
    axc = fig.add_subplot(gs[2, :2])
    ccounts, cedges = np.histogram(d["lon"].dropna(), bins=np.arange(-90, 91, 10))
    axc.set_xlim(-95, 95)
    axc.set_ylim(0, max(ccounts.max() * 1.18, 1))
    axc.set_xticks(range(-90, 91, 30))
    axc.set_xlabel("Central meridian distance  [deg]   (E $-$  /  W $+$)")
    axc.set_ylabel(f"จำนวน {cls}-flare")
    axc.set_title("การกระจายตามลองจิจูด — ขอบจานถูกรายงานต่ำกว่าความเป็นจริง")
    _quiet_axes(axc, count_axis="y")
    _rounded_bars(axc, fig, (cedges[:-1] + cedges[1:]) / 2, ccounts, 9.3, SERIES)

    # ---- Carrington longitude ----------------------------------------------- #
    axr = fig.add_subplot(gs[2, 2:])
    axr.set_xlabel("Carrington longitude  [deg]")
    axr.set_ylabel(f"จำนวน {cls}-flare")
    axr.set_title("Active longitude — พิกัดที่หมุนไปกับดวงอาทิตย์ (คำนวณด้วย sunpy)")
    axr.set_xlim(-8, 368)
    axr.set_xticks(range(0, 361, 60))
    _quiet_axes(axr, count_axis="y")
    if "carr_lon" in d.columns and d["carr_lon"].notna().any():
        rcounts, redges = np.histogram(d["carr_lon"].dropna(), bins=np.arange(0, 361, 20))
        axr.set_ylim(0, max(rcounts.max() * 1.18, 1))
        mean = float(rcounts.mean())
        axr.axhline(mean, color=BASELINE, linewidth=1.0, zorder=2)
        axr.text(0.99, 0.97, f"เส้นแนวนอน = ค่าเฉลี่ย {mean:.1f} ต่อช่วง 20°",
                 transform=axr.transAxes, color=MUTED, fontsize=8,
                 va="top", ha="right", zorder=6,
                 bbox=dict(facecolor=SURFACE, edgecolor="none", pad=2.0))
        _rounded_bars(axr, fig, (redges[:-1] + redges[1:]) / 2, rcounts, 18.7, SERIES)
    else:
        axr.text(0.5, 0.5, "ไม่มีข้อมูลพิกัดพอจะคำนวณ", transform=axr.transAxes,
                 color=MUTED, fontsize=9, ha="center", va="center")

    # ---- title block -------------------------------------------------------- #
    fig.text(0.062, 0.985,
             f"ภาพรวม {class_name(cls)} flare — Solar Cycle {cycle}",
             color=PRIMARY, fontsize=17, fontweight="semibold", va="center")
    fig.text(0.062, 0.962,
             f"{pd.to_datetime(full['date']).min():%b %Y} – "
             f"{pd.to_datetime(full['date']).max():%b %Y}  ·  "
             + SOURCE_NOTE + "  ·  "
             + ("ความเข้มใน butterfly แปรตามจำนวนเหตุการณ์"
                if dense else "ขนาดจุดใน butterfly แปรตาม magnitude"),
             color=SECONDARY, fontsize=10, va="center")
    fig.text(0.062, 0.012, scale_note(scale), color=MUTED, fontsize=8, va="center")

    fig.savefig(outfile, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 3) cross-cycle comparison
# --------------------------------------------------------------------------- #
def plot_cycle_comparison(frames: dict[int, pd.DataFrame], outfile: str,
                          cls: str = "X", incomplete: set[int] | None = None,
                          scale: str = "science") -> None:
    frames = {c: f for c, f in frames.items() if not f.empty}
    if len(frames) < 2:
        return
    incomplete = incomplete or set()
    cycles = sorted(frames)

    fig = plt.figure(figsize=(14.6, 5.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.52, 1, 0.62], left=0.055,
                          right=0.985, top=0.775, bottom=0.21, wspace=0.26)

    # ---- totals ------------------------------------------------------------- #
    ax = fig.add_subplot(gs[0])
    totals = np.array([len(frames[c]) for c in cycles], float)
    ax.set_xlim(-0.7, len(cycles) - 0.3)
    ax.set_ylim(0, totals.max() * 1.22)
    ax.set_xticks(range(len(cycles)),
                  [f"Cycle {c}" + (" *" if c in incomplete else "") for c in cycles])
    ax.set_ylabel(f"จำนวน {cls}-flare")
    ax.set_title("จำนวนรวมต่อ cycle")
    _quiet_axes(ax, count_axis="y")
    _rounded_bars(ax, fig, np.arange(len(cycles)), totals, 0.5, SERIES)
    for i, v in enumerate(totals):
        ax.text(i, v + totals.max() * 0.035, f"{int(v)}", ha="center",
                color=PRIMARY, fontsize=10, fontweight="semibold")

    # ---- yearly profile, aligned to cycle start ----------------------------- #
    # ต้องนับจาก "จุดเริ่ม cycle" ที่นิยามไว้ ไม่ใช่ปีของ X-flare ดวงแรก มิฉะนั้น
    # cycle ที่เงียบตอนต้น (24 เริ่ม 2008 แต่ X แรกปี 2010) จะถูกเลื่อนมาชิดซ้าย
    # แล้วดูเหมือนพุ่งขึ้นเร็วกว่าความจริง
    axp = fig.add_subplot(gs[1])
    span_max = 0
    ends: list[tuple[float, float, str, str]] = []      # (x, y, label, color)
    for i, c in enumerate(cycles):
        f = frames[c]
        start, end = CYCLES[c]
        elapsed = ((pd.to_datetime(f["date"]) - pd.Timestamp(start)).dt.days
                   // 365.25).astype(int)
        span = np.arange(0, int((pd.Timestamp(end) - pd.Timestamp(start)).days
                                // 365.25) + 1)
        span_max = max(span_max, int(span[-1]))
        vals = elapsed.value_counts().reindex(span, fill_value=0).to_numpy()
        color = SERIES_ALT[i % len(SERIES_ALT)]
        axp.plot(span, vals, color=color, linewidth=2.0, solid_joinstyle="round",
                 solid_capstyle="round", label=f"Cycle {c}", zorder=3)
        axp.scatter(span[-1:], vals[-1:], s=44, color=color, edgecolors=SURFACE,
                    linewidths=2.0, zorder=4)
        ends.append((float(span[-1]), float(vals[-1]),
                     f"Cycle {c}" + (" *" if c in incomplete else ""), color))

    # ป้ายท้ายเส้นวางไว้ "เหนือ" จุดสุดท้าย ไม่ใช่ด้านขวา เพราะ cycle ที่จบแล้ว
    # ลงไปแตะศูนย์เหมือนกันหมด ป้ายแนวขวาจะทับกัน ส่วนแนวตั้งแยกกันด้วยแกน x อยู่แล้ว
    for ex, ey, lab, _ in ends:
        axp.annotate(lab, (ex, ey), xytext=(0, 12), textcoords="offset points",
                     color=SECONDARY, fontsize=9, ha="center", va="bottom")
    axp.set_xlabel("ปีที่นับจากจุดต่ำสุด (solar minimum) ของ cycle")
    axp.set_ylabel(f"จำนวน {cls}-flare ต่อปี")
    axp.set_title("รูปทรงของ cycle เมื่อจัดให้จุดเริ่มตรงกัน")
    axp.set_xlim(-0.4, span_max + 0.5)
    axp.set_xticks(range(0, span_max + 1))
    _quiet_axes(axp, count_axis="y")

    # ---- position coverage — ข้อจำกัดที่ต้องบอกก่อนเทียบตำแหน่งข้าม cycle --- #
    axc = fig.add_subplot(gs[2])
    cover = np.array([100 * _located(frames[c]).shape[0] / len(frames[c])
                      for c in cycles])
    axc.set_xlim(-0.7, len(cycles) - 0.3)
    axc.set_ylim(0, 118)
    axc.set_yticks(range(0, 101, 25))
    axc.set_xticks(range(len(cycles)), [f"Cycle {c}" for c in cycles])
    axc.set_ylabel("% ที่มีพิกัด")
    axc.set_title("สัดส่วนที่ระบุตำแหน่งได้")
    _quiet_axes(axc)
    _rounded_bars(axc, fig, np.arange(len(cycles)), cover, 0.5, SERIES)
    for i, v in enumerate(cover):
        axc.text(i, v + 4, f"{v:.0f}%", ha="center", color=PRIMARY,
                 fontsize=10, fontweight="semibold")

    # ยิ่ง class เล็ก ยิ่งพึ่งการสังเกตภาคพื้น: C-class ของ cycle 23 ระบุตำแหน่ง
    # ได้ ~38% แต่ cycle 25 เกือบ 100% เพราะ XRS ของ GOES-16 ให้พิกัดแทบทุกดวง
    # ตัวเลข "จำนวนรวม" ยังเทียบกันได้ แต่ "การกระจายตำแหน่ง" ของ cycle เก่าเป็น
    # ตัวอย่างที่ถูกคัดมาแล้ว จึงต้องเตือนตรงนี้ ไม่ใช่ปล่อยให้ไปสรุปผิดเอง
    gap = float(cover.max() - cover.min())
    warn = gap > 20.0
    if warn:
        axc.text(0.0, -0.13, "ช่องว่างนี้ทำให้เทียบ *การกระจายตำแหน่ง* ข้าม cycle "
                             "ตรง ๆ ไม่ได้\ncycle เก่าเห็นเฉพาะดวงที่ภาคพื้นสังเกตทัน",
                 transform=axc.transAxes, color=FLAG, fontsize=8.5, va="top")

    fig.text(0.055, 0.945, f"เปรียบเทียบข้าม Solar Cycle — {class_name(cls)} flare",
             color=PRIMARY, fontsize=16, fontweight="semibold", va="center")
    note = "* cycle ยังไม่จบ — ตัวเลขยังเพิ่มได้อีก   ·   " if incomplete else ""
    fig.text(0.055, 0.905,
             note + "จำนวนและ magnitude มาจากไฟล์เดียวกันและสเกล irradiance เดียวกัน "
             "จึงเทียบกันได้โดยตรง"
             + ("   ·   แต่สัดส่วนที่ระบุตำแหน่งได้ต่างกันมาก (ดูแผงขวา)"
                if warn else ""),
             color=SECONDARY, fontsize=9.5, va="center")
    fig.text(0.055, 0.028, SOURCE_NOTE + "   |   " + scale_note(scale),
             color=MUTED, fontsize=8, va="center")

    fig.savefig(outfile, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
