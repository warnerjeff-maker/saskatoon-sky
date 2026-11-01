"""
Saskatoon Sky Tonight  (skygazer_v01.py)

A phone-friendly Streamlit app that draws the night sky over Saskatoon:
stars to magnitude 6, constellation figures, the planets, and the Moon,
plus a table of rise/set times and twilight.

Run locally:   streamlit run skygazer_v01.py
Data:          data/stars.csv, data/constellation_lines.csv
               (from the d3-celestial project, BSD licence, derived from Hipparcos)
Ephemerides:   PyEphem (no downloads needed)
"""

from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import ephem
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------
# Location (Saskatoon). Saskatchewan stays on CST (UTC-6) all year.
# --------------------------------------------------------------------------
SITE_NAME = "Saskatoon, SK"
LAT, LON, ELEV_M = 52.1332, -106.6700, 480
TZ = ZoneInfo("America/Regina")

DATA = Path(__file__).parent / "data"

PLANETS = {
    "Moon": (ephem.Moon, "#f2f2e6"),
    "Mercury": (ephem.Mercury, "#b8b8b8"),
    "Venus": (ephem.Venus, "#fff4c2"),
    "Mars": (ephem.Mars, "#ff7a4d"),
    "Jupiter": (ephem.Jupiter, "#ffd9a0"),
    "Saturn": (ephem.Saturn, "#e8d27a"),
    "Uranus": (ephem.Uranus, "#9fe3e8"),
    "Neptune": (ephem.Neptune, "#7fa6ff"),
}


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
@st.cache_data
def load_catalogs():
    stars = pd.read_csv(DATA / "stars.csv")
    lines = pd.read_csv(DATA / "constellation_lines.csv")
    return stars, lines


def make_observer(dt_local: datetime) -> ephem.Observer:
    obs = ephem.Observer()
    obs.lat, obs.lon, obs.elevation = str(LAT), str(LON), ELEV_M
    obs.date = ephem.Date(dt_local.astimezone(ZoneInfo("UTC")).replace(tzinfo=None))
    obs.pressure = 1010  # include atmospheric refraction
    return obs


def to_local(d: ephem.Date) -> datetime:
    return d.datetime().replace(tzinfo=ZoneInfo("UTC")).astimezone(TZ)


# --------------------------------------------------------------------------
# Astronomy helpers
# --------------------------------------------------------------------------
def radec_to_altaz(ra_deg, dec_deg, lst_rad, lat_deg=LAT):
    """Equatorial (deg) -> horizontal (deg). Azimuth measured from N through E."""
    ha = lst_rad - np.radians(ra_deg)
    dec = np.radians(dec_deg)
    lat = np.radians(lat_deg)
    sin_alt = np.sin(dec) * np.sin(lat) + np.cos(dec) * np.cos(lat) * np.cos(ha)
    alt = np.arcsin(np.clip(sin_alt, -1, 1))
    az = np.arctan2(
        -np.cos(dec) * np.sin(ha),
        np.sin(dec) * np.cos(lat) - np.cos(dec) * np.sin(lat) * np.cos(ha),
    )
    return np.degrees(alt), np.degrees(az) % 360


def ecliptic_radec(eps_deg=23.4393):
    lam = np.radians(np.linspace(0, 360, 361))
    eps = np.radians(eps_deg)
    ra = np.degrees(np.arctan2(np.sin(lam) * np.cos(eps), np.cos(lam))) % 360
    dec = np.degrees(np.arcsin(np.sin(eps) * np.sin(lam)))
    return ra, dec


def label_positions(lines: pd.DataFrame) -> pd.DataFrame:
    """Centre of each constellation figure (vector mean of its vertices)."""
    ra, dec = np.radians(lines.ra_deg), np.radians(lines.dec_deg)
    v = pd.DataFrame({
        "name": lines.name,
        "x": np.cos(dec) * np.cos(ra),
        "y": np.cos(dec) * np.sin(ra),
        "z": np.sin(dec),
    }).groupby("name").mean()
    out = pd.DataFrame({
        "ra_deg": np.degrees(np.arctan2(v.y, v.x)) % 360,
        "dec_deg": np.degrees(np.arctan2(v.z, np.hypot(v.x, v.y))),
    })
    return out.reset_index()


def body_positions(obs: ephem.Observer) -> pd.DataFrame:
    rows = []
    for name, (cls, colour) in PLANETS.items():
        b = cls(obs)
        rows.append({
            "name": name,
            "alt": np.degrees(b.alt),
            "az": np.degrees(b.az),
            "mag": b.mag,
            "colour": colour,
        })
    return pd.DataFrame(rows)


def fmt_time(d):
    return to_local(d).strftime("%a %H:%M") if d is not None else "—"


def rise_set_table(night_start: datetime, chart_obs: ephem.Observer) -> pd.DataFrame:
    """Rise/set for each body, searching from 12:00 local on the chosen date."""
    rows = []
    for name, (cls, _) in PLANETS.items():
        obs = make_observer(night_start)
        b = cls()
        try:
            rise = obs.next_rising(b)
            sett = obs.next_setting(b, start=rise)  # the set that follows this rise
            rise_s, set_s = fmt_time(rise), fmt_time(sett)
        except ephem.AlwaysUpError:
            rise_s, set_s = "up all night", ""
        except ephem.NeverUpError:
            rise_s, set_s = "below horizon", ""
        b.compute(chart_obs)
        alt = np.degrees(b.alt)
        rows.append({
            "Object": name,
            "Rises": rise_s,
            "Sets": set_s,
            "Alt at chart time": f"{alt:.0f}°",
            "Brightness (mag)": round(b.mag, 1),
        })
    return pd.DataFrame(rows)


def twilight_info(night_start: datetime):
    obs = make_observer(night_start)
    sun = ephem.Sun()
    sunset = obs.next_setting(sun)
    obs.horizon = "-18"
    try:
        dark_start = obs.next_setting(sun, use_center=True)
        dark_end = obs.next_rising(sun, use_center=True)
    except (ephem.AlwaysUpError, ephem.NeverUpError):
        dark_start = dark_end = None  # no full darkness (midsummer at this latitude)
    obs.horizon = "0"
    sunrise = obs.next_rising(sun, start=sunset)
    return sunset, dark_start, dark_end, sunrise


def moon_phase_text(obs):
    m = ephem.Moon(obs)
    illum = m.phase  # percent illuminated
    nxt_new = ephem.next_new_moon(obs.date)
    nxt_full = ephem.next_full_moon(obs.date)
    waxing = nxt_full < nxt_new
    if illum < 3:
        word = "New Moon"
    elif illum > 97:
        word = "Full Moon"
    elif 45 <= illum <= 55:
        word = "First Quarter" if waxing else "Last Quarter"
    else:
        word = ("Waxing " if waxing else "Waning ") + ("Crescent" if illum < 50 else "Gibbous")
    return word, illum, to_local(nxt_full), to_local(nxt_new)


# --------------------------------------------------------------------------
# Plot
# --------------------------------------------------------------------------
def draw_sky(dt_local, stars, lines, labels, mag_limit, show_lines, show_names, show_ecl):
    obs = make_observer(dt_local)
    lst = float(obs.sidereal_time())

    fig = plt.figure(figsize=(7, 7), dpi=150, facecolor="#05070f")
    ax = fig.add_subplot(projection="polar", facecolor="#0b1230")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(1)  # East on the left: as seen looking up, facing south
    ax.set_rlim(0, 90)
    ax.set_xticks([])
    ax.set_yticks([30, 60])
    ax.set_yticklabels([])
    ax.grid(color="#2a3566", lw=0.5, alpha=0.6)
    ax.spines["polar"].set_color("#6f7fb8")

    # Ecliptic (the planets' highway)
    if show_ecl:
        era, edec = ecliptic_radec()
        ealt, eaz = radec_to_altaz(era, edec, lst)
        r = np.where(ealt > -1, 90 - ealt, np.nan)
        ax.plot(np.radians(eaz), r, ls=(0, (2, 3)), lw=0.8, color="#c9a227", alpha=0.6)

    # Constellation lines
    if show_lines:
        lalt, laz = radec_to_altaz(lines.ra_deg.values, lines.dec_deg.values, lst)
        L = lines.assign(alt=lalt, az=laz)
        for _, seg in L.groupby("segment"):
            if (seg.alt > -5).any():
                r = np.where(seg.alt > -5, np.minimum(90 - seg.alt, 90), np.nan)
                ax.plot(np.radians(seg.az), r, color="#5f8fd6", lw=0.7, alpha=0.75)

    # Stars
    s = stars[stars.mag <= mag_limit]
    salt, saz = radec_to_altaz(s.ra_deg.values, s.dec_deg.values, lst)
    up = salt > 0
    size = 0.6 * (mag_limit + 0.5 - s.mag.values[up]) ** 2.1
    ax.scatter(np.radians(saz[up]), 90 - salt[up], s=size, c="white",
               edgecolors="none", alpha=0.95, zorder=3)

    if show_names:
        named = s[s.name.notna()]
        nalt, naz = radec_to_altaz(named.ra_deg.values, named.dec_deg.values, lst)
        for nm, a, z in zip(named.name, nalt, naz):
            if a > 3:
                ax.annotate(nm, (np.radians(z), 90 - a), xytext=(4, 3),
                            textcoords="offset points", fontsize=6.5, color="#d9e2ff")

        calt, caz = radec_to_altaz(labels.ra_deg.values, labels.dec_deg.values, lst)
        for nm, a, z in zip(labels.name, calt, caz):
            if a > 8:
                ax.text(np.radians(z), 90 - a, nm.upper(), fontsize=5.5,
                        color="#7fa8e8", alpha=0.85, ha="center", va="center")

    # Planets and Moon
    bodies = body_positions(obs)
    for _, b in bodies[bodies.alt > 0].iterrows():
        big = b["name"] == "Moon"
        ax.scatter(np.radians(b.az), 90 - b.alt, s=170 if big else 46,
                   c=b.colour, edgecolors="black", linewidths=0.6, zorder=5)
        ax.annotate(b["name"], (np.radians(b.az), 90 - b.alt), xytext=(6, -9),
                    textcoords="offset points", fontsize=8, color=b.colour,
                    fontweight="bold", zorder=6)

    # Compass points just outside the horizon circle
    for lbl, az in [("N", 0), ("NE", 45), ("E", 90), ("SE", 135),
                    ("S", 180), ("SW", 225), ("W", 270), ("NW", 315)]:
        ax.text(np.radians(az), 97, lbl, ha="center", va="center",
                fontsize=11 if len(lbl) == 1 else 7, color="#ffcf6b",
                fontweight="bold" if len(lbl) == 1 else "normal", clip_on=False)

    ax.set_title(f"{SITE_NAME}  ·  {dt_local:%a %b %d, %Y  %H:%M} CST",
                 color="#e6ebff", fontsize=10, pad=22)
    fig.tight_layout()
    return fig, obs, bodies


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------
st.set_page_config(page_title="Saskatoon Sky", page_icon="✨", layout="centered")
st.markdown(
    "<style>.block-container{padding-top:1.2rem;padding-left:0.6rem;"
    "padding-right:0.6rem}</style>",
    unsafe_allow_html=True,
)
st.markdown("### ✨ Saskatoon Sky Tonight")

stars, lines = load_catalogs()
labels = label_positions(lines)

now = datetime.now(TZ)
# Before 6 am counts as "last night"
default_date = (now - timedelta(days=1)).date() if now.hour < 6 else now.date()

c1, c2 = st.columns([1, 1])
night = c1.date_input("Night of", value=default_date)
times = [time(h, m) for h in list(range(17, 24)) + list(range(0, 8)) for m in (0, 15, 30, 45)]
default_t = time(21, 0)
if night == default_date and (now.hour >= 17 or now.hour < 6):
    default_t = time(now.hour, (now.minute // 15) * 15)
t = c2.select_slider("Time", options=times, value=default_t,
                     format_func=lambda x: x.strftime("%H:%M"))

chart_day = night + timedelta(days=1) if t.hour < 12 else night
dt_local = datetime.combine(chart_day, t, tzinfo=TZ)

with st.expander("Display options"):
    mag_limit = st.slider("Faintest stars shown (magnitude)", 2.0, 6.0, 5.0, 0.5,
                          help="~4 for a lit backyard, ~6 for a dark acreage sky")
    show_lines = st.checkbox("Constellation lines", True)
    show_names = st.checkbox("Names", True)
    show_ecl = st.checkbox("Ecliptic (path of the planets)", True)

fig, obs, bodies = draw_sky(dt_local, stars, lines, labels, mag_limit,
                            show_lines, show_names, show_ecl)
st.pyplot(fig, width="stretch")
plt.close(fig)
st.caption("Hold the phone overhead with the **S** at the bottom facing south. "
           "Centre = straight up; edge = horizon.")

# Moon and twilight
noon = datetime.combine(night, time(12, 0), tzinfo=TZ)
word, illum, full, new = moon_phase_text(make_observer(noon + timedelta(hours=10)))
sunset, dark_start, dark_end, sunrise = twilight_info(noon)

m1, m2 = st.columns(2)
m1.metric("Moon", word, f"{illum:.0f}% lit", delta_color="off")
m2.metric("Fully dark", fmt_time(dark_start).split()[-1] if dark_start else "not tonight",
          f"until {fmt_time(dark_end).split()[-1]}" if dark_end else "", delta_color="off")
st.caption(f"Sunset {fmt_time(sunset)} · Sunrise {fmt_time(sunrise)} · "
           f"Next full Moon {full:%b %d} · Next new Moon {new:%b %d}")

st.markdown("##### Planets & Moon")
st.dataframe(rise_set_table(noon, obs), hide_index=True, width="stretch")

visible = bodies[(bodies.alt > 5) & (bodies.name != "Moon")
                 & ~bodies.name.isin(["Uranus", "Neptune"])]
if len(visible):
    st.success("Naked-eye planets up at chart time: " + ", ".join(visible.name))
else:
    st.info("No bright planets above the horizon at chart time.")

st.caption("Star data: d3-celestial (Hipparcos). Positions: PyEphem. "
           "Lower magnitude = brighter; the naked-eye limit is about 6.")
