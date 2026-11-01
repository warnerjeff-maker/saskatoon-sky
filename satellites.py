"""
Satellite passes for a fixed observer.

Orbit data: CelesTrak "GP" element sets (TLE format), fetched at runtime.
Propagation: SGP4 (sgp4 package), vectorized over satellites and times.
A pass is listed when the satellite is above MIN_ALT, lit by the Sun, and
the observer's sky is dark (Sun more than 6 degrees below the horizon).
"""

import urllib.request
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from sgp4.api import Satrec, SatrecArray, jday

CELESTRAK = "https://celestrak.org/NORAD/elements/gp.php?GROUP={}&FORMAT=tle"
GROUPS = {"stations": "Space stations", "visual": "Brightest satellites",
          "starlink": "Starlink"}

R_EARTH = 6378.137          # km, WGS84 equatorial
F_WGS84 = 1 / 298.257223563
AU_KM = 149_597_870.7

# Approximate standard magnitudes (1000 km range, half-lit). Rough by nature.
STD_MAG = {"ISS": -1.8, "STARLINK": 5.5}


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def fetch_group(group: str, timeout=25) -> list[tuple[str, str, str]]:
    req = urllib.request.Request(CELESTRAK.format(group),
                                 headers={"User-Agent": "saskatoon-sky/0.2"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        text = r.read().decode("utf-8", "replace")
    rows = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    out = []
    for i in range(0, len(rows) - 2, 3):
        name, l1, l2 = rows[i:i + 3]
        if l1.startswith("1 ") and l2.startswith("2 "):
            out.append((name.strip(), l1, l2))
    if not out:
        raise ValueError(f"No orbit data returned for '{group}'")
    return out


def tle_frame(tles: list[tuple[str, str, str]], group: str) -> pd.DataFrame:
    df = pd.DataFrame(tles, columns=["name", "l1", "l2"])
    df["norad"] = df.l1.str[2:7].str.strip()
    df["group"] = group
    return df


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------
def gmst_rad(jd_ut1):
    """Greenwich mean sidereal time (IAU 1982), radians."""
    t = (jd_ut1 - 2451545.0) / 36525.0
    g = (280.46061837 + 360.98564736629 * (jd_ut1 - 2451545.0)
         + 0.000387933 * t ** 2 - t ** 3 / 38710000.0)
    return np.radians(g % 360.0)


def sun_unit_eci(jd):
    """Low-precision Sun direction (Astronomical Almanac), equator of date."""
    n = jd - 2451545.0
    L = np.radians((280.460 + 0.9856474 * n) % 360)
    g = np.radians((357.528 + 0.9856003 * n) % 360)
    lam = L + np.radians(1.915) * np.sin(g) + np.radians(0.020) * np.sin(2 * g)
    eps = np.radians(23.439 - 4e-7 * n)
    return np.stack([np.cos(lam), np.cos(eps) * np.sin(lam),
                     np.sin(eps) * np.sin(lam)], axis=-1)


def rot_z(vec, ang):
    """Rotate vectors (..., 3) about z by -ang (ECI -> Earth-fixed)."""
    c, s = np.cos(ang), np.sin(ang)
    x, y, z = vec[..., 0], vec[..., 1], vec[..., 2]
    return np.stack([c * x + s * y, -s * x + c * y, z], axis=-1)


def observer_ecef(lat, lon, elev_m):
    phi, lam = np.radians(lat), np.radians(lon)
    e2 = F_WGS84 * (2 - F_WGS84)
    N = R_EARTH / np.sqrt(1 - e2 * np.sin(phi) ** 2)
    h = elev_m / 1000.0
    return np.array([(N + h) * np.cos(phi) * np.cos(lam),
                     (N + h) * np.cos(phi) * np.sin(lam),
                     (N * (1 - e2) + h) * np.sin(phi)])


def enu_alt_az(vec_ecef, lat, lon):
    """Earth-fixed topocentric vectors (..., 3) -> altitude, azimuth (deg)."""
    phi, lam = np.radians(lat), np.radians(lon)
    x, y, z = vec_ecef[..., 0], vec_ecef[..., 1], vec_ecef[..., 2]
    e = -np.sin(lam) * x + np.cos(lam) * y
    n = (-np.sin(phi) * np.cos(lam) * x - np.sin(phi) * np.sin(lam) * y
         + np.cos(phi) * z)
    u = np.cos(phi) * np.cos(lam) * x + np.cos(phi) * np.sin(lam) * y + np.sin(phi) * z
    rng = np.sqrt(e * e + n * n + u * u)
    alt = np.degrees(np.arcsin(np.clip(u / rng, -1, 1)))
    az = np.degrees(np.arctan2(e, n)) % 360
    return alt, az, rng


def compass(az):
    pts = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
           "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    return pts[int((az % 360) / 22.5 + 0.5) % 16]


def est_mag(std_mag, rng_km, phase_rad):
    """Diffuse-sphere brightness estimate, normalized to half phase at 1000 km."""
    f = (np.sin(phase_rad) + (np.pi - phase_rad) * np.cos(phase_rad)) / np.pi
    f = np.maximum(f, 1e-4)
    return std_mag + 5 * np.log10(rng_km / 1000.0) - 2.5 * np.log10(np.pi * f)


# --------------------------------------------------------------------------
# Passes
# --------------------------------------------------------------------------
def find_passes(sats: pd.DataFrame, start_utc: datetime, end_utc: datetime,
                lat, lon, elev_m, min_alt=10.0, step_s=20, chunk=1500):
    """Return (passes table, dict of tracks keyed by pass id)."""
    n_steps = int((end_utc - start_utc).total_seconds() // step_s) + 1
    times = [start_utc + timedelta(seconds=step_s * i) for i in range(n_steps)]
    jd, fr = map(np.array, zip(*[jday(t.year, t.month, t.day, t.hour, t.minute,
                                      t.second + t.microsecond / 1e6) for t in times]))
    jd_full = jd + fr
    theta = gmst_rad(jd_full)                               # (T,)
    sun_eci = sun_unit_eci(jd_full)                         # (T, 3)
    obs = observer_ecef(lat, lon, elev_m)

    sun_alt, _, _ = enu_alt_az(rot_z(sun_eci * AU_KM, theta) - obs, lat, lon)
    sky_dark = sun_alt < -6.0                               # (T,)
    if not sky_dark.any():
        return pd.DataFrame(), {}

    rows, tracks = [], {}
    for c0 in range(0, len(sats), chunk):
        part = sats.iloc[c0:c0 + chunk]
        recs = [Satrec.twoline2rv(a, b) for a, b in zip(part.l1, part.l2)]
        err, r_teme, _ = SatrecArray(recs).sgp4(jd, fr)    # (S, T, 3) km
        ok = err == 0

        r_ecef = rot_z(r_teme, theta[None, :])
        rho = r_ecef - obs
        alt, az, rng = enu_alt_az(rho, lat, lon)

        # Cylindrical Earth shadow
        proj = np.einsum("stk,tk->st", r_teme, sun_eci)
        perp = np.linalg.norm(r_teme - proj[..., None] * sun_eci[None], axis=-1)
        sunlit = (proj > 0) | (perp > R_EARTH)

        visible = ok & sunlit & sky_dark[None, :] & (alt > min_alt)

        # Phase angle (Sun-satellite-observer), using rho in ECI frame
        obs_eci = rot_z(np.broadcast_to(obs, (len(theta), 3)), -theta)   # (T, 3)
        rho_eci = r_teme - obs_eci[None]
        cosph = -np.einsum("stk,tk->st", rho_eci, sun_eci) / rng
        phase = np.arccos(np.clip(cosph, -1, 1))

        for i in np.where(visible.any(axis=1))[0]:
            v = np.concatenate([[False], visible[i], [False]])
            edges = np.flatnonzero(np.diff(v.astype(int)))
            for a, b in zip(edges[::2], edges[1::2]):       # [a, b) visible run
                k = a + int(np.argmax(alt[i, a:b]))
                name = part.name.iloc[i]
                std = (STD_MAG["ISS"] if part.norad.iloc[i] == "25544"
                       else STD_MAG["STARLINK"] if name.startswith("STARLINK") else None)
                mag = (round(float(np.min(est_mag(std, rng[i, a:b], phase[i, a:b]))), 1)
                       if std is not None else np.nan)
                pid = len(rows) + 1
                rows.append({
                    "id": pid,
                    "name": name.title() if not name.startswith("ISS") else "ISS",
                    "norad": part.norad.iloc[i],
                    "group": part.group.iloc[i],
                    "start": times[a],
                    "peak": times[k],
                    "end": times[b - 1],
                    "start_az": az[i, a], "start_alt": alt[i, a],
                    "peak_az": az[i, k], "peak_alt": alt[i, k],
                    "end_az": az[i, b - 1], "end_alt": alt[i, b - 1],
                    "est_mag": mag,
                    "clipped": a == 0 or b == n_steps,
                })
                tracks[pid] = (alt[i, a:b].copy(), az[i, a:b].copy())

    passes = pd.DataFrame(rows)
    if len(passes):
        passes = passes.drop_duplicates(["norad", "start"]).sort_values("peak")
    return passes, tracks
