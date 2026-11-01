# Saskatoon Sky Tonight

A phone-friendly sky chart for Saskatoon: stars to magnitude 6, constellation figures,
the planets and Moon, rise/set times, Moon phase, when full darkness starts, and
satellite passes (ISS, bright satellites, Starlink) within an hour of the chosen time,
drawn as numbered tracks on the chart. The chart can be pinched, double-tapped, or
zoomed with + / − buttons, and the label size is adjustable. No ads, no tracking.

## Files

```
skygazer_v03.py              the app (current version)
satellites.py                satellite pass calculations (used by the app)
skygazer_v01.py, _v02.py     previous versions, kept for reference
requirements.txt             Python packages
data/stars.csv               ~5,000 stars (Hipparcos, via d3-celestial, BSD licence)
data/constellation_lines.csv the 88 constellation figures
```

## Try it on your computer first (optional)

```
pip install -r requirements.txt
streamlit run skygazer_v03.py
```

## Publish it free (Streamlit Community Cloud)

1. Create a free account at github.com if you don't have one.
2. Make a new repository (e.g. `saskatoon-sky`). Public is simplest.
   Use **Add file → Upload files** and drag in `skygazer_v03.py`, `satellites.py`,
   `requirements.txt`, and the `data` folder, then **Commit**.
3. Go to share.streamlit.io and sign in with your GitHub account.
4. Choose **Create app**, pick your repository, branch `main`, and set the
   main file path to `skygazer_v03.py`. Optionally choose a custom web address.
5. Click **Deploy**. The first build takes a few minutes.

You'll get a web address such as `https://saskatoon-sky.streamlit.app`.

**On your phone:** open the address in Safari or Chrome, then use
*Share → Add to Home Screen* (iPhone) or *⋮ → Add to Home screen* (Android).
It will open like an app.

Free apps go to sleep after a period with no visitors. If you see a
"wake up" button, tap it and wait about half a minute.

## Updating

Changes to a file the app already uses (for example `satellites.py` or the data
files) go live automatically a minute or so after you upload them to GitHub.

A new version with a new file name (e.g. `skygazer_v03.py`) needs the app pointed
at the new file. Streamlit's documentation doesn't describe changing the main file
path of an existing app, so the dependable route is:

1. Upload the new script (and any changed files) to the GitHub repository.
2. On share.streamlit.io, open the **⋮** menu beside the app and choose **Delete**.
3. Click **Create app** and deploy again exactly as before, with the new main file
   path and the **same App URL**. Your phone shortcut keeps working because the
   address hasn't changed.

## Notes

- Satellite orbits come from CelesTrak (celestrak.org) and are re-downloaded at most
  every 6 hours. A pass is listed only when the satellite is sunlit and your sky is dark
  (Sun more than 6° below the horizon), so most passes cluster in the first couple of
  hours after dusk and before dawn.
- Satellite brightness is a rough estimate. The ISS is reliable (often −2 to −3.5);
  individual Starlinks vary a lot, and freshly launched "trains" are much brighter
  than the settled ones.

- Saskatchewan stays on CST (UTC−6) year-round, so times never shift for daylight saving.
- Star positions are J2000 (about 0.4° off today's positions), which is invisible at
  phone-chart scale. Planet and Moon positions are computed for the exact time.
- To use a different location, edit `LAT`, `LON`, `ELEV_M`, and `TZ` near the top of the script.
