# Data & Analysis — Overview

> `kshiked.data` + `kshiked.analysis` — Data assets and quality tools for SENTINEL.

---

## Purpose

These modules manage the geographic and news data that feeds the dashboard's Regional Map tab and document intelligence features.

---

## GeoJSON Boundary Data (`kshiked/data/`)

Kenya geographic boundary files for county-level visualization:

| File | Size | Description |
|------|------|-------------|
| `kenya_counties.geojson` | 16 KB | Simplified 47-county boundaries |
| `kenya_outline.geojson` | 5 KB | Kenya national outline |
| `kenya_adm1_simplified.geojson` | 862 KB | Admin Level 1 (counties) — simplified |
| `kenya_adm1_full.geojson` | 7.9 MB | Admin Level 1 — full resolution |
| `kenya_adm2_simplified.geojson` | 2.0 MB | Admin Level 2 (sub-counties) — simplified |
| `kenya_adm2_full.geojson` | 10.9 MB | Admin Level 2 — full resolution |
| `download_boundaries.py` | 1.5 KB | Script to download boundaries from GADM |
| `README_GEOJSON.md` | 883 B | Data source attribution and usage notes |

### Download Script

```bash
python kshiked/data/download_boundaries.py
```

Downloads Kenya admin boundaries from GADM and saves both full and simplified versions.

---

## News Database (`kshiked/data/`)

| File | Description |
|------|-------------|
| `news_db.sqlite` | SQLite database storing fetched news articles |
| `news_cache/` | Local cache for raw article data |

The `NewsDatabase` class (defined in `kshiked/pulse/news_db.py`) provides:

- Article storage with deduplication
- Signal tagging from Pulse analysis
- V3 fields: TTA (Time-to-Action), Role taxonomy, Resilience index

---

## Analysis Utilities (`kshiked/analysis/`)

| File | Size | Purpose |
|------|------|---------|
| `analyze_data_quality.py` | 2.7 KB | Data quality checks — missing values, outliers, distribution analysis |
| `find_crash.py` | 1.5 KB | Crash root-cause finder — traces exceptions in log files |

---

## Integration

```
download_boundaries.py → GeoJSON files
    │
    ▼
kenya_data_loader.py (kshiked/ui/)
    │
    ▼
kenya_threatmap.py (kshiked/ui/)
    │
    ▼
Dashboard Tab 8: Regional Map
```

```
NewsIngestor (kshiked/pulse/news.py)
    │
    ▼
news_db.sqlite (kshiked/data/)
    │
    ▼
Dashboard Tab 9: Documents
```

---

*Source: `kshiked/data/`, `kshiked/analysis/` · Last updated: 2026-02-11*
