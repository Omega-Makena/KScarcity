# Boundary Processing Pipeline

## Overview
This document describes how the administrative boundaries for Kenya (ADM1 and ADM2) were acquired and processed for the CesiumJS globe visualization.

## Data Sources
We utilize the **geoBoundaries** open database (v5.0.0) for Kenya boundaries.
- **ADM1 (Counties)**: 47 polygon features.
- **ADM2 (Sub-Counties)**: ~290 polygon features.

## Acquisition Script
A Python script `kshiked/data/download_boundaries.py` was created to automate the fetching of these files from the geoBoundaries GitHub repository.

### Key features of the script:
1.  **LFS Support**: The script targets `media.githubusercontent.com` to correctly resolve Git LFS pointers for large GeoJSON files.
2.  **Versioning**: Fetches from the `main` branch to ensure availability.
3.  **Formats downloaded**:
    - `kenya_adm1_simplified.geojson`: Optimized for web rendering (mid-zoom).
    - `kenya_adm2_simplified.geojson`: Optimized for sub-county details (close-zoom).
    - `kenya_adm1_full.geojson`: High-resolution backup.

### Usage
To update the boundaries, run:
```powershell
python kshiked/data/download_boundaries.py
```

## Integration with Visualization
The `globe_viz.py` component:
1.  Reads the simplified GeoJSON files from disk.
2.  Injects them as JSON variables (`ADM1_DATA`, `ADM2_DATA`) into the generated HTML.
3.  **CesiumJS Logic**:
    - `GeoJsonDataSource.load()` is used to render polygons.
    - **LOD (Level of Detail)**: A listener on `viewer.scene.postRender` checks the camera height.
        - **Altitude > 200km**: Shows ADM1 (Counties) only.
        - **Altitude < 200km**: Enables ADM2 (Sub-Counties).

## Future Improvements
- For production scaling, serve GeoJSON files via a static file server / CDN instead of inlining them in the HTML to reduce initial page load size.
- Implement a tiling service (Cesium Ion or similar) for ADM2/ADM3 if full resolution is required at scale.
