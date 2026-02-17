"""
Sentinel Cinematic Globe - PROPER 3D VERSION
Fixed zoom levels for proper 3D globe effect
"""

import os
import json
import logging

logger = logging.getLogger(__name__)

def get_globe_html(counties_data=None, height=700):
    # ---------- GeoJSON loading ----------
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        "kenya_adm1_simplified.geojson",
        "kenya_counties.geojson",
        "kenya_outline.geojson"
    ]
    
    geojson_str = "null"
    for name in candidates:
        path = os.path.join(base_dir, "data", name)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    geojson_data = json.load(f)
                    geojson_str = json.dumps(geojson_data)
                    break
            except Exception as e:
                logger.error(f"Error loading {name}: {e}")
    
    # ---------- Risk Data ----------
    risk_map = {}
    if counties_data:
        for item in counties_data:
            name = item.get("name", "Unknown")
            risk_map[name.title()] = float(item.get("risk_score", 0.0))
            
    risk_map_json = json.dumps(risk_map)

    # ---------- HTML Template ----------
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    body {{
      margin: 0;
      padding: 0;
      overflow: hidden;
      background: #000;
      font-family: 'Space Mono', 'Courier New', monospace;
    }}
    #globeViz {{
      width: 100vw;
      height: {height}px;
      background: #000;
    }}
    #globeViz canvas {{
      outline: none;
    }}
    .globe-overlay {{
      position: absolute;
      top: 20px;
      left: 20px;
      color: white;
      background: rgba(0, 0, 0, 0.7);
      padding: 15px;
      border-radius: 10px;
      border-left: 4px solid #4CAF50;
      z-index: 1000;
      max-width: 300px;
    }}
  </style>
  <script src="https://unpkg.com/globe.gl"></script>
  <!-- Smart Loader for Three.js to avoid duplicates -->
  <script>
    if (!window.THREE) {{
        document.write('<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"><\\/script>');
    }}
  </script>
</head>

<body>
  <div id="globeViz"></div>
  <div class="globe-overlay">
    <h3 style="margin: 0 0 10px 0;">🌍 Kenya Risk Map</h3>
    <div id="globe-status" style="font-size: 0.8em; color: #aaa;">Initializing...</div>
  </div>

  <script>
    console.log("🚀 Initializing 3D Globe...");
    
    // Color palette
    const COLORS = {{
        CRITICAL: '#ff0044',
        HIGH: '#ff6b35',
        MODERATE: '#f5d547',
        LOW: '#00ff88',
        DEFAULT: 'rgba(150, 150, 150, 0.3)'
    }};
    
    // Data from Python
    const KENYA_GEOJSON = {geojson_str};
    const RISKS = {risk_map_json};
    
    // DEBUG: Check Data
    const statusDiv = document.getElementById('globe-status');
    if (!KENYA_GEOJSON || !KENYA_GEOJSON.features) {{
        statusDiv.innerHTML = '<span style="color:red">ERROR: GeoJSON Data Missing!</span>';
        console.error("CRITICAL: KENYA_GEOJSON is null or empty.");
    }} else {{
        statusDiv.innerHTML = `<span style="color:#00ff88">Active Nodes: ${{KENYA_GEOJSON.features.length}}</span><br>Cinematic Mode: ON`;
    }}
    
    console.log("GeoJSON features:", KENYA_GEOJSON?.features?.length || 0);
    console.log("Risk data:", Object.keys(RISKS).length, "counties");
    
    // NORMALIZE HELPER
    function normName(s) {{
        return (s || '').trim().toUpperCase();
    }}

    // Get county name from properties
    function getCountyName(d) {{
        if (!d?.properties) return null;
        const props = d.properties;
        return props.shapeName || props.name || props.COUNTY || props.county || null;
    }}
    
    // Get risk score with normalization
    function getRiskScore(countyName) {{
        return RISKS[normName(countyName)] || 0;
    }}

    // Get color based on risk score
    function getRiskColor(countyName) {{
        const score = getRiskScore(countyName);
        if (score >= 0.7) return COLORS.CRITICAL;
        if (score >= 0.5) return COLORS.HIGH;
        if (score >= 0.3) return COLORS.MODERATE;
        if (score > 0) return COLORS.LOW;
        return COLORS.DEFAULT;
    }}
    
    // Get extrusion height
    function getExtrusion(countyName) {{
        const score = getRiskScore(countyName);
        return 0.005 + (score * 0.015);
    }}
    
    // Initialize Globe
    try {{
        const container = document.getElementById('globeViz');
        

        // Create the globe
        const globe = Globe()(container)
            // Earth textures (Cyber/Holo Mode)
            .globeImageUrl('https://unpkg.com/three-globe/example/img/earth-night.jpg')
            .bumpImageUrl('https://unpkg.com/three-globe/example/img/earth-topology.png')
            .backgroundImageUrl('https://unpkg.com/three-globe/example/img/night-sky.png')
            
            // Globe settings (Kaspersky Style)
            .width(window.innerWidth)
            .height(container.clientHeight) // Use container height
            .backgroundColor('#020a06') // Matches theme background
            .showAtmosphere(true)
            .atmosphereColor('#00f3ff') // Neon Cyan Atmosphere
            .atmosphereAltitude(0.25)   // Stronger glow
            
            // Lighting for 3D effect
            .showGraticules(true)
            .graticuleColor('rgba(0, 243, 255, 0.05)') // Faint Cyber Grid
            .enablePointerInteraction(true)
            
            // County polygons (Holographic)
            .polygonsData(KENYA_GEOJSON?.features || [])
            .polygonCapColor(d => {{
                // Glassy/Holographic fill based on risk
                const color = getRiskColor(getCountyName(d));
                // Convert hex to rgba for glass effect
                // Simple hack: return color but rely on the renderer blending or use a helper if needed.
                // Globe.gl accepts rgba strings.
                // Let's assume getRiskColor returns hex, we make it transparent.
                return color === COLORS.DEFAULT ? 'rgba(100,100,100,0.1)' : color + '40'; // 25% Opacity (Hex Alpha)
            }})
            .polygonSideColor(() => 'rgba(0, 255, 136, 0.1)') // Green faint sides
            .polygonStrokeColor(() => '#00ff88') // Neon Green strokes for Kenya
            .polygonAltitude(d => getExtrusion(getCountyName(d)))
            .polygonLabel(d => {{
                const name = getCountyName(d) || 'Unknown';
                const score = getRiskScore(name);
                return `
                    <div style="
                        background: rgba(2, 10, 6, 0.9);
                        color: #00ff88;
                        padding: 12px;
                        border-radius: 4px;
                        border: 1px solid #00ff88;
                        box-shadow: 0 0 15px rgba(0, 255, 136, 0.3);
                        font-family: 'Space Mono', monospace;
                        max-width: 250px;
                    ">
                        <div style="font-size: 1.2em; font-weight: bold; margin-bottom: 5px; text-transform: uppercase;">
                            ${{name}}
                        </div>
                        <div style="margin-bottom: 3px; border-bottom: 1px solid #333; padding-bottom: 5px;">
                            Risk Level: <span style="color: ${{getRiskColor(name)}}; font-weight: bold; text-shadow: 0 0 5px ${{getRiskColor(name)}};">
                                ${{score >= 0.7 ? 'CRITICAL' : 
                                  score >= 0.5 ? 'HIGH' : 
                                  score >= 0.3 ? 'MODERATE' : 
                                  score > 0 ? 'LOW' : 'NO DATA'}}
                            </span>
                        </div>
                        <div style="font-size: 0.9em; color: #aaa;">
                            Score: <span style="color: white;">${{score.toFixed(2)}}</span>
                        </div>
                    </div>
                `;
            }});

        // 4. OPTIMIZED HOVER
        let hoveredPolygon = null;
        globe.onPolygonHover(hoverD => {{
            hoveredPolygon = hoverD;
            globe.polygonCapColor(d => {{
                // Use cached ID check for speed
                const name = getCountyName(d);
                const color = getRiskColor(name);
                // Highlight: Solid Neon or Glassy
                if (d === hoveredPolygon) return color; // Solid on hover
                return color === COLORS.DEFAULT ? 'rgba(100,100,100,0.1)' : color + '40'; // Glassy otherwise
            }});
            globe.polygonStrokeColor(d => {{
                return d === hoveredPolygon ? '#ffffff' : '#00ff88';
            }});
        }});
        
        // 1. FIX RESOLUTION
        if (globe.renderer()) {{
            globe.renderer().setPixelRatio(window.devicePixelRatio);
            console.log("💎 High-DPI Resolution Enabled");
        }}

        // 2. ADD WORLD CONTEXT (Kaspersky Neon Borders)
        // Fetch low-res world borders
        fetch('https://raw.githubusercontent.com/vasturiano/globe.gl/master/example/datasets/ne_110m_admin_0_countries.geojson')
            .then(res => res.json())
            .then(countriesData => {{
                // Render as paths (outlines)
                // Globe.gl lineData expects {{ startLat, startLng, endLat, endLng }}
                // Globe.gl pathsData expects [ [ [lat, lng], ... ], ... ]
                
                // We need to convert GeoJSON features to paths
                // Simple helper to extract coordinate arrays from Poly/MultiPoly
                const paths = [];
                countriesData.features.forEach(f => {{
                    if (f.geometry.type === 'Polygon') {{
                        f.geometry.coordinates.forEach(ring => {{
                            paths.push(ring.map(c => [c[1], c[0]])); // GeoJSON is Lng,Lat -> Globe is Lat,Lng
                        }});
                    }} else if (f.geometry.type === 'MultiPolygon') {{
                        f.geometry.coordinates.forEach(poly => {{
                            poly.forEach(ring => {{
                                paths.push(ring.map(c => [c[1], c[0]]));
                            }});
                        }});
                    }}
                }});

                globe.pathsData(paths)
                     .pathColor(() => '#005577') // Dark Cyan for world context
                     .pathDashLength(0.01)
                     .pathDashGap(0.005)
                     .pathDashAnimateTime(60000);
                
                console.log("🌍 World Borders Loaded as Paths (Cyber Mode)");
            }});

        // Get Three.js controls
        const controls = globe.controls();
        
        let starFieldGroup = null;
        let starSpinRequest = null;
        const STARFIELD_SPIN_SPEED = 0.00025;
        const STARFIELD_PITCH_SPEED = 0.00008;

        function animateStarfield() {{
            if (starFieldGroup) {{
                starFieldGroup.rotation.y += STARFIELD_SPIN_SPEED;
                starFieldGroup.rotation.x += STARFIELD_PITCH_SPEED;
            }}
            starSpinRequest = requestAnimationFrame(animateStarfield);
        }}

        function startStarfieldSpin() {{
            if (!starSpinRequest) {{
                animateStarfield();
            }}
        }}
        
        // 6. FIX CAMERA DISTANCE (Scale by Radius)
        // Globe radius is 100 by default in Globe.gl
        const R = 100; 

        if (controls) {{
            controls.enableRotate = true;
            controls.enableZoom = true;
            controls.enablePan = false;
            controls.autoRotate = true;
            controls.autoRotateSpeed = 0.3;
            
            // Tuned limits based on Radius
            controls.minDistance = 1.2 * R; // 120
            controls.maxDistance = 10 * R;  // 1000
            
            controls.minPolarAngle = Math.PI / 6;
            controls.maxPolarAngle = Math.PI * 5/6;
            
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
        }}
        
        // Access Three.js camera
        const camera = globe.camera();
        if (camera) {{
            camera.fov = 50;
            camera.near = 0.1;
            camera.far = 10000;
            camera.updateProjectionMatrix();
        }}
        
        // 3. CONSISTENT RESIZE
        window.addEventListener('resize', () => {{
            const w = window.innerWidth;
            const h = container.clientHeight; // Trust container
            globe.width(w);
            globe.height(h);
            if (camera) {{
                camera.aspect = w / h;
                camera.updateProjectionMatrix();
            }}
        }});
        
        console.log(" Starting cinematic sequence...");

        const cinematicSequence = [
            {{
                view: {{ lat: -5, lng: 20, altitude: 3.5 }},
                duration: 2200,
                pause: 400
            }},
            {{
                view: {{ lat: 5, lng: 30, altitude: 2.2 }},
                duration: 3000,
                pause: 600
            }},
            {{
                view: {{ lat: 1, lng: 37.5, altitude: 0.9 }},
                duration: 3500,
                pause: 400
            }}
        ];

        function runCinematic(index = 0) {{
            if (index >= cinematicSequence.length) {{
                console.log(" Cinematic sequence complete. Globe will hold while the starfield spins.");
                if (controls) {{
                    controls.autoRotate = false;
                }}
                return;
            }}

            const {{ view, duration, pause }} = cinematicSequence[index];
            globe.pointOfView(view, duration);
            if (index === cinematicSequence.length - 1) {{
                console.log(" Diving into Kenya...");
            }}
            setTimeout(() => runCinematic(index + 1), duration + (pause || 500));
        }}

        if (controls) {{
            controls.autoRotate = false;
        }}

        setTimeout(() => runCinematic(), 600);
        
        // Add starfield
        setTimeout(() => {{
            if (globe.scene) {{
                const starCount = 1000;
                const starGeometry = new THREE.BufferGeometry();
                const starPositions = new Float32Array(starCount * 3);
                for (let i = 0; i < starCount * 3; i += 3) {{
                    const rDist = 500 + Math.random() * 500;
                    const theta = Math.random() * Math.PI * 2;
                    const phi = Math.random() * Math.PI;
                    starPositions[i] = rDist * Math.sin(phi) * Math.cos(theta);
                    starPositions[i + 1] = rDist * Math.cos(phi);
                    starPositions[i + 2] = rDist * Math.sin(phi) * Math.sin(theta);
                }}
                starGeometry.setAttribute('position', new THREE.BufferAttribute(starPositions, 3));
                const starMaterial = new THREE.PointsMaterial({{
                    color: 0xffffff,
                    size: 1,
                    transparent: true,
                    opacity: 0.7
                }});
                const stars = new THREE.Points(starGeometry, starMaterial);
                stars.frustumCulled = false;
                const starsContainer = new THREE.Group();
                starsContainer.name = 'starfield';
                starsContainer.add(stars);
                starFieldGroup = starsContainer;
                globe.scene().add(starFieldGroup);
                startStarfieldSpin();
            }}
        }}, 5000);
        
        console.log("✅ Globe initialized successfully!");
        
        // 7. REMOVED DEBUG INTERVAL
        
    }} catch (error) {{
        console.error(" Globe init failed:", error);
        container.innerHTML = `<h3 style='color:white'>Globe Error: ${{error.message}}</h3>`;
    }}
    
  </script>
</body>
</html>
"""
    return html
