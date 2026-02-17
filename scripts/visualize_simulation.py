
import pandas as pd
import json
import os
from datetime import datetime

# Configuration
INPUT_FILE = "data/synthetic_kenya/tweets.csv"
OUTPUT_FILE = "simulation_playback.html"

def generate_viz():
    print(f"Loading {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found. Run generation script first.")
        return

    # Filter for visual relevance (Sampling to avoid browser crash)
    # We want High Risk / Escalation / Infra Stress
    relevant_intents = ["infrastructure_stress", "mobilization", "migration_signal", "escalation", "rumor_mill"]
    
    # Take all relevant tweets + sample of casuals
    relevant_df = df[df["intent"].isin(relevant_intents)]
    casual_df = df[~df["intent"].isin(relevant_intents)].sample(n=min(2000, len(df)//10))
    
    viz_df = pd.concat([relevant_df, casual_df])
    
    # Normalize Timestamps
    viz_df["dt"] = pd.to_datetime(viz_df["timestamp"])
    start_time = viz_df["dt"].min()
    viz_df["time_offset"] = (viz_df["dt"] - start_time).dt.total_seconds() / 3600 # Hours
    
    # Prepare Data for JS
    points_data = []
    
    for _, row in viz_df.iterrows():
        color = "#00ff88" # Default Green
        size = 0.2
        
        if row["intent"] == "infrastructure_stress":
            color = "#ff0044" # Red
            size = 1.5
        elif row["intent"] == "mobilization":
            color = "#ff6b35" # Orange
            size = 1.0
        elif row["intent"] == "migration_signal":
            color = "#00f3ff" # Cyan
            size = 0.8
            
        points_data.append({
            "lat": row["latitude"],
            "lng": row["longitude"],
            "time": row["time_offset"],
            "color": color,
            "size": size,
            "text": f"{row['account_id'][:8]}: {row['intent']}",
            "intent": row["intent"]
        })
        
    points_json = json.dumps(points_data)
    
    # Arcs (Replies/Retweets)
    arcs_data = []
    # Create lookup for lat/lon
    loc_map = viz_df.set_index("post_id")[["latitude", "longitude"]].to_dict("index")
    
    for _, row in viz_df[viz_df["interaction_type"].isin(["Reply", "Retweet"])].iterrows():
        if row["reply_to_post_id"] in loc_map:
            target = loc_map[row["reply_to_post_id"]]
            arcs_data.append({
                "startLat": row["latitude"],
                "startLng": row["longitude"],
                "endLat": target["latitude"],
                "endLng": target["longitude"],
                "color": ["rgba(0, 255, 136, 0.5)", "rgba(255, 0, 68, 0.8)"],
                "time": row["time_offset"]
            })
            
    arcs_json = json.dumps(arcs_data)

    # HTML Template
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <style> 
        body {{ margin: 0; background: #000; font-family: monospace; overflow: hidden; }} 
        #controls {{
            position: absolute; bottom: 20px; left: 20px; right: 20px;
            background: rgba(0,0,0,0.8); padding: 20px; border-radius: 8px;
            color: #00ff88; display: flex; align-items: center; gap: 20px;
            border: 1px solid #00ff88;
        }}
        input[type=range] {{ flex-grow: 1; }}
        #time-display {{ min-width: 150px; font-weight: bold; font-size: 1.2em; }}
        .legend {{
            position: absolute; top: 20px; right: 20px;
            background: rgba(0,0,0,0.8); padding: 15px;
            color: white; border-radius: 8px; border: 1px solid #333;
        }}
        .dot {{ width: 10px; height: 10px; display: inline-block; border-radius: 50%; margin-right: 5px; }}
    </style>
    <script src="https://unpkg.com/globe.gl"></script>
</head>
<body>
    <div id="globeViz"></div>
    
    <div class="legend">
        <h3>Simulation Legend</h3>
        <div><span class="dot" style="background:#ff0044"></span>Infra Stress (Blackout)</div>
        <div><span class="dot" style="background:#ff6b35"></span>Mobilization (Protest)</div>
        <div><span class="dot" style="background:#00f3ff"></span>Migration (Going Dark)</div>
        <div><span class="dot" style="background:#00ff88"></span>Casual/Normal</div>
    </div>

    <div id="controls">
        <button id="play-btn" style="background:#00ff88; border:none; padding:10px 20px; font-weight:bold; cursor:pointer;">PLAY</button>
        <input type="range" id="time-slider" min="0" max="{viz_df['time_offset'].max()}" step="1" value="0">
        <div id="time-display">Day 0 - 00:00</div>
    </div>

    <script>
        const POINTS = {points_json};
        const ARCS = {arcs_json};
        
        const globe = Globe()
            (document.getElementById('globeViz'))
            .globeImageUrl('https://unpkg.com/three-globe/example/img/earth-night.jpg')
            .bumpImageUrl('https://unpkg.com/three-globe/example/img/earth-topology.png')
            .backgroundImageUrl('https://unpkg.com/three-globe/example/img/night-sky.png')
            .pointAltitude('size')
            .pointColor('color')
            .pointsData([])
            .arcsData([])
            .arcColor('color')
            .arcDashLength(0.4)
            .arcDashGap(4)
            .arcDashAnimateTime(1000)
            .pointRadius(0.5);
            
        // Initial View: Kenya
        globe.pointOfView({{ lat: -1.29, lng: 36.82, altitude: 2.0 }});

        // Time Controller
        const slider = document.getElementById('time-slider');
        const display = document.getElementById('time-display');
        const playBtn = document.getElementById('play-btn');
        let isPlaying = false;
        let animationId;
        
        function updateArgs() {{
            const currentHour = parseFloat(slider.value);
            const windowSize = 24; // Show 24h window
            
            // Filter Data
            const activePoints = POINTS.filter(d => d.time >= currentHour && d.time < currentHour + windowSize);
            const activeArcs = ARCS.filter(d => d.time >= currentHour && d.time < currentHour + windowSize);
            
            globe.pointsData(activePoints);
            globe.arcsData(activeArcs);
            
            // Update Display
            const day = Math.floor(currentHour / 24);
            const hour = Math.floor(currentHour % 24);
            display.innerText = `Day ${{day + 1}} - ${{hour.toString().padStart(2, '0')}}:00`;
            
            // Cinematic Camera Move (Drift)
            globe.controls().autoRotate = isPlaying;
            globe.controls().autoRotateSpeed = 0.5;
        }}
        
        slider.addEventListener('input', updateArgs);
        
        playBtn.addEventListener('click', () => {{
            isPlaying = !isPlaying;
            playBtn.innerText = isPlaying ? "PAUSE" : "PLAY";
            if (isPlaying) animate();
            else cancelAnimationFrame(animationId);
        }});
        
        function animate() {{
            if (!isPlaying) return;
            let val = parseFloat(slider.value);
            val += 1; // Speed
            if (val > slider.max) val = 0; // Loop
            slider.value = val;
            updateArgs();
            animationId = requestAnimationFrame(animate);
        }}
        
        // Init
        updateArgs();
        
    </script>
</body>
</html>
    """
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(html)
        
    print(f"✅ Visualization generated: {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_viz()
