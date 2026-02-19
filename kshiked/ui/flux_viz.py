
import json

def get_flux_graph_html(flow_data, width="100%", height=500):
    """
    Generate HTML for 3D-Force-Graph visualization of Economic Flux.
    
    Args:
        flow_data: List of frames [{'t': 1, 'flows': {'consumption': 10...}}]
    """
    
    # 1. Define Fixed Nodes (The 4 Sectors)
    nodes = [
        {"id": "Households", "color": "#00ff88", "val": 20, "fx": -50, "fy": 0, "fz": 0},
        {"id": "Firms", "color": "#00f3ff", "val": 20, "fx": 50, "fy": 0, "fz": 0},
        {"id": "Government", "color": "#ff6b35", "val": 15, "fx": 0, "fy": 60, "fz": 0},
        {"id": "Banks", "color": "#aa00ff", "val": 15, "fx": 0, "fy": -60, "fz": 0}
    ]
    
    # 2. Define Links (The Pipes)
    # IDs here correspond to logic in JS update
    links = [
        {"source": "Households", "target": "Firms", "type": "consumption", "color": "#00ff88"},
        {"source": "Firms", "target": "Households", "type": "wages", "color": "#00f3ff"}, # Assuming simplified income flow
        {"source": "Households", "target": "Government", "type": "tax_revenue", "color": "#ff6b35"},
        {"source": "Government", "target": "Firms", "type": "govt_spending", "color": "#ff6b35"},
        {"source": "Households", "target": "Banks", "type": "savings", "color": "#aa00ff"},
        {"source": "Firms", "target": "Firms", "type": "investment", "color": "#ffffff"} # Self-loop
    ]

    # Clean flow data for JS
    frames_json = json.dumps(flow_data)
    nodes_json = json.dumps(nodes)
    links_json = json.dumps(links)

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style> 
            body {{ margin: 0; overflow: hidden; background-color: #00000000; }} 
            #fluxViz {{ width: 100vw; height: 100vh; }}
            .overlay {{
                position: absolute; top: 10px; left: 10px;
                color: #00ff88; font-family: monospace; font-size: 1.2em;
                pointer-events: none;
            }}
        </style>
        <script src="https://unpkg.com/3d-force-graph"></script>
    </head>
    <body>
        <div id="fluxViz"></div>
        <div class="overlay" id="timeDisplay">T=0</div>
        
        <script>
            const FRAMES = {frames_json};
            const NODES = {nodes_json};
            const LINKS = {links_json};
            
            const Graph = ForceGraph3D()
                (document.getElementById('fluxViz'))
                .graphData({{ nodes: NODES, links: LINKS }})
                .backgroundColor('rgba(0,0,0,0)')
                
                // Nodes
                .nodeLabel('id')
                .nodeColor('color')
                .nodeVal('val')
                
                // Links
                .linkWidth(2)
                .linkColor('color')
                .linkDirectionalParticles(2)
                .linkDirectionalParticleSpeed(0.005)
                .linkCurvature(d => d.source === d.target ? 0.5 : 0) // Curve self-loops
                
                // Forces - Freeze positions
                .d3Force('charge', null)
                .d3Force('link', null)
                .d3Force('center', null);
                
            // Animation Loop
            let step = 0;
            setInterval(() => {{
                if (step >= FRAMES.length) step = 0;
                
                const frame = FRAMES[step];
                const flows = frame.flows;
                document.getElementById('timeDisplay').innerText = `T=${{frame.t}} | GDP Growth: ${{ (frame.outcomes.gdp_growth*100).toFixed(2) }}%`;
                
                // Update Particle Density/Speed based on flow magnitude
                Graph.linkDirectionalParticles(link => {{
                    if (link.type === 'consumption') return Math.max(1, flows.consumption / 5);
                    if (link.type === 'govt_spending') return Math.max(1, flows.govt_spending / 5);
                    if (link.type === 'investment') return Math.max(1, flows.investment / 5);
                    if (link.type === 'tax_revenue') return Math.max(1, flows.tax_revenue / 5);
                    if (link.type === 'savings') return Math.max(0, flows.savings / 2);
                    if (link.type === 'wages') return Math.max(1, (flows.consumption + flows.savings) / 5); // Approximate wages
                    return 2;
                }});
                
                // Pulse nodes if receiving heavy flow
                Graph.nodeVal(node => {{
                     if (node.id === "Firms") return 20 + flows.consumption/10;
                     return 15;
                }});
                
                if(step % 5 === 0) Graph.refresh(); // Periodic refresh
                step++; 
            }}, 200); // 5 steps per second
            
            // Camera Orbit
            let angle = 0;
            setInterval(() => {{
                Graph.cameraPosition({{
                    x: 180 * Math.sin(angle),
                    z: 180 * Math.cos(angle)
                }});
                angle += Math.PI / 1000;
            }}, 20);

            window.addEventListener('resize', () => Graph.width(window.innerWidth).height(window.innerHeight));
        </script>
    </body>
    </html>
    """
    return html_code
