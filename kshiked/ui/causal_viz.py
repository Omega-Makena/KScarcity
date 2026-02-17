"""
3D Causal Network Visualization
Generates HTML for a 3D force-directed graph of causal relationships.
"""

import json

def get_causal_graph_html(nodes, links, width="100%", height=500):
    """
    Generate HTML for 3D-Force-Graph visualization.
    
    Args:
        nodes: List of dicts [{'id': 'name', 'group': 'group', 'val': size}]
        links: List of dicts [{'source': 'id', 'target': 'id', 'color': 'color'}]
    """
    
    nodes_json = json.dumps(nodes)
    links_json = json.dumps(links)

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style> 
            body {{ margin: 0; overflow: hidden; background-color: #00000000; }} 
            #graphViz {{ width: 100vw; height: 100vh; }}
            .scene-tooltip {{
                color: #fff !important;
                background: rgba(0,0,0,0.8) !important;
                border: 1px solid rgba(255,255,255,0.2);
                border-radius: 4px;
                padding: 8px;
                font-family: 'Inter', sans-serif;
            }}
        </style>
        <script src="https://unpkg.com/3d-force-graph"></script>
    </head>
    <body>
        <div id="graphViz"></div>
        <script>
            const gData = {{
                nodes: {nodes_json},
                links: {links_json}
            }};

            const Graph = ForceGraph3D()
                (document.getElementById('graphViz'))
                .graphData(gData)
                .backgroundColor('rgba(0,0,0,0)')
                
                // Nodes
                .nodeLabel('id')
                .nodeColor(node => node.color || '#6366f1')
                .nodeVal('val')
                .nodeOpacity(0.9)
                .nodeResolution(16)
                
                // Links
                .linkWidth(link => link.width || 1)
                .linkColor('color')
                .linkDirectionalParticles(2)
                .linkDirectionalParticleSpeed(d => d.value * 0.005)
                .linkDirectionalParticleWidth(2)
                
                // Forces
                .d3Force('charge', null) // Disable default charge to customize
                
                // Interaction
                .enableNodeDrag(true)
                .onNodeClick(node => {{
                    // Aim at node from outside it
                    const distance = 40;
                    const distRatio = 1 + distance/Math.hypot(node.x, node.y, node.z);
                    Graph.cameraPosition(
                        {{ x: node.x * distRatio, y: node.y * distRatio, z: node.z * distRatio }}, // new position
                        node, // lookAt ({{ x, y, z }})
                        3000  // ms transition duration
                    );
                }});

            // Spread graph out
            Graph.d3Force('charge').strength(-120);
            
            // Auto rotate
            let angle = 0;
            setInterval(() => {{
                Graph.cameraPosition({{
                    x: 200 * Math.sin(angle),
                    z: 200 * Math.cos(angle)
                }});
                angle += Math.PI / 600;
            }}, 20);

            // Resize handling
            window.addEventListener('resize', () => {{
                Graph.width(window.innerWidth).height(window.innerHeight);
            }});
        </script>
    </body>
    </html>
    """
    return html_code
