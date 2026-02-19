"""
Animated WebGL Header Component
Generates HTML for a Particles.js animated header.
"""

def get_animated_header_html(title="SENTINEL COMMAND CENTER", subtitle="Strategic National Economic & Threat Intelligence Layer", height=150):
    """
    Generate HTML for animated header.
    """
    
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&display=swap');
            
            body {{ margin: 0; overflow: hidden; background-color: #020a06; font-family: 'Space Mono', monospace; }}
            #particles-js {{ 
                position: absolute; 
                width: 100%; 
                height: 100%; 
                background: linear-gradient(135deg, #0d2116 0%, #020a06 100%);
                z-index: 1;
            }}
            .content {{
                position: absolute;
                width: 100%;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                text-align: center;
                z-index: 2;
                pointer-events: none;
                text-transform: uppercase;
                letter-spacing: 2px;
            }}
            h1 {{
                margin: 0;
                color: #ffffff;
                font-size: 2.2rem;
                font-weight: 700;
                text-shadow: 0 0 30px rgba(0, 255, 136, 0.4);
                letter-spacing: 4px;
            }}
            p {{
                margin: 10px 0 0 0;
                color: #6e8a70;
                font-size: 0.85rem;
                font-weight: 400;
            }}
        </style>
        <script src="https://cdn.jsdelivr.net/particles.js/2.0.0/particles.min.js"></script>
    </head>
    <body>
        <div id="particles-js"></div>
        <div class="content">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        
        <script>
            particlesJS("particles-js", {{
                "particles": {{
                    "number": {{ "value": 60, "density": {{ "enable": true, "value_area": 800 }} }},
                    "color": {{ "value": "#00ff88" }},
                    "shape": {{ "type": "circle" }},
                    "opacity": {{ "value": 0.4, "random": true }},
                    "size": {{ "value": 2, "random": true }},
                    "line_linked": {{ "enable": true, "distance": 150, "color": "#00ff88", "opacity": 0.2, "width": 1 }},
                    "move": {{ "enable": true, "speed": 1.5, "direction": "none", "random": false, "out_mode": "out" }}
                }},
                "interactivity": {{
                    "detect_on": "canvas",
                    "events": {{
                        "onhover": {{ "enable": true, "mode": "grab" }},
                        "onclick": {{ "enable": true, "mode": "push" }},
                        "resize": true
                    }},
                    "modes": {{
                        "grab": {{ "distance": 140, "line_linked": {{ "opacity": 0.6 }} }}
                    }}
                }},
                "retina_detect": true
            }});
        </script>
    </body>
    </html>
    """
    return html_code
