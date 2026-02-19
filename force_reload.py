
import os
import time

# Touch the main dashboard file to force Streamlit to reload
dashboard_file = r"c:\Users\omegam\OneDrive - Innova Limited\scace4\kshiked\ui\sentinel_dashboard.py"
if os.path.exists(dashboard_file):
    os.utime(dashboard_file, None)
    print(f"Touched {dashboard_file} to trigger reload")
else:
    print(f"File not found: {dashboard_file}")
