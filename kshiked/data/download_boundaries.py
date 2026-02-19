
import os
import urllib.request
import ssl

def download_file(url, dest_path):
    print(f"Downloading {url} to {dest_path}...")
    try:
        # Create unverified context to avoid SSL errors in some environments
        context = ssl._create_unverified_context()
        with urllib.request.urlopen(url, context=context) as response:
            data = response.read()
            with open(dest_path, 'wb') as f:
                f.write(data)
        print("Success.")
    except Exception as e:
        print(f"Failed: {e}")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # URLs - Using media.githubusercontent.com for LFS support
    base_url = "https://media.githubusercontent.com/media/wmgeolab/geoBoundaries/main/releaseData/gbOpen/KEN"
    
    files = {
        "kenya_adm1_simplified.geojson": f"{base_url}/ADM1/geoBoundaries-KEN-ADM1_simplified.geojson",
        "kenya_adm1_full.geojson": f"{base_url}/ADM1/geoBoundaries-KEN-ADM1.geojson",
        "kenya_adm2_simplified.geojson": f"{base_url}/ADM2/geoBoundaries-KEN-ADM2_simplified.geojson",
        "kenya_adm2_full.geojson": f"{base_url}/ADM2/geoBoundaries-KEN-ADM2.geojson"
    }
    
    for filename, url in files.items():
        dest_path = os.path.join(base_dir, filename)
        if os.path.exists(dest_path):
            print(f"File {filename} already exists. Skipping.")
            continue
        download_file(url, dest_path)

if __name__ == "__main__":
    main()
