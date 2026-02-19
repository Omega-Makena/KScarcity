import requests
import json
import os

API_KEY = "9b404b4922ed4ff7a71c9f2247b5c722"

def probe():
    print("--- NewsAPI Probe ---")
    
    # 1. Broad Top Headlines for KE
    url = "https://newsapi.org/v2/top-headlines"
    params = {"country": "ke", "apiKey": API_KEY}
    print(f"\n1. Top Headlines (KE): {url}")
    try:
        r = requests.get(url, params=params)
        data = r.json()
        print(f"   Count: {data.get('totalResults')}")
        if data.get('articles'):
            print(f"   Sample Source: {data['articles'][0]['source']['name']}")
    except Exception as e:
        print(f"   Error: {e}")

    # 2. Everything 'Kenya' (No domains)
    url = "https://newsapi.org/v2/everything"
    params = {"q": "kenya", "sortBy": "publishedAt", "apiKey": API_KEY, "pageSize": 5}
    print(f"\n2. Everything 'Kenya' (No domains): {url}")
    try:
        r = requests.get(url, params=params)
        data = r.json()
        print(f"   Count: {data.get('totalResults')}")
        if data.get('articles'):
            for a in data['articles'][:3]:
                print(f"   - {a['title']} ({a['source']['name']} - {a['url']})")
    except Exception as e:
        print(f"   Error: {e}")

    # 3. Everything 'Politics' + Domains
    domains = "nation.africa,standardmedia.co.ke,the-star.co.ke"
    params = {"q": "politics", "domains": domains, "sortBy": "publishedAt", "apiKey": API_KEY}
    print(f"\n3. Everything 'Politics' + Domains ({domains}):")
    try:
        r = requests.get(url, params=params)
        data = r.json()
        print(f"   Count: {data.get('totalResults')}")
    except Exception as e:
         print(f"   Error: {e}")

if __name__ == "__main__":
    probe()
