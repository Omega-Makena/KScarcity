
import random
import uuid
import numpy as np
import numpy as np
from .vocabulary import COUNTY_COORDINATES, COUNTY_WEIGHTS, DEVICE_TYPES

class AccountGenerator:
    def __init__(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)

    def generate_accounts(self, num_accounts=1000):
        accounts = []
        for _ in range(num_accounts):
            account = self._generate_single_account()
            accounts.append(account)
        return accounts

    def _generate_single_account(self):
        account_id = str(uuid.uuid4())
        account_age_days = int(np.random.exponential(scale=365*2)) + 1 
        
        # Account Type Determination
        rand_type = random.random()
        if rand_type < 0.70:
            account_type = "Individual"
        elif rand_type < 0.90:
            account_type = "Bot"
        elif rand_type < 0.95:
            account_type = "Organization"
        else:
            account_type = "Government"
            
        # Defaults
        risk_band = "Low"
        network_cluster_id = random.randint(11, 100)
        
        if account_type == "Individual":
            followers = int(np.random.lognormal(mean=6, sigma=1.5))
            following = int(np.random.lognormal(mean=6, sigma=1.0))
            baseline_post_rate = np.random.uniform(0.5, 3.0)
            
            # Risk for Individuals
            rand_risk = random.random()
            if rand_risk < 0.80: risk_band = "Low"
            elif rand_risk < 0.95: risk_band = "Medium"
            elif rand_risk < 0.99: risk_band = "High"
            else: risk_band = "Critical"
            
            if risk_band == "Critical": network_cluster_id = random.randint(1, 5)
            elif risk_band == "High": network_cluster_id = random.randint(1, 10)
            
        elif account_type == "Bot":
            followers = int(np.random.exponential(scale=50))
            following = int(np.random.uniform(100, 2000))
            baseline_post_rate = np.random.uniform(10.0, 50.0)
            risk_band = "High" if random.random() < 0.5 else "Low"
            network_cluster_id = random.randint(1, 20)
            
        elif account_type == "Organization":
            followers = int(np.random.lognormal(mean=9, sigma=1.0))
            following = int(np.random.uniform(10, 500))
            baseline_post_rate = np.random.uniform(1.0, 5.0)
            risk_band = "Low"
            
        elif account_type == "Government":
            followers = int(np.random.lognormal(mean=11, sigma=1.0))
            following = int(np.random.uniform(0, 50))
            baseline_post_rate = np.random.uniform(2.0, 8.0)
            risk_band = "Low"
            network_cluster_id = 0 # Special cluster for govt

        # Geolocation Assignment
        counties = list(COUNTY_WEIGHTS.keys())
        weights = list(COUNTY_WEIGHTS.values())
        # Normalize weights to sum to 1 just in case, or handle remaining probability
        current_sum = sum(weights)
        if current_sum < 1.0:
            remaining = 1.0 - current_sum
            other_counties = [c for c in COUNTY_COORDINATES.keys() if c not in counties]
            if other_counties:
                navg = remaining / len(other_counties)
                for c in other_counties:
                    counties.append(c)
                    weights.append(navg)
        
        
        home_county = random.choices(counties, weights=weights, k=1)[0]
        
        # Device Assignment
        if account_type == "Individual":
            # 80% Android (Ground), 20% iPhone (Slayqueen/Elite)
            if risk_band == "Low" and random.random() < 0.2:
                 primary_device = random.choice(DEVICE_TYPES["High_End"])
            else:
                 primary_device = random.choice(DEVICE_TYPES["Standard"])
                 
        elif account_type == "Bot":
            # Web App (Scripted) or Android (Click farms)
            primary_device = random.choice(DEVICE_TYPES["Automated"])
            
        elif account_type == "Organization":
            primary_device = random.choice(DEVICE_TYPES["Official"])
            
        elif account_type == "Government":
            primary_device = random.choice(DEVICE_TYPES["Official"])
            if "iPhone" in primary_device and random.random() < 0.5:
                # Narrative: "Tweet from iPhone complaining about poverty"
                pass 

        return {
            "account_id": account_id,
            "account_type": account_type,
            "account_age_days": account_age_days,
            "followers_count": followers,
            "following_count": following,
            "baseline_post_rate": baseline_post_rate,
            "network_cluster_id": network_cluster_id, # Can still correspond to political affiliation
            "risk_band": risk_band, # Ground truth label
            "language_mix_ratio": np.random.beta(2, 2), # Sheng/English mix
            "home_county": home_county,
            "primary_device": primary_device
        }
