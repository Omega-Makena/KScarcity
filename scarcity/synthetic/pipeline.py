
import pandas as pd
from datetime import datetime
import os
import random
import numpy as np
from .accounts import AccountGenerator
from .content import ContentGenerator
from .behavior import BehaviorSimulator
from .vocabulary import COUNTY_COORDINATES, INTERACTION_WEIGHTS
from .scenarios import ScenarioManager

class SyntheticPipeline:
    def __init__(self, output_dir="data/synthetic"):
        self.output_dir = output_dir
        self.account_gen = AccountGenerator()
        self.content_gen = ContentGenerator()
        self.behavior_sim = BehaviorSimulator()
        
    def run(self, num_accounts=100, duration_days=30, start_date=None):
        if start_date is None:
            start_date = datetime.now()
            
        print(f"Generating {num_accounts} accounts...")
        accounts = self.account_gen.generate_accounts(num_accounts)
        
        print(f"Simulating behavior over {duration_days} days...")
        
        # Initialize Scenario Manager (Phase 8)
        scenario_manager = ScenarioManager(start_date, duration_days)
        print("Initialized Scenario Engine with events:", [e["name"] for e in scenario_manager.events])
        
        # Sort all activities by time to simulate chronological flow for interactions
        all_activities = []
        for account in accounts:
            # Pass scenario_manager to behavior_sim
            activity_log = self.behavior_sim.generate_activity_schedule(account, start_date, duration_days, scenario_manager)
            for act in activity_log:
                act["account"] = account # Attach account to activity
                all_activities.append(act)
        
        # Sort by timestamp to allow "past" referencing
        all_activities.sort(key=lambda x: x["timestamp"])
        
        all_tweets = []
        recent_tweets = [] # Buffer for interactions
        
        print(f"Generating content for {len(all_activities)} activities...")
        
        for i, activity in enumerate(all_activities):
            account = activity["account"]
            intent = activity["intent"]
            
            # Determine Interaction Type
            acc_type = account.get("account_type", "Individual")
            weights = INTERACTION_WEIGHTS.get(acc_type, INTERACTION_WEIGHTS["Individual"])
            interaction_type = random.choices(list(weights.keys()), weights=list(weights.values()), k=1)[0]
            
            # Interact if possible (needs recent tweets)
            ref_tweet = None
            if interaction_type in ["Retweet", "Reply", "Quote"] and len(recent_tweets) > 10:
                # Bots amplify high risk content, others random
                if acc_type == "Bot":
                     # Bots target High Risk / Escalation tweets
                     candidates = [t for t in recent_tweets[-200:] if t.get("escalation_score", 0) > 0.5]
                     if candidates: ref_tweet = random.choice(candidates)
                     else: ref_tweet = random.choice(recent_tweets[-100:])
                else:
                     ref_tweet = random.choice(recent_tweets[-100:])
            
            # Fallback to Tweet if no ref_tweet found
            if not ref_tweet:
                interaction_type = "Tweet"

            # Generate Content
            if interaction_type == "Retweet":
                tweet_text = f"RT @{ref_tweet['account_id'][:8]}: {ref_tweet['text']}"
                scores = {k: v for k, v in ref_tweet.items() if "score" in k or "_rate" in k} # Inherit scores
                
            elif interaction_type == "Reply":
                reply_text = self.content_gen.generate_tweet(account, intent=intent)
                tweet_text = f"@{ref_tweet['account_id'][:8]} {reply_text}"
                scores = self.content_gen.calculate_scores(reply_text, intent)
                
            elif interaction_type == "Quote":
                quote_text = self.content_gen.generate_tweet(account, intent=intent)
                tweet_text = f"{quote_text} QT @{ref_tweet['account_id'][:8]}: {ref_tweet['text'][:50]}..."
                scores = self.content_gen.calculate_scores(quote_text, intent)
                
            else: # Normal Tweet
                tweet_text = self.content_gen.generate_tweet(account, intent=intent)
                scores = self.content_gen.calculate_scores(tweet_text, intent)

            # Metadata & Metrics
            source = account.get("primary_device", "Twitter for Android")
            
            # Simulated Metrics (Power Lawish)
            followers = account.get("followers_count", 100)
            base_engagement = np.random.lognormal(mean=np.log(max(1, followers/100)), sigma=1.0)
            
            # Adjust metrics by interaction type
            likes = int(base_engagement)
            retweets = int(base_engagement * 0.3)
            if acc_type == "Bot": 
                likes = int(likes * 0.1) # Bots don't get likes
                retweets = int(retweets * 2.0) # But they get RTs (bot rings)
            
            # Geolocation (County Level + Jitter)
            home_county = account.get("home_county", "Nairobi")
            base_lat, base_lon = COUNTY_COORDINATES.get(home_county, (-1.2921, 36.8219))
            lat = base_lat + random.uniform(-0.04, 0.04)
            lon = base_lon + random.uniform(-0.04, 0.04)

            tweet_record = {
                "post_id": f"tweet_{len(all_tweets)+1}",
                "account_id": account["account_id"],
                "timestamp": activity["timestamp"],
                "text": tweet_text,
                "intent": intent,
                "interaction_type": interaction_type,
                "reply_to_post_id": ref_tweet["post_id"] if ref_tweet else None,
                "source_label": source,
                "like_count": likes,
                "retweet_count": retweets,
                "location_county": home_county,
                "latitude": round(lat, 6),
                "longitude": round(lon, 6),
                **scores
            }
            all_tweets.append(tweet_record)
            
            # Update buffer
            if interaction_type == "Tweet": # Only index original tweets for easier flow
                recent_tweets.append(tweet_record)
            
        # Calculate Trajectory Metrics (Needs regrouping by account)
        # Re-using existing logic but simpler loop
        account_metrics = []
        for account in accounts:
            # Filter activities for this account to calc metrics
            # This is a bit inefficient but fine for <10k tweets
            acc_activities = [x for x in all_activities if x["account"]["account_id"] == account["account_id"]]
            metrics = self.behavior_sim.calculate_trajectory_metrics(acc_activities)
            account_record = {**account, **metrics}
            account_metrics.append(account_record)
            
        # Create DataFrames
        df_accounts = pd.DataFrame(account_metrics)
        df_tweets = pd.DataFrame(all_tweets)
        
        # Save to CSV
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        df_accounts.to_csv(f"{self.output_dir}/accounts.csv", index=False)
        df_tweets.to_csv(f"{self.output_dir}/tweets.csv", index=False)
        
        print(f"Successfully generated {len(df_accounts)} accounts and {len(df_tweets)} tweets.")
        print(f"Files saved to {self.output_dir}")

if __name__ == "__main__":
    pipeline = SyntheticPipeline()
    pipeline.run(num_accounts=100, duration_days=30)
