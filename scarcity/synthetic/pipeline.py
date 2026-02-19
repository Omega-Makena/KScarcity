
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
from .policy_events import PolicyEventInjector


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
        
        # Initialize Scenario Manager (includes policy events)
        scenario_manager = ScenarioManager(start_date, duration_days)
        print("Initialized Scenario Engine with crisis events:", [e["name"] for e in scenario_manager.events])
        print("Initialized Policy Engine with events:", scenario_manager.policy_injector.event_ids)
        
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
        
        # Stats counters
        policy_tweet_count = 0
        crisis_tweet_count = 0
        organic_tweet_count = 0
        
        print(f"Generating content for {len(all_activities)} activities...")
        
        for i, activity in enumerate(all_activities):
            account = activity["account"]
            intent = activity["intent"]
            policy_event_id = activity.get("policy_event_id")
            policy_phase = activity.get("policy_phase")
            stance = activity.get("stance")
            
            # Determine Interaction Type
            acc_type = account.get("account_type", "Individual")
            weights = INTERACTION_WEIGHTS.get(acc_type, INTERACTION_WEIGHTS["Individual"])
            interaction_type = random.choices(list(weights.keys()), weights=list(weights.values()), k=1)[0]
            
            # Interact if possible (needs recent tweets)
            ref_tweet = None
            if interaction_type in ["Retweet", "Reply", "Quote"] and len(recent_tweets) > 10:
                # Context-aware referencing: prefer tweets about same policy event
                if policy_event_id and acc_type == "Bot":
                    # Bots amplify same-policy-event tweets
                    candidates = [t for t in recent_tweets[-200:]
                                  if t.get("policy_event_id") == policy_event_id]
                    if not candidates:
                        candidates = [t for t in recent_tweets[-200:]
                                      if t.get("escalation_score", 0) > 0.5]
                    if candidates:
                        ref_tweet = random.choice(candidates)
                    else:
                        ref_tweet = random.choice(recent_tweets[-100:])
                elif acc_type == "Bot":
                     # Bots target High Risk / Escalation tweets
                     candidates = [t for t in recent_tweets[-200:] if t.get("escalation_score", 0) > 0.5]
                     if candidates: ref_tweet = random.choice(candidates)
                     else: ref_tweet = random.choice(recent_tweets[-100:])
                else:
                     ref_tweet = random.choice(recent_tweets[-100:])
            
            # Fallback to Tweet if no ref_tweet found
            if not ref_tweet:
                interaction_type = "Tweet"

            # ── Generate Content ──────────────────────────────────────
            is_policy_tweet = policy_event_id is not None

            if is_policy_tweet:
                # Look up the actual PolicyEvent object via scenario_manager
                pe = scenario_manager.policy_injector.get_event_by_id(policy_event_id)
                from .policy_events import PolicyPhase
                phase_enum = PolicyPhase(policy_phase)

            if interaction_type == "Retweet":
                tweet_text = f"RT @{ref_tweet['account_id'][:8]}: {ref_tweet['text']}"
                # Inherit scores from original tweet
                scores = {k: v for k, v in ref_tweet.items() if "score" in k or "_rate" in k
                          or k in ("policy_event_id", "policy_phase", "topic_cluster",
                                   "stance_score", "sentiment_score", "policy_severity")}
                
            elif interaction_type == "Reply":
                if is_policy_tweet and pe:
                    reply_text = self.content_gen.generate_policy_tweet(
                        account, pe, phase_enum, stance or "neutral"
                    )
                    scores = self.content_gen.calculate_policy_scores(
                        reply_text, pe, phase_enum, stance or "neutral"
                    )
                else:
                    reply_text = self.content_gen.generate_tweet(account, intent=intent)
                    scores = self.content_gen.calculate_scores(reply_text, intent)
                tweet_text = f"@{ref_tweet['account_id'][:8]} {reply_text}"
                
            elif interaction_type == "Quote":
                if is_policy_tweet and pe:
                    quote_text = self.content_gen.generate_policy_tweet(
                        account, pe, phase_enum, stance or "neutral"
                    )
                    scores = self.content_gen.calculate_policy_scores(
                        quote_text, pe, phase_enum, stance or "neutral"
                    )
                else:
                    quote_text = self.content_gen.generate_tweet(account, intent=intent)
                    scores = self.content_gen.calculate_scores(quote_text, intent)
                tweet_text = f"{quote_text} QT @{ref_tweet['account_id'][:8]}: {ref_tweet['text'][:50]}..."
                
            else:  # Normal Tweet
                if is_policy_tweet and pe:
                    tweet_text = self.content_gen.generate_policy_tweet(
                        account, pe, phase_enum, stance or "neutral"
                    )
                    scores = self.content_gen.calculate_policy_scores(
                        tweet_text, pe, phase_enum, stance or "neutral"
                    )
                    policy_tweet_count += 1
                else:
                    tweet_text = self.content_gen.generate_tweet(account, intent=intent)
                    scores = self.content_gen.calculate_scores(tweet_text, intent)
                    if activity.get("state") == "ESCALATING":
                        crisis_tweet_count += 1
                    else:
                        organic_tweet_count += 1

            # ── Ensure policy fields exist (defaults for non-policy tweets) ──
            scores.setdefault("sentiment_score", 0.0)
            scores.setdefault("stance_score", 0.0)
            scores.setdefault("policy_event_id", None)
            scores.setdefault("policy_phase", None)
            scores.setdefault("topic_cluster", None)
            scores.setdefault("policy_severity", 0.0)

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
                # Base threat scores
                "imperative_rate": scores.get("imperative_rate", 0.0),
                "urgency_rate": scores.get("urgency_rate", 0.0),
                "coordination_score": scores.get("coordination_score", 0.0),
                "escalation_score": scores.get("escalation_score", 0.0),
                "threat_score": scores.get("threat_score", 0.0),
                # Policy-tracing fields (new)
                "sentiment_score": scores.get("sentiment_score", 0.0),
                "stance_score": scores.get("stance_score", 0.0),
                "policy_event_id": scores.get("policy_event_id"),
                "policy_phase": scores.get("policy_phase"),
                "topic_cluster": scores.get("topic_cluster"),
                "policy_severity": scores.get("policy_severity", 0.0),
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
        
        print(f"\nSuccessfully generated {len(df_accounts)} accounts and {len(df_tweets)} tweets.")
        print(f"  Policy-reaction tweets: {policy_tweet_count} ({policy_tweet_count*100//max(1,len(df_tweets))}%)")
        print(f"  Crisis-scenario tweets: {crisis_tweet_count}")
        print(f"  Organic tweets:         {organic_tweet_count}")
        
        # Policy event coverage summary
        if "policy_event_id" in df_tweets.columns:
            policy_df = df_tweets[df_tweets["policy_event_id"].notna()]
            if len(policy_df) > 0:
                print(f"\nPolicy event coverage:")
                for eid, group in policy_df.groupby("policy_event_id"):
                    phases = group["policy_phase"].value_counts().to_dict()
                    stances = group["stance_score"].apply(
                        lambda x: "anti" if x < 0 else ("pro" if x > 0 else "neutral")
                    ).value_counts().to_dict()
                    print(f"  {eid}: {len(group)} tweets | phases: {phases} | stances: {stances}")
        
        print(f"\nFiles saved to {self.output_dir}")

if __name__ == "__main__":
    pipeline = SyntheticPipeline()
    pipeline.run(num_accounts=100, duration_days=30)
