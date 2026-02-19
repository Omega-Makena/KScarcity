
import random
import numpy as np
from datetime import timedelta, datetime


class BehaviorSimulator:
    def __init__(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)

    def generate_activity_schedule(self, account, start_date, duration_days, scenario_manager=None):
        """
        Generates a schedule of posts for an account over the given duration.
        Returns a list of dicts with: timestamp, intent, is_coordination, state,
        and optional policy_event_id, policy_phase, stance.
        """
        timestamps = []
        current_time = start_date
        
        # Risk-based parameters
        risk_band = account["risk_band"]
        baseline_rate = account["baseline_post_rate"]
        home_county = account.get("home_county", "Nairobi")
        account_type = account.get("account_type", "Individual")
        
        # Behavior State Machine
        # States: NORMAL, ESCALATING, IDLE, RECOVERING
        state = "NORMAL"
        escalation_start = None
        
        for day in range(duration_days):
            daily_rate = baseline_rate
            
            # 1. Check for Active Crisis Scenario Events
            active_events = []
            if scenario_manager:
                active_events = scenario_manager.get_active_events(current_time)
                
            # Apply Scenario-based Rate Modifiers (e.g., Silence)
            for event in active_events:
                if event["type"] == "silence" and risk_band in event["target_risk"]:
                    daily_rate *= 0.1 # Significant drop
                elif event["type"] == "migration_signal" and risk_band in event["target_risk"]:
                     daily_rate *= 0.5 # Moving to encrypted channels
            
            # 2. Check for Active Policy Events (boost tweet rate)
            active_policy_events = []
            if scenario_manager and hasattr(scenario_manager, "get_active_policy_events"):
                active_policy_events = scenario_manager.get_active_policy_events(current_time)
                # Policy events boost posting rate based on phase intensity
                # Use diminishing returns: each additional event adds less
                for idx, (_pe, phase) in enumerate(active_policy_events):
                    boost = baseline_rate * phase.tweet_intensity * 0.15 / (1 + idx * 0.5)
                    daily_rate += boost

            # Apply state modifiers
            if state == "ESCALATING":
                daily_rate *= random.uniform(2.0, 5.0) # Burst activity
                if random.random() < 0.2: # Chance to burn out
                    state = "RECOVERING"
            elif state == "RECOVERING":
                daily_rate *= 0.1
                if random.random() < 0.3:
                    state = "NORMAL"
            elif state == "NORMAL":
                if risk_band in ["High", "Critical"] and random.random() < 0.1:
                    state = "ESCALATING"
                    escalation_start = current_time
            
            # Calculate number of posts for today (Poisson process approximation)
            num_posts = np.random.poisson(daily_rate)
            
            for _ in range(num_posts):
                # Distribute posts throughout the day (favoring active hours 8am-10pm)
                hour = int(np.random.beta(5, 2) * 24)
                minute = random.randint(0, 59)
                second = random.randint(0, 59)
                post_time = current_time + timedelta(hours=hour, minutes=minute, seconds=second)
                
                # Determine Intent
                intent = "casual"
                is_coordination = False
                policy_event_id = None
                policy_phase = None
                stance = None
                
                # ── Policy Event Override (checked first) ─────────────
                policy_override = False
                if active_policy_events and scenario_manager:
                    for pe, phase in active_policy_events:
                        if scenario_manager.should_account_react_to_policy(pe, phase, account):
                            policy_event_id = pe.event_id
                            policy_phase = phase.value
                            stance = scenario_manager.get_policy_stance(phase, account_type)
                            intent = scenario_manager.get_policy_intent(
                                phase, risk_band, account_type
                            ) or "frustration"
                            if intent in ("mobilization", "coordination"):
                                is_coordination = True
                            policy_override = True
                            break  # first matching policy event wins

                # ── Crisis Scenario Override ───────────────────────────
                scenario_override = False
                if not policy_override:
                    for event in active_events:
                        # Check targeting mapping
                        county_match = not event["target_counties"] or home_county in event["target_counties"]
                        risk_match = not event["target_risk"] or risk_band in event["target_risk"]
                        
                        if county_match and risk_match and random.random() < event["intensity"]:
                            intent = event["type"]
                            scenario_override = True
                            if event["type"] in ["mobilization", "migration_signal"]:
                                 is_coordination = True
                            break # First matching event wins
                
                if not policy_override and not scenario_override:
                    # Default logic fallback
                    if state == "ESCALATING":
                        rand = random.random()
                        if rand < 0.25:
                            intent = "escalation"
                        elif rand < 0.45:
                            intent = "mobilization"
                        elif rand < 0.60:
                            intent = "satire_mockery" 
                        elif rand < 0.75:
                            # New Phase 7 Threat Signals
                            threat_rand = random.random()
                            if threat_rand < 0.4: intent = "rumor_mill"
                            elif threat_rand < 0.7: intent = "infrastructure_stress"
                            else: intent = "migration_signal"
                        else:
                            intent = "opinion"
                            
                        if risk_band == "Critical":
                             is_coordination = random.random() < 0.3
    
                    elif risk_band in ["High", "Critical"]:
                        rand = random.random()
                        if rand < 0.2:
                            intent = "opinion"
                        elif rand < 0.4:
                            intent = "satire_mockery" 
                        elif rand < 0.5:
                            intent = "rumor_mill" # Sowing confusion
                        elif rand < 0.55:
                            intent = "migration_signal" # Going dark indicators
                            
                    elif risk_band == "Medium":
                        if random.random() < 0.15:
                            intent = "satire_mockery" 
                        elif random.random() < 0.05:
                            intent = "infrastructure_stress" # Complaints about services
                
                timestamps.append({
                    "timestamp": post_time,
                    "intent": intent,
                    "is_coordination": is_coordination,
                    "state": state,
                    "policy_event_id": policy_event_id,
                    "policy_phase": policy_phase,
                    "stance": stance,
                })
            
            current_time += timedelta(days=1)
            
        timestamps.sort(key=lambda x: x["timestamp"])
        return timestamps

    def calculate_trajectory_metrics(self, activity_log):
        """
        Derives Layer 3 metrics from the activity log.
        """
        if not activity_log:
            return {
                "escalation_velocity": 0,
                "escalation_acceleration": 0
            }
        
        # Simple heuristic: Rate of "escalation" intents over time
        escalation_events = [1 if x["intent"] in ["escalation", "mobilization"] else 0 for x in activity_log]
        
        # Velocity = average daily count of escalation events
        total_days = (activity_log[-1]["timestamp"] - activity_log[0]["timestamp"]).days + 1
        velocity = sum(escalation_events) / max(1, total_days)
        
        return {
            "escalation_velocity": round(velocity, 3),
            "escalation_acceleration": 0.0 # Placeholder for more complex calculation
        }
