
import random
from datetime import timedelta
from scarcity.synthetic.policy_events import PolicyEventInjector


class ScenarioManager:
    def __init__(self, start_date, duration_days):
        self.start_date = start_date
        self.duration_days = duration_days
        self.events = []
        self._initialize_scenarios()

        # ── Policy Event Layer ──────────────────────────────────────────
        self.policy_injector = PolicyEventInjector(start_date=start_date)
        
    def _initialize_scenarios(self):
        """
        Defines the 'Ground Truth' storyline for the simulation.
        """
        # Event 1: Infrastructure Collapse (Nairobi) - Day 3-5
        self.events.append({
            "name": "Nairobi Grid Failure",
            "type": "infrastructure_stress",
            "start_day": 3,
            "end_day": 5,
            "target_counties": ["Nairobi"],
            "target_risk": ["Low", "Medium", "High", "Critical"], # Everyone affects
            "intensity": 0.8 # High probability of posting about this
        })
        
        # Event 2: Contagion (Spread to Neighbors) - Day 5-8
        # Spreads from Nairobi to Kiambu, Machakos, Kajiado
        self.events.append({
            "name": "Regional Protests (Contagion)",
            "type": "mobilization",
            "start_day": 5,
            "end_day": 8,
            "target_counties": ["Kiambu", "Machakos", "Kajiado"],
            "target_risk": ["Medium", "High", "Critical"],
            "intensity": 0.6
        })
        
        # Event 3: Going Dark (High Risk Groups) - Day 7 onwards
        self.events.append({
            "name": "Secure Channel Migration",
            "type": "migration_signal",
            "start_day": 7,
            "end_day": 10,
            "target_counties": [], # All
            "target_risk": ["High", "Critical"],
            "intensity": 0.9,
            "effect": "silence" # Special flag to reduce volume of other posts
        })
        
        # Event 4: False Calm (The Eye of the Storm) - Day 9
        self.events.append({
            "name": "Tactical Silence",
            "type": "silence", # Just drop volume
            "start_day": 9,
            "end_day": 9,
            "target_counties": ["Nairobi", "Mombasa"],
            "target_risk": ["Critical"],
            "intensity": 1.0
        })

    def get_active_events(self, current_date):
        """
        Returns a list of crisis scenario events active on the given date.
        """
        day_offset = (current_date - self.start_date).days
        active = []
        for event in self.events:
            if event["start_day"] <= day_offset <= event["end_day"]:
                active.append(event)
        return active

    def get_active_policy_events(self, current_date):
        """
        Returns list of (PolicyEvent, PolicyPhase) tuples active on current_date.
        Delegates to the PolicyEventInjector.
        """
        return self.policy_injector.get_active_policy_events(current_date)

    def should_account_react_to_policy(self, policy_event, phase, account):
        """
        Probabilistic check: should this account tweet about this policy event?
        """
        return self.policy_injector.should_account_react(policy_event, phase, account)

    def get_policy_intent(self, phase, risk_band, account_type="Individual"):
        """Select an appropriate tweet intent for a policy phase."""
        return self.policy_injector.select_intent(phase, risk_band, account_type)

    def get_policy_stance(self, phase, account_type="Individual"):
        """Select a stance (anti / pro / neutral) for the account."""
        return self.policy_injector.select_stance(phase, account_type)
