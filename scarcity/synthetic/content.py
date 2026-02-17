import random
from scarcity.synthetic.vocabulary import (
    THREAT_LEXICON, SLANG_CATEGORIES, POLITICAL_SPECIFIC, 
    HATE_TERMS, TEMPLATES, PRONOUNS, VERBS_FAIL, SATIRE_TERMS
)

class ContentGenerator:
    def __init__(self, seed=42):
        random.seed(seed)
    
    def generate_tweet(self, account, intent="casual"):
        """
        Generates a tweet text based on the account's persona and the intended intent.
        intent: casual, frustration, mobilization, escalation, coordination, hate_incitement
        """
        if intent not in TEMPLATES:
            intent = "casual"
            
        # Select a template
        template = random.choice(TEMPLATES[intent])
        
        # Fill placeholders with Code-Switching logic
        text = self._fill_template(template, intent)
        
        # Add "Kenyan" flavor (random interjections/slang injection)
        # High value synthetic realism: 5-15% slang tokens per post
        if random.random() < 0.4:
            text = self._inject_slang(text, intent)
            
        return text

    def _fill_template(self, template, intent):
        text = template
        
        # 1. Pronouns
        if "{pronoun}" in text:
            text = text.replace("{pronoun}", random.choice(PRONOUNS["me"] + PRONOUNS["us"]))
            
        # 2. Entities & Politics
        if "{entity}" in text:
            text = text.replace("{entity}", random.choice(POLITICAL_SPECIFIC["entities"]))
        
        # 3. Slang Categories
        for cat in ["urban_mix", "internet_youth", "political_frustration", 
                   "protest_tone", "callout_culture", "coordination_slang", 
                   "escalation_energy", "rumor_suspicion", "sarcastic_mocking"]:
            placeholder = "{" + f"slang_{cat.split('_')[0]}" + "}" # e.g. {slang_urban}
            # Simplified placeholder matching for the template keys I defined
            if f"{{slang_{cat}}}" in text: # exact match
                 text = text.replace(f"{{slang_{cat}}}", random.choice(SLANG_CATEGORIES[cat]))
            elif "{slang_" in text and cat.split('_')[0] in text: # fuzzy match attempt if needed
                 pass 

        # Handle simplified template keys from my vocabulary file definition
        if "{slang_urban}" in text: text = text.replace("{slang_urban}", random.choice(SLANG_CATEGORIES["urban_mix"]))
        if "{slang_internet}" in text: text = text.replace("{slang_internet}", random.choice(SLANG_CATEGORIES["internet_youth"]))
        if "{slang_frustration}" in text: text = text.replace("{slang_frustration}", random.choice(SLANG_CATEGORIES["political_frustration"]))
        if "{slang_sarcastic}" in text: text = text.replace("{slang_sarcastic}", random.choice(SLANG_CATEGORIES["sarcastic_mocking"]))
        if "{slang_protest}" in text: text = text.replace("{slang_protest}", random.choice(SLANG_CATEGORIES["protest_tone"]))
        if "{slang_coordination}" in text: text = text.replace("{slang_coordination}", random.choice(SLANG_CATEGORIES["coordination_slang"]))
        if "{slang_escalation}" in text: text = text.replace("{slang_escalation}", random.choice(SLANG_CATEGORIES["escalation_energy"]))

        # 4. Threat Lexicon Categories
        if "{imperative}" in text: text = text.replace("{imperative}", random.choice(THREAT_LEXICON["imperative"]))
        if "{urgency}" in text: text = text.replace("{urgency}", random.choice(THREAT_LEXICON["urgency"]))
        if "{mobilization}" in text: text = text.replace("{mobilization}", random.choice(THREAT_LEXICON["mobilization"]))
        if "{coordination}" in text: text = text.replace("{coordination}", random.choice(THREAT_LEXICON["coordination"]))
        if "{persistence}" in text: text = text.replace("{persistence}", random.choice(THREAT_LEXICON["persistence"]))
        if "{escalation}" in text: text = text.replace("{escalation}", random.choice(THREAT_LEXICON["escalation"]))
        if "{threat_like}" in text: text = text.replace("{threat_like}", random.choice(THREAT_LEXICON["threat_like"]))
        if "{crowd}" in text: text = text.replace("{crowd}", random.choice(THREAT_LEXICON["crowd_activation"]))
        
        # 5. Specifics
        if "{location}" in text: text = text.replace("{location}", random.choice(["CBD", "Parliament", "Uhuru Park", "State House", "Tao"]))
        if "{date_time}" in text: text = text.replace("{date_time}", random.choice(THREAT_LEXICON["coordination"])) # Often overlaps
        if "{hashtag}" in text: text = text.replace("{hashtag}", random.choice(POLITICAL_SPECIFIC["hashtags"]))
        if "{verb_fail}" in text: text = text.replace("{verb_fail}", random.choice(VERBS_FAIL))
        
        # 6. Hate Terms (Critical Only)
        if "{hate_term}" in text: text = text.replace("{hate_term}", random.choice(HATE_TERMS["coded"] + HATE_TERMS["harassment"]))

        # 7. Satire & Culture (New Phase 3)
        if "{satire_nickname}" in text: text = text.replace("{satire_nickname}", random.choice(SATIRE_TERMS["nicknames"]))
        if "{gen_z_proverb}" in text: text = text.replace("{gen_z_proverb}", random.choice(SATIRE_TERMS["gen_z_proverbs"]))
        if "{slang_political}" in text: text = text.replace("{slang_political}", random.choice(SATIRE_TERMS["broad_based_sarcasm"]))
        if "{the_simulation}" in text: text = text.replace("{the_simulation}", "The Simulation") # Fixed term usually
        if "{color_code}" in text: text = text.replace("{color_code}", random.choice(SATIRE_TERMS["color_codes"]))

        return text

    def _inject_slang(self, text, intent):
        """Injects a slang word at start or end for flavor."""
        slang_pool = SLANG_CATEGORIES["urban_mix"] + SLANG_CATEGORIES["internet_youth"]
        word = random.choice(slang_pool)
        if random.random() < 0.5:
            return f"{word.capitalize()} {text}"
        else:
            return f"{text} {word}"

    def calculate_scores(self, text, intent):
        """
        Calculates feature scores based on pattern presence, not just intent.
        Returns a dict of scores.
        """
        scores = {
            "imperative_rate": 0.0,
            "urgency_rate": 0.0,
            "coordination_score": 0.0,
            "escalation_score": 0.0,
            "threat_score": 0.0
        }
        
        text_lower = text.lower()
        
        # Check for presence of terms from lexicons
        # Imperative
        if any(t in text_lower for t in THREAT_LEXICON["imperative"]):
            scores["imperative_rate"] = 1.0
            
        # Urgency
        if any(t in text_lower for t in THREAT_LEXICON["urgency"]):
            scores["urgency_rate"] = 1.0
            
        # Coordination
        if any(t in text_lower for t in THREAT_LEXICON["coordination"]) or \
           any(t in text_lower for t in SLANG_CATEGORIES["coordination_slang"]):
            scores["coordination_score"] = 1.0
            
        # Escalation
        if any(t in text_lower for t in THREAT_LEXICON["escalation"]) or \
           any(t in text_lower for t in SLANG_CATEGORIES["escalation_energy"]):
            scores["escalation_score"] = 1.0
            
        # Threat / Incitement
        if any(t in text_lower for t in THREAT_LEXICON["threat_like"]) or \
           any(t in text_lower for t in HATE_TERMS["coded"]):
            scores["threat_score"] = 1.0
            
        # Base intent boosting
        if intent == "mobilization":
            scores["coordination_score"] = max(scores["coordination_score"], 0.8)
            scores["imperative_rate"] = max(scores["imperative_rate"], 0.7)
        elif intent == "escalation":
            scores["escalation_score"] = max(scores["escalation_score"], 0.9)
            scores["urgency_rate"] = max(scores["urgency_rate"], 0.8)
        elif intent == "coordination":
            scores["coordination_score"] = max(scores["coordination_score"], 0.9)
        elif intent == "infrastructure_stress":
            scores["urgency_rate"] = max(scores["urgency_rate"], 0.6)
            scores["escalation_score"] = max(scores["escalation_score"], 0.4)
        elif intent == "migration_signal":
            scores["coordination_score"] = max(scores["coordination_score"], 0.7)
            scores["urgency_rate"] = max(scores["urgency_rate"], 0.7)
        elif intent == "rumor_mill":
            scores["threat_score"] = max(scores["threat_score"], 0.3)
            scores["urgency_rate"] = max(scores["urgency_rate"], 0.5)
            
        return scores
