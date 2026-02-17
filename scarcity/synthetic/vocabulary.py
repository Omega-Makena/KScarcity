
# Kenyan Vocabulary and Threat Lexicon for Synthetic Tweet Generation

# ==========================================
# 1. THREAT-RANKING LEXICON (Operational)
# ==========================================

THREAT_LEXICON = {
    "urgency": [
        "now", "today", "immediately", "asap",
        "no more", "enough", "mwisho", "sasa", "sasa hivi",
        "this can’t wait", "cannot wait", "lazima sasa",
        "time is up", "deadline", "mwisho wa mchezo", "saa hii"
    ],
    "imperative": [
        "do it", "act", "move", "go", "come out", "toka", "amka",
        "join", "show up", "attend", "participate", "jitokeze",
        "share", "retweet", "repost", "spread", "sambaza",
        "boycott", "refuse", "stop", "block", "kataa",
        "report", "expose", "call out", "anika", "mulika"
    ],
    "mobilization": [
        "let’s gather", "meet up", "assembly", "rally", "kutana",
        "march", "protest", "demonstration", "maandamano",
        "turn up", "show up", "tuko wengi", "jitokezeni",
        "mass action", "nationwide", "countrywide", "kenya nzima"
    ],
    "coordination": [
        "at 8am", "at 9am", "on Tuesday", "Saa tatu", "CBD", "location", "venue",
        "bring", "carry", "wear", "beba", "vaa",
        "route", "meetup point", "stage", "starting point",
        "DM me", "inbox", "WhatsApp", "Telegram", "group link",
        "thread for updates", "pinned", "live updates"
    ],
    "persistence": [
        "again", "repeat", "daily", "every day", "kila siku",
        "keep pushing", "don’t stop", "continue", "songa mbele",
        "we won’t back down", "we keep going", "haturudi nyuma",
        "until they go", "mpaka kieleweke"
    ],
    "escalation": [
        "this is serious", "it’s getting worse", "mambo imeharibika",
        "we’ve tried everything", "tumechoka",
        "they won’t listen", "we must act", "lazima tujiteteee",
        "crossed the line", "last straw", "imetosha"
    ],
    "collective_identity": [
        "we the people", "wananchi", "sisi", "Wakenya",
        "together", "unity", "one voice", "pamoja",
        "our movement", "our struggle", "jeshi", "mbogi",
        "solidarity", "stand with"
    ],
    "blame": [
        "responsible", "accountable", "answer for",
        "failed us", "betrayed", "lied", "watuongopea",
        "corrupt", "theft", "impunity", "wizi", "ufisadi",
        "resign", "step down", "must go", "aende"
    ],
    "threat_like": [
        "you’ll see", "consequences", "pay for it", "mtalipa",
        "we will deal with you", "tutawafunza",
        "it won’t end well", "hamtapenda",
        "no mercy", "hakuna huruma"
    ],
    "crowd_activation": [
        "wake up", "open your eyes", "amkeni",
        "enough is enough", "imetosha",
        "it’s time", "the moment is here", "saa ni sasa",
        "share widely", "make it trend", "trendisha"
    ],
    "evasion": [
        "use code words", "don’t say it here",
        "delete after reading", "futa",
        "screenshots", "mirror", "backup accounts",
        "new account", "alt", "burner"
    ]
}

# ==========================================
# 2. SLANG & CULTURAL MARKERS
# ==========================================

SLANG_CATEGORIES = {
    "urban_mix": [
        "bana", "maze", "mazematics", "buda", "mathe", "wasee", "msee",
        "noma", "kali", "safi", "poa", "mbaya", "fiti", "legit",
        "iko aje", "aje sasa", "sawa tu", "si poa", "manze", "walai"
    ],
    "internet_youth": [
        "fr", "btw", "lowkey", "highkey", "tbh", "smh", "imo", "idk",
        "no cap", "cap", "main character", "energy", "vibes", "pressure",
        "receipts", "expose", "trending", "ratio", "pov"
    ],
    "political_frustration": [
        "hii mambo imezidi", "hatuwezi endelea hivi", "this is too much",
        "watu wamechoka", "enough is enough", "we are tired",
        "system ni mbovu", "hakuna accountability", "mchezo wa taon",
        "wanatucheza", "wanajifanya hawajui", "tunawatch"
    ],
    "protest_tone": [
        "lazima wasee watoke", "watu wataongea", "sauti lazima isikike",
        "watu wako ready", "hii ni moment", "hakuna kurudi nyuma",
        "tuko wengi", "tuko pamoja", "tunasimama", "lazima something ifanyike",
        "time imefika", "anguka nayo"
    ],
    "callout_culture": [
        "drop receipts", "leta evidence", "tunataka proof", "sema ukweli",
        "expose kila kitu", "tag them", "call them out", "don’t delete",
        "we saw everything", "salimia", "kusalimia"
    ],
    "coordination_slang": [
        "link up", "tuchanue hapo", "meetup hapo", "tuko location",
        "dm details", "tuma location", "check thread", "follow updates",
        "pin hii", "share kwa group", "tuko site"
    ],
    "escalation_energy": [
        "hii ni serious", "mambo ni moto", "situation ni mbaya",
        "pressure iko juu", "watu wamejam", "watu wako charged",
        "hii si mchezo", "sasa ni sasa", "kama mbaya mbaya"
    ],
    "rumor_suspicion": [
        "kuna kitu haiko sawa", "hii story iko off", "something is not adding up",
        "kuna mchezo inaendelea", "kuna kitu wanapanga", "watu wanajua but hawasemi",
        "they are hiding", "cover-up"
    ],
    "sarcastic_mocking": [
        "interesting...", "very convenient", "we are watching", "noted",
        "okay...", "make it make sense", "sure...", "wueh!", "alaar!"
    ]
}

# ==========================================
# 3. POLITICAL ENTITIES & SPECIFIC TERMS
# ==========================================

POLITICAL_SPECIFIC = {
    "entities": [
        "Zakayo", "Ruto", "Riggy G", "Gachagua", "Kindiki", "MPs", 
        "State House", "Parliament", "Govt", "System", "Dynasties",
        "Hustler", "Deep State", "KRA", "Police", "Afande"
    ],
    "terms": [
        "Kitu Kidogo", "Chai", "Tenderpreneur", "Handshake", "BBI",
        "Kuuma Nje", "Ground", "Vitu kwa Ground ni Different",
        "Mbele Pamoja", "Bottom-up", "Finance Bill", "Housing Levy", "SHIF"
    ],
    "hashtags": [
        "#RejectFinanceBill2026", "#RutoMustGo", "#OccupyStateHouse", "#GenZote", 
        "#SisiNiNumbers", "#Maandamano", "#KenyaKwanza", "#BottomUp", "#Dynasties",
        "#OccupyParliament", "#FinanceBill2026", "#ZakayoShuka"
    ]
}

# ==========================================
# 4. HATE SPEECH & INCITEMENT (DANGER ZONE)
# ==========================================
# These are for training models to DETECT, so they need to be realistic but handled carelessly
# We will use these sparingly in "Critical" risk profiles

HATE_TERMS = {
    "coded": [
        "madoadoa", "tugege", "fumigation", "kufumigate", "cut the tall grass",
        "adui", "bunyot", "kama noma noma", "wabara waende kwao",
        "pack and go", "mende", "kwekwe", "nyoka", "kihii", "kamwana"
    ],
    "harassment": [
        "salimia", # mild on its own, hate if combined with doxxing
        "doxxing",
        "slayqueen" # gender based
    ]
}

# ==========================================
# 5. TEMPLATES (Updated for Code-Switching)
# ==========================================

TEMPLATES = {
    "casual": [
        "{slang_urban} {pronoun} niko tu.",
        "Rada ni gani leo {slang_urban}?",
        "Mambo ni {slang_urban} tu.",
        "Leo form ni gani {slang_urban}?",
        "Hii weather inataka {slang_urban}.",
        "{slang_internet} this vibe is unmatched.",
        "Niko shwari {slang_urban}."
    ],
    "frustration": [
        "{slang_frustration}. {pronoun} {verb_fail}.",
        "{entity} {verb_fail} again. {slang_sarcastic}",
        "Imagine {slang_frustration}. {entity} must go.",
        "Bei ya unga is too high {slang_urban}. {slang_frustration}",
        "We are tired of {entity}. {slang_frustration}!"
    ],
    "mobilization": [
        "{slang_protest} {date_time}. {imperative}!",
        "{urgent} {imperative} {location}. {hashtag}",
        "We meet {location} {date_time}. {coordination}. {hashtag}",
        "{slang_coordination}. {imperative} everyone.",
        "{crowd}. {mobilization} starts {urgency}!"
    ],
    "escalation": [
        "{slang_escalation}. {threat_like}!",
        "{entity} has pushed us too far. {escalation}. {imperative}.",
        "{urgency} {imperative}. {slang_escalation}!",
        "They think we are joking. {threat_like}. {hashtag}",
        "{slang_protest}. {persistence}. {imperative}!"
    ],
    "hate_incitement": [ # Use with caution
        "Remove the {hate_term} from our land!",
        "{hate_term} must go. {imperative}!",
        "Kama noma noma. {hate_term} out!",
        "Fumigate the {hate_term}. {threat_like}."
    ],
    "satire_mockery": [
        "Mambo ni matatu: {slang_sarcastic}, {slang_sarcastic}, ama {slang_urban}.",
        "{satire_nickname} amefika. {slang_frustration}.",
        "Hiyo fire si fire. {slang_internet}",
        "Vitu kwa ground ni different. {pronoun} {verb_fail}.",
        "We are watching {the_simulation}. {slang_sarcastic}",
        "{satire_nickname} is just {color_code} in disguise. {slang_political}",
        "{gen_z_proverb}. {slang_internet}"
    ],
    "infrastructure_stress": [
        "KPLC wamechukua stima again! {slang_frustration}",
        "Net is throttling. VPN is essential now. {imperative}",
        "Roads backed up at {location}. Police roadblock. {urgency}",
        "Safaricom is acting up. M-Pesa down? {slang_internet}",
        "No water in {location} for 3 days. {slang_frustration}",
        "Blackout again. {satire_nickname} must go. {slang_political}"
    ],
    "migration_signal": [
        "Join the Telegram channel. Link in bio. {evasion}",
        "We are moving to Signal. {imperative}",
        "Don't post here anymore. They are watching. {urgency}",
        "Switch to VPN. {imperative}. {evasion}",
        "Updates available on the secure channel. {coordination}"
    ],
    "rumor_mill": [
        "I heard that {entity} is fleeing. {rumor_suspicion}",
        "Sources say {location} is surrounded. {threat_like}",
        "They are hiding the truth about {entity}. {rumor_suspicion}",
        "Leaked documents show {slang_political}. {callout_culture}",
        "Confirming reports of {escalation} at {location}. {urgency}"
    ]
}

# ==========================================
# 6. NEW 2026 CULTURAL & SATIRE TERMS
# ==========================================

SATIRE_TERMS = {
    "nicknames": [
        "Zakayo", "The Cousins", "Cuzo", "Watermelon 2.0", "The Simulation",
        "Wenye Nchi", "The Owners", "Shareholders"
    ],
    "gen_z_proverbs": [
        "Fire si fire", "Tunasoma", "Vitu kwa ground ni different", 
        "Maombi pekee haitoshi", "Fear is gone", "We are reading",
        "Salimia salamu"
    ],
    "broad_based_sarcasm": [
        "Kukula na Serikali", "Eating with the Govt", 
        "Broad-based confusion", "Handshake 2.0"
    ],
    "color_codes": [
        "Yellow", "Orange", "Red", "Green"
    ]
}

PRONOUNS = {
    "me": ["Mimi", "Me", "I", "Yangu", "Mine"],
    "us": ["Sisi", "We", "Wakenya", "Gen Zote", "Jeshi", "Mbogi", "Our movement"]
}

VERBS_FAIL = [
    "wanatuangusha", "have failed us", "are lying", "wanatucheza", "must go", 
    "have betrayed us", "are stealing", "have been bought", "wamekula na serikali"
]

# ==========================================
# 7. GEOLOCATION DEFINITIONS (County Level)
# ==========================================

COUNTY_COORDINATES = {
    "Nairobi": (-1.2921, 36.8219),
    "Mombasa": (-4.0435, 39.6682),
    "Kisumu": (-0.0917, 34.7680),
    "Nakuru": (-0.3031, 36.0800),
    "Uasin Gishu": (0.5143, 35.2698), # Eldoret
    "Kiambu": (-1.1714, 36.8356),
    "Machakos": (-1.5177, 37.2634),
    "Nyeri": (-0.4167, 36.9500),
    "Kajiado": (-1.8524, 36.7768),
    "Meru": (0.0463, 37.6559),
    "Kakamega": (0.2827, 34.7519),
    "Kisii": (-0.6817, 34.7667),
    "Bungoma": (0.5695, 34.5584),
    "Kilifi": (-3.6333, 39.8500),
    "Turkana": (3.1167, 35.6000),
    "Garissa": (-0.4536, 39.6460),
    "Narok": (-1.0783, 35.8601),
    "Kericho": (-0.3689, 35.2863),
    "Bomet": (-0.7967, 35.3000),
    "Murang'a": (-0.7210, 37.1526)
}

# Weighted for population/activity realism
COUNTY_WEIGHTS = {
    "Nairobi": 0.40,
    "Mombasa": 0.10,
    "Kiambu": 0.10,
    "Nakuru": 0.08,
    "Uasin Gishu": 0.05,
    "Kisumu": 0.05,
    "Machakos": 0.04,
    "Kajiado": 0.03,
    "Nyeri": 0.02
    # Others share the remainder
}

# ==========================================
# 8. METADATA & INTERACTIONS
# ==========================================

DEVICE_TYPES = {
    "High_End": ["Twitter for iPhone", "Twitter for iPad"],
    "Standard": ["Twitter for Android"],
    "Official": ["Twitter Media Studio", "Twitter Web App", "Twitter for iPhone"], # Govt/Orgs
    "Automated": ["Twitter Web App", "Twitter for Android", "Sprout Social", "Hootsuite"] # Bots/PR
}

# Probability of interaction type per Account Type
INTERACTION_WEIGHTS = {
    "Individual": {"Tweet": 0.6, "Retweet": 0.2, "Reply": 0.15, "Quote": 0.05},
    "Bot":        {"Tweet": 0.2, "Retweet": 0.75, "Reply": 0.05, "Quote": 0.0}, # Amplifiers
    "Organization": {"Tweet": 0.8, "Retweet": 0.1, "Reply": 0.1, "Quote": 0.0},
    "Government":   {"Tweet": 0.9, "Retweet": 0.1, "Reply": 0.0, "Quote": 0.0}
}
