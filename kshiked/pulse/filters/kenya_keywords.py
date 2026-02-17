# Kenya-focused filter keywords
"""
Kenya-Focused Keywords for KShield Pulse

Collections of keywords for filtering and prioritizing
Kenya-related content across all platforms.

Usage:
    from filters.kenya_keywords import (
        KENYA_POLITICAL, KENYA_ECONOMIC, KENYA_THREAT_SIGNALS
    )
    
    if any(kw in text.lower() for kw in KENYA_POLITICAL):
        # This post is Kenya politics related
        pass
"""

from typing import List, Set


# =============================================================================
# Political Keywords
# =============================================================================

KENYA_POLITICAL: List[str] = [
    # Current leadership (2024-2026)
    "ruto", "william ruto", "president ruto",
    "gachagua", "rigathi gachagua", 
    "raila", "raila odinga", "baba",
    "uhuru", "uhuru kenyatta",
    "kalonzo", "mudavadi", "wetangula",
    
    # Political parties and coalitions
    "kenya kwanza", "kenyakwanza",
    "azimio", "azimio la umoja",
    "uda", "united democratic alliance",
    "odm", "orange democratic movement",
    "jubilee", "jubilee party",
    
    # Political terms
    "hustler", "hustler nation", "bottom up",
    "tangatanga", "kieleweke",
    "handshake", "bomas",
    
    # Government institutions
    "parliament", "bunge", "state house",
    "judiciary", "iebc", "electoral commission",
    "senate", "national assembly",
    "county assembly", "mca", "mp", "governor",
    
    # Political events
    "election", "by-election", "referendum",
    "cabinet", "cs", "cabinet secretary",
]

# =============================================================================
# Economic Keywords
# =============================================================================

KENYA_ECONOMIC: List[str] = [
    # Cost of living
    "cost of living", "bei ya maisha",
    "inflation", "mfumuko wa bei",
    "unga", "unga prices", "maize flour",
    "sukari", "sugar prices",
    "fuel", "fuel prices", "petrol", "diesel",
    "cooking gas", "gas prices", "lpg",
    "electricity", "kplc", "power bills",
    "rent", "house rent", "keja",
    
    # Economic terms
    "economy", "gdp", "unemployment",
    "tax", "taxes", "kra", "etims",
    "budget", "national budget",
    "shilling", "dollar rate", "forex",
    "debt", "imf", "world bank",
    
    # Business & trade
    "business", "biashara",
    "import", "export", "sgr",
    "kebs", "standards",
    
    # Financial hardship
    "broke", "pesa hakuna", "no money",
    "struggling", "suffering",
    "poverty", "hungry", "njaa",
]

# =============================================================================
# Threat Signals
# =============================================================================

KENYA_THREAT_SIGNALS: List[str] = [
    # Protest/action keywords
    "maandamano", "demonstration", "protest",
    "strike", "mgomo",
    "shutdown", "lockdown",
    "walk out", "boycott",
    
    # Resistance language
    "rise up", "revolution",
    "resist", "resistance",
    "haki yetu", "our rights",
    "tunakataa", "we refuse",
    "enough", "imetosha",
    
    # Violence indicators
    "kill", "ua", "murder",
    "burn", "choma",
    "stone", "throw stones",
    "fight", "pigana",
    "war", "vita",
    
    # Threat targets
    "state house", "parliament",
    "police", "polisi", "askari",
    "government", "serikali",
    "mps", "politicians",
    
    # Coordination signals
    "join us", "come together",
    "tomorrow", "kesho",
    "everyone", "all of us",
    "meet at", "assemble",
]

# =============================================================================
# Location Keywords
# =============================================================================

KENYA_LOCATIONS: List[str] = [
    # Major cities
    "nairobi", "mombasa", "kisumu", "nakuru", "eldoret",
    "thika", "malindi", "kitale", "garissa", "nyeri",
    "machakos", "meru", "kericho", "kakamega", "kisii",
    
    # Nairobi areas
    "cbd", "westlands", "kilimani", "langata", "karen",
    "eastleigh", "githurai", "umoja", "kibera", "mathare",
    "kawangware", "kangemi", "dagoretti", "kasarani",
    "roysambu", "ruaka", "juja",
    
    # Landmarks
    "uhuru park", "freedom corner", "kencom",
    "city hall", "supreme court", "parliament road",
    
    # Counties
    "nairobi county", "kiambu county", "mombasa county",
    "kisumu county", "nakuru county", "uasin gishu",
    "kakamega county", "bungoma county", "trans nzoia",
]

# =============================================================================
# Ethnic/Tribal Keywords (for sensitivity monitoring)
# =============================================================================

KENYA_ETHNIC_SENSITIVE: List[str] = [
    # Major ethnic groups (neutral mentions)
    "kikuyu", "luo", "kalenjin", "kamba", "luhya",
    "kisii", "meru", "masai", "turkana", "somali",
    
    # Potentially divisive terms
    "tribal", "tribalism", "kabila",
    "our people", "watu wetu",
    "them", "those people",
    "outsiders", "foreigners",
]

# =============================================================================
# Utility Functions
# =============================================================================

def get_all_kenya_keywords() -> Set[str]:
    """Get all Kenya-related keywords as a set."""
    all_keywords = (
        KENYA_POLITICAL +
        KENYA_ECONOMIC +
        KENYA_THREAT_SIGNALS +
        KENYA_LOCATIONS
    )
    return set(kw.lower() for kw in all_keywords)


def contains_kenya_keyword(text: str) -> bool:
    """Check if text contains any Kenya-related keyword."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in get_all_kenya_keywords())


def get_matched_keywords(text: str) -> List[str]:
    """Get list of Kenya keywords found in text."""
    text_lower = text.lower()
    return [kw for kw in get_all_kenya_keywords() if kw in text_lower]


def is_threat_related(text: str) -> bool:
    """Check if text contains threat-related keywords."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in KENYA_THREAT_SIGNALS)


def get_threat_level_keywords(text: str) -> dict:
    """Categorize matched keywords by type."""
    text_lower = text.lower()
    
    return {
        "political": [kw for kw in KENYA_POLITICAL if kw in text_lower],
        "economic": [kw for kw in KENYA_ECONOMIC if kw in text_lower],
        "threat": [kw for kw in KENYA_THREAT_SIGNALS if kw in text_lower],
        "location": [kw for kw in KENYA_LOCATIONS if kw in text_lower],
        "ethnic": [kw for kw in KENYA_ETHNIC_SENSITIVE if kw in text_lower],
    }
