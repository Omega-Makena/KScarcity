"""
Kenya Regional Geo-Mapper for Stress Signals.

Multi-source geo-resolution with confidence scoring:
1. User/Location Metadata (highest priority)
2. Hashtags (explicit, high-signal)
3. Text Mentions (noisiest, strictest checks)

Stores auditable fields: geo_county, geo_confidence, geo_source, geo_evidence, geo_candidates
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from enum import Enum
import re


class GeoSource(Enum):
    """Source of geo-resolution."""
    METADATA = "meta"
    HASHTAGS = "hashtags"
    TEXT = "text"
    NONE = "none"


# Confidence thresholds for each source
T_META = 0.7      # Metadata threshold
T_HASH = 0.6      # Hashtag threshold  
T_TEXT = 0.5      # Text mention threshold


# Kenya's 47 Counties with approximate center coordinates
KENYA_COUNTIES: Dict[str, Tuple[float, float]] = {
    "Mombasa": (-4.0435, 39.6682),
    "Kwale": (-4.1816, 39.4521),
    "Kilifi": (-3.5107, 39.9093),
    "Tana River": (-1.8167, 40.0333),
    "Lamu": (-2.2686, 40.9020),
    "Taita Taveta": (-3.3160, 38.4850),
    "Garissa": (-0.4536, 39.6401),
    "Wajir": (1.7471, 40.0573),
    "Mandera": (3.9366, 41.8670),
    "Marsabit": (2.3284, 37.9899),
    "Isiolo": (0.3556, 37.5833),
    "Meru": (0.0500, 37.6500),
    "Tharaka Nithi": (-0.3078, 37.8010),
    "Embu": (-0.5389, 37.4583),
    "Kitui": (-1.3667, 38.0167),
    "Machakos": (-1.5177, 37.2634),
    "Makueni": (-1.8039, 37.6203),
    "Nyandarua": (-0.1804, 36.5227),
    "Nyeri": (-0.4197, 36.9553),
    "Kirinyaga": (-0.5000, 37.2833),
    "Murang'a": (-0.7210, 37.1526),
    "Kiambu": (-1.1714, 36.8356),
    "Turkana": (3.1167, 35.6000),
    "West Pokot": (1.6167, 35.1167),
    "Samburu": (1.2167, 36.9333),
    "Trans Nzoia": (1.0167, 35.0000),
    "Uasin Gishu": (0.5167, 35.2833),
    "Elgeyo Marakwet": (0.8167, 35.5333),
    "Nandi": (0.1833, 35.1333),
    "Baringo": (0.4667, 35.9667),
    "Laikipia": (0.3667, 36.7833),
    "Nakuru": (-0.3031, 36.0800),
    "Narok": (-1.0833, 35.8667),
    "Kajiado": (-2.0981, 36.7820),
    "Kericho": (-0.3667, 35.2833),
    "Bomet": (-0.7833, 35.3500),
    "Kakamega": (0.2833, 34.7500),
    "Vihiga": (0.0833, 34.7167),
    "Bungoma": (0.5667, 34.5667),
    "Busia": (0.4667, 34.1167),
    "Siaya": (-0.0617, 34.2422),
    "Kisumu": (-0.1022, 34.7617),
    "Homa Bay": (-0.5273, 34.4571),
    "Migori": (-1.0634, 34.4731),
    "Kisii": (-0.6817, 34.7667),
    "Nyamira": (-0.5633, 34.9350),
    "Nairobi": (-1.2921, 36.8219),
}

# Primary county keywords (direct county names)
COUNTY_PRIMARY: Dict[str, List[str]] = {
    county: [county.lower()] for county in KENYA_COUNTIES
}
# Add alternate spellings
COUNTY_PRIMARY["Murang'a"].extend(["muranga", "murang'a county"])

# Secondary keywords (towns, neighborhoods) - lower confidence
COUNTY_SECONDARY: Dict[str, List[str]] = {
    "Nairobi": ["cbd", "westlands", "kibera", "mathare", "eastleigh", "uhuru park", 
                "parliament", "kasarani", "embakasi", "langata", "karen", "lavington"],
    "Mombasa": ["likoni", "nyali", "kilindini", "old town", "bamburi", "shanzu"],
    "Kisumu": ["kondele", "kibuye", "mamboleo", "nyalenda"],
    "Nakuru": ["naivasha", "gilgil", "molo", "njoro", "subukia"],
    "Garissa": ["dadaab", "balambala", "fafi", "ijara"],
    "Turkana": ["kakuma", "lodwar", "lokichoggio", "lokichar"],
    "Mandera": ["elwak", "banissa", "lafey"],
    "Wajir": ["habaswein", "wajir bor", "tarbaj"],
    "Marsabit": ["moyale", "sololo", "laisamis"],
    "Kajiado": ["ngong", "kitengela", "ongata rongai", "loitokitok", "namanga"],
    "Kiambu": ["thika", "ruiru", "juja", "limuru", "kikuyu", "gatundu", "githunguri"],
    "Machakos": ["athi river", "mlolongo", "kangundo", "matuu"],
    "Nyeri": ["karatina", "othaya", "mukurweini"],
    "Meru": ["nkubu", "timau", "isiolo junction"],
    "Kakamega": ["butere", "mumias", "lurambi", "malava"],
    "Bungoma": ["webuye", "kimilili", "sirisia"],
    "Kisii": ["ogembo", "suneka", "keroka junction"],
    "Migori": ["rongo", "awendo", "isebania"],
    "Homa Bay": ["mbita", "kendu bay", "oyugis"],
    "Siaya": ["bondo", "ugunja", "yala"],
    "Uasin Gishu": ["eldoret", "burnt forest", "turbo", "moiben"],
    "Trans Nzoia": ["kitale", "endebess"],
    "West Pokot": ["kapenguria", "makutano", "chepareria"],
    "Baringo": ["kabarnet", "marigat", "mogotio"],
    "Laikipia": ["nanyuki", "rumuruti", "nyahururu"],
    "Nyandarua": ["ol kalou", "engineer", "kinangop"],
    "Kirinyaga": ["kerugoya", "wang'uru", "kutus"],
    "Embu": ["runyenjes", "siakago"],
    "Tharaka Nithi": ["chuka", "chogoria"],
    "Kitui": ["mwingi", "mutomo"],
    "Makueni": ["wote", "sultan hamud", "emali"],
    "Taita Taveta": ["voi", "taveta", "wundanyi", "mwatate"],
    "Kwale": ["diani", "ukunda", "msambweni", "lunga lunga"],
    "Kilifi": ["malindi", "watamu", "mtwapa", "mariakani"],
    "Tana River": ["hola", "garsen", "bura"],
    "Lamu": ["mpeketoni", "hindi", "witu"],
    "Isiolo": ["garbatulla", "merti", "kula mawe"],
    "Samburu": ["maralal", "wamba", "archer's post"],
    "Narok": ["narok town", "maasai mara", "kilgoris", "ole ntutu"],
    "Bomet": ["sotik", "longisa", "bomet town"],
    "Kericho": ["londiani", "litein", "kipkelion"],
    "Nandi": ["kapsabet", "nandi hills", "mosoriot"],
    "Elgeyo Marakwet": ["iten", "kapsowar", "tambach"],
    "Vihiga": ["maragoli", "mbale", "luanda"],
    "Busia": ["malaba", "funyula", "nambale"],
    "Nyamira": ["keroka", "ekerenyo"],
}


@dataclass
class GeoCandidate:
    """A geo-resolution candidate."""
    county: str
    confidence: float
    source: GeoSource
    evidence: str  # The matched string


@dataclass 
class GeoResolution:
    """Complete geo-resolution result with audit trail."""
    geo_county: Optional[str] = None
    geo_confidence: float = 0.0
    geo_source: GeoSource = GeoSource.NONE
    geo_evidence: str = ""
    geo_candidates: List[GeoCandidate] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "geo_county": self.geo_county,
            "geo_confidence": self.geo_confidence,
            "geo_source": self.geo_source.value,
            "geo_evidence": self.geo_evidence,
            "geo_candidates": [
                {"county": c.county, "confidence": c.confidence, 
                 "source": c.source.value, "evidence": c.evidence}
                for c in self.geo_candidates[:3]  # Top 3
            ]
        }


@dataclass
class CountySignal:
    """Signal data for a specific county."""
    county: str
    signal_count: int = 0
    risk_score: float = 0.0
    signals: List[str] = field(default_factory=list)
    resolutions: List[GeoResolution] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def add_signal(self, signal_type: str, magnitude: float = 0.1, 
                   resolution: Optional[GeoResolution] = None):
        """Add a signal to this county."""
        self.signal_count += 1
        self.signals.append(signal_type)
        if resolution:
            self.resolutions.append(resolution)
        # Weighted average for risk score
        self.risk_score = min(1.0, self.risk_score + magnitude * 0.15)
        self.last_updated = datetime.now()


class KenyaGeoResolver:
    """
    Multi-source geo-resolver for Kenya counties.
    
    Resolution priority:
    1. User/Location Metadata (if conf >= T_META)
    2. Hashtags (if conf >= T_HASH)
    3. Text Mentions (if conf >= T_TEXT)
    4. UNKNOWN
    """
    
    def __init__(
        self,
        t_meta: float = T_META,
        t_hash: float = T_HASH,
        t_text: float = T_TEXT,
    ):
        self.t_meta = t_meta
        self.t_hash = t_hash
        self.t_text = t_text
        
        # Build lookup indices
        self._build_indices()
    
    def _build_indices(self):
        """Build fast lookup indices for matching."""
        # Primary patterns (exact county names) - high confidence
        self._primary_patterns: Dict[str, str] = {}
        for county, keywords in COUNTY_PRIMARY.items():
            for kw in keywords:
                self._primary_patterns[kw.lower()] = county
        
        # Secondary patterns (towns, neighborhoods) - lower confidence
        self._secondary_patterns: Dict[str, str] = {}
        for county, keywords in COUNTY_SECONDARY.items():
            for kw in keywords:
                self._secondary_patterns[kw.lower()] = county
    
    def resolve(
        self,
        text: str = "",
        user_location: Optional[str] = None,
        hashtags: Optional[List[str]] = None,
    ) -> GeoResolution:
        """
        Resolve location from multiple sources.
        
        Args:
            text: Post text content
            user_location: User profile location (metadata)
            hashtags: List of hashtags from the post
            
        Returns:
            GeoResolution with county, confidence, source, and candidates
        """
        candidates: List[GeoCandidate] = []
        
        # 1. Try metadata first
        meta_result = self._resolve_metadata(user_location)
        if meta_result:
            candidates.append(meta_result)
        
        # 2. Try hashtags
        hash_results = self._resolve_hashtags(hashtags or [])
        candidates.extend(hash_results)
        
        # 3. Try text mentions
        text_results = self._resolve_text(text)
        candidates.extend(text_results)
        
        # Sort by confidence
        candidates.sort(key=lambda c: c.confidence, reverse=True)
        
        # Apply priority rules
        chosen = self._choose_best(candidates)
        
        return GeoResolution(
            geo_county=chosen.county if chosen else None,
            geo_confidence=chosen.confidence if chosen else 0.0,
            geo_source=chosen.source if chosen else GeoSource.NONE,
            geo_evidence=chosen.evidence if chosen else "",
            geo_candidates=candidates[:3],  # Top 3 for debugging
        )
    
    def _resolve_metadata(self, user_location: Optional[str]) -> Optional[GeoCandidate]:
        """Resolve from user metadata."""
        if not user_location:
            return None
        
        loc_lower = user_location.lower().strip()
        
        # Check for "Kenya" to validate it's actually Kenya
        is_kenya = "kenya" in loc_lower or "ke" in loc_lower.split(",")
        
        # Try primary matches (county names)
        for pattern, county in self._primary_patterns.items():
            if pattern in loc_lower:
                conf = 0.95 if is_kenya else 0.80
                return GeoCandidate(
                    county=county,
                    confidence=conf,
                    source=GeoSource.METADATA,
                    evidence=f"metadata:{user_location}"
                )
        
        # Try secondary matches (towns)
        for pattern, county in self._secondary_patterns.items():
            if pattern in loc_lower:
                conf = 0.85 if is_kenya else 0.70
                return GeoCandidate(
                    county=county,
                    confidence=conf,
                    source=GeoSource.METADATA,
                    evidence=f"metadata:{user_location}"
                )
        
        return None
    
    def _resolve_hashtags(self, hashtags: List[str]) -> List[GeoCandidate]:
        """Resolve from hashtags."""
        results = []
        
        for tag in hashtags:
            tag_lower = tag.lower().replace("#", "").strip()
            
            # Try primary matches
            for pattern, county in self._primary_patterns.items():
                if pattern in tag_lower or tag_lower == pattern.replace(" ", ""):
                    results.append(GeoCandidate(
                        county=county,
                        confidence=0.90,
                        source=GeoSource.HASHTAGS,
                        evidence=f"hashtag:#{tag}"
                    ))
            
            # Try secondary matches
            for pattern, county in self._secondary_patterns.items():
                pattern_nospace = pattern.replace(" ", "").replace("'", "")
                if pattern_nospace in tag_lower or tag_lower == pattern_nospace:
                    results.append(GeoCandidate(
                        county=county,
                        confidence=0.75,
                        source=GeoSource.HASHTAGS,
                        evidence=f"hashtag:#{tag}"
                    ))
        
        return results
    
    def _resolve_text(self, text: str) -> List[GeoCandidate]:
        """Resolve from text content with strict checks."""
        results = []
        text_lower = text.lower()
        
        # Word boundary pattern for stricter matching
        def word_match(pattern: str, text: str) -> bool:
            """Check if pattern exists as a word (not substring)."""
            pattern_escaped = re.escape(pattern)
            return bool(re.search(rf'\b{pattern_escaped}\b', text, re.IGNORECASE))
        
        # Try primary matches (strictest - require word boundary)
        for pattern, county in self._primary_patterns.items():
            if word_match(pattern, text_lower):
                results.append(GeoCandidate(
                    county=county,
                    confidence=0.70,
                    source=GeoSource.TEXT,
                    evidence=f"text:{pattern}"
                ))
        
        # Try secondary matches (also require word boundary)
        for pattern, county in self._secondary_patterns.items():
            if word_match(pattern, text_lower):
                # Lower confidence for secondary text matches
                results.append(GeoCandidate(
                    county=county,
                    confidence=0.55,
                    source=GeoSource.TEXT,
                    evidence=f"text:{pattern}"
                ))
        
        return results
    
    def _choose_best(self, candidates: List[GeoCandidate]) -> Optional[GeoCandidate]:
        """Choose best candidate applying priority rules."""
        if not candidates:
            return None
        
        # Group by source
        meta = [c for c in candidates if c.source == GeoSource.METADATA]
        hashes = [c for c in candidates if c.source == GeoSource.HASHTAGS]
        texts = [c for c in candidates if c.source == GeoSource.TEXT]
        
        # Priority 1: Metadata if passes threshold
        if meta and meta[0].confidence >= self.t_meta:
            return meta[0]
        
        # Priority 2: Hashtags if passes threshold
        if hashes and hashes[0].confidence >= self.t_hash:
            # Don't overwrite high-confidence metadata
            if meta and meta[0].confidence > hashes[0].confidence:
                return meta[0]
            return hashes[0]
        
        # Priority 3: Text if passes threshold
        if texts and texts[0].confidence >= self.t_text:
            # Don't overwrite higher-confidence matches
            if meta and meta[0].confidence > texts[0].confidence:
                return meta[0]
            if hashes and hashes[0].confidence > texts[0].confidence:
                return hashes[0]
            return texts[0]
        
        # Return best candidate below threshold (or None)
        return candidates[0] if candidates[0].confidence > 0.3 else None


class KenyaGeoMapper:
    """
    Maps Pulse signals to Kenya counties with multi-source geo-resolution.
    
    Usage:
        mapper = KenyaGeoMapper()
        resolution = mapper.process_signal(
            text="Protests in Kibera today!",
            signal_type="MOBILIZATION",
            user_location="Nairobi, Kenya",
            hashtags=["#Kibera", "#Protests"]
        )
    """
    
    def __init__(self):
        self.resolver = KenyaGeoResolver()
        self.county_data: Dict[str, CountySignal] = {
            county: CountySignal(county=county)
            for county in KENYA_COUNTIES
        }
        self._unknown_signals: List[Tuple[str, GeoResolution]] = []
    
    def process_signal(
        self, 
        text: str,
        signal_type: str,
        magnitude: float = 0.1,
        user_location: Optional[str] = None,
        hashtags: Optional[List[str]] = None,
    ) -> GeoResolution:
        """
        Process a signal with multi-source geo-resolution.
        
        Args:
            text: Signal text content
            signal_type: Type of signal detected
            magnitude: Signal magnitude (0-1)
            user_location: User profile location metadata
            hashtags: List of hashtags from post
            
        Returns:
            GeoResolution with full audit trail
        """
        # Resolve location
        resolution = self.resolver.resolve(
            text=text,
            user_location=user_location,
            hashtags=hashtags,
        )
        
        # Add to county data or unknown
        if resolution.geo_county and resolution.geo_county in self.county_data:
            self.county_data[resolution.geo_county].add_signal(
                signal_type, magnitude, resolution
            )
        else:
            self._unknown_signals.append((signal_type, resolution))
        
        return resolution
    
    def get_county_data(self) -> Dict[str, Dict]:
        """Get all county data for map rendering."""
        result = {}
        
        for county, signal in self.county_data.items():
            lat, lon = KENYA_COUNTIES[county]
            result[county] = {
                "lat": lat,
                "lon": lon,
                "risk_score": signal.risk_score,
                "signal_count": signal.signal_count,
                "signals": signal.signals[-10:],
                "risk_level": self._get_risk_level(signal.risk_score),
                "resolution_audit": [r.to_dict() for r in signal.resolutions[-5:]],
            }
        
        return result
    
    def get_high_risk_counties(self, threshold: float = 0.5) -> List[str]:
        """Get counties with risk above threshold."""
        return [
            county for county, signal in self.county_data.items()
            if signal.risk_score >= threshold
        ]
    
    def get_unknown_signals(self) -> List[Dict]:
        """Get signals that couldn't be geo-resolved."""
        return [
            {"signal": s, "resolution": r.to_dict()}
            for s, r in self._unknown_signals
        ]
    
    def _get_risk_level(self, score: float) -> str:
        """Convert numeric score to risk level."""
        if score < 0.3:
            return "Low"
        elif score < 0.5:
            return "Moderate"
        elif score < 0.7:
            return "High"
        else:
            return "Critical"
    
    def reset(self):
        """Reset all county data."""
        for county in self.county_data:
            self.county_data[county] = CountySignal(county=county)
        self._unknown_signals = []
    
    def inject_sample_data(self):
        """Inject sample data with realistic geo-resolution."""
        samples = [
            # (text, signal_type, magnitude, user_location, hashtags)
            ("Protests happening now!", "MOBILIZATION", 0.6, "Nairobi, Kenya", ["#UhuruPark", "#Protests"]),
            ("Fuel prices killing us in Mombasa", "ECONOMIC_DISTRESS", 0.4, "Mombasa", ["#MombasaCounty"]),
            ("Violence reported in Kibera slums", "SECURITY_ALERT", 0.7, None, ["#Kibera"]),
            ("Food shortage in Turkana very serious", "SCARCITY_SIGNAL", 0.6, "Lodwar, Turkana", []),
            ("Political tension rising in Kisumu", "ETHNIC_TENSION", 0.5, "Kisumu, Kenya", ["#Kisumu"]),
            ("Dadaab camp facing water crisis", "RESOURCE_CONFLICT", 0.7, None, ["#Dadaab", "#Garissa"]),
            ("Riots in Eldoret town center", "MOBILIZATION", 0.8, "Eldoret", ["#Eldoret"]),
            ("Cattle rustling incidents in Baringo", "SECURITY_ALERT", 0.6, "Kabarnet, Kenya", []),
            ("Drought affecting Marsabit pastoralists", "SCARCITY_SIGNAL", 0.5, "Marsabit County", []),
            ("Thika road blocked by protesters", "MOBILIZATION", 0.4, None, ["#Thika", "#Kiambu"]),
        ]
        
        for text, signal_type, magnitude, user_loc, tags in samples:
            self.process_signal(
                text=text,
                signal_type=signal_type,
                magnitude=magnitude,
                user_location=user_loc,
                hashtags=tags,
            )
