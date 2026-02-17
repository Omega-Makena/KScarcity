"""DOT graph utilities for Scarcity causal pipeline."""
from dataclasses import dataclass, field
import re
from typing import Dict, List, Optional, Sequence, Tuple

_EDGE_RE = re.compile(r'("[^"]+"|[A-Za-z0-9_:@+\-\.]+)\s*->\s*("[^"]+"|[A-Za-z0-9_:@+\-\.]+)')
_TIME_PATTERNS = [
    re.compile(r"@t([+-]?\d+)?$", re.IGNORECASE),
    re.compile(r"_t([+-]?\d+)?$", re.IGNORECASE),
]


@dataclass
class DotGraph:
    raw: str
    edges: List[Tuple[str, str]] = field(default_factory=list)


def load_dot(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def parse_dot_edges(dot_text: str) -> List[Tuple[str, str]]:
    edges: List[Tuple[str, str]] = []
    if not dot_text:
        return edges
    for match in _EDGE_RE.finditer(dot_text):
        src = match.group(1).strip("\"")
        tgt = match.group(2).strip("\"")
        edges.append((src, tgt))
    return edges


def extract_time_index(node: str) -> Optional[int]:
    for pattern in _TIME_PATTERNS:
        match = pattern.search(node)
        if match:
            value = match.group(1)
            return int(value) if value not in (None, "") else 0
    return None


def validate_temporal_edges(edges: Sequence[Tuple[str, str]]) -> List[Dict[str, int]]:
    violations: List[Dict[str, int]] = []
    for src, tgt in edges:
        src_t = extract_time_index(src)
        tgt_t = extract_time_index(tgt)
        if src_t is None or tgt_t is None:
            continue
        if src_t > tgt_t:
            violations.append({"source": src, "target": tgt, "source_t": src_t, "target_t": tgt_t})
    return violations


def build_dot(edges: Sequence[Tuple[str, str]], name: str = "LearnedGraph") -> str:
    lines = [f"digraph {name} {{", "  rankdir=LR;", "}"]
    if edges:
        lines = [f"digraph {name} {{", "  rankdir=LR;"]
        for src, tgt in edges:
            lines.append(f"  \"{src}\" -> \"{tgt}\";")
        lines.append("}")
    return "\n".join(lines)


def merge_edges(edge_lists: Sequence[Sequence[Tuple[str, str]]]) -> List[Tuple[str, str]]:
    merged: List[Tuple[str, str]] = []
    seen = set()
    for edges in edge_lists:
        for edge in edges:
            if edge not in seen:
                merged.append(edge)
                seen.add(edge)
    return merged
