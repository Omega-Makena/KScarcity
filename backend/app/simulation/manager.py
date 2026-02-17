"""In-memory simulation engine for generating demo data."""

from __future__ import annotations

import asyncio
import itertools
import json
import random
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator, Literal

from app.core.datasets import DatasetRegistry
from app.schemas.controls import DriftRequest, PauseRequest
from app.schemas.data import DataEntry, DataEntryRecord
from app.schemas.domains import ClientSummary, DomainCompliance, DomainListResponse, DomainSummary
from app.schemas.metrics import KpiMetric, SummaryMetricsResponse
from app.schemas.risk import AuditEvent, RiskCard, RiskStatusResponse

StakeholderMode = Literal["stakeholder", "client"]


class SimulationManager:
    """Generates mock data for the dashboard."""

    def __init__(self, seed: int = 42, tick_seconds: float = 1.0, dataset_registry: DatasetRegistry | None = None) -> None:
        self._random = random.Random(seed)
        self._tick_seconds = tick_seconds
        self._global_paused = False
        self._paused_domains: set[str] = set()
        self._paused_clients: set[str] = set()

        self._kpi_state: dict[StakeholderMode, list[KpiMetric]] = {
            "stakeholder": [
                KpiMetric(id="accuracy_lift", label="Accuracy Lift", value=18.5, unit="%", delta=0.4, trend="up"),
                KpiMetric(id="latency", label="Inference Latency", value=145.0, unit="ms", delta=-5.0, trend="up"),
                KpiMetric(id="data_retained", label="Data Retained On-Device", value=99.1, unit="%", delta=0.1, trend="up"),
                KpiMetric(id="meta_boost", label="Meta-Learning Boost", value=7.4, unit="%", delta=0.2, trend="up"),
            ],
            "client": [
                KpiMetric(id="accuracy_lift", label="Accuracy Lift", value=12.7, unit="%", delta=0.2, trend="up"),
                KpiMetric(id="latency", label="Inference Latency", value=162.0, unit="ms", delta=-3.0, trend="up"),
                KpiMetric(id="data_retained", label="Data Retained On-Device", value=98.4, unit="%", delta=0.1, trend="up"),
                KpiMetric(id="meta_boost", label="Meta-Learning Boost", value=5.9, unit="%", delta=0.1, trend="up"),
            ],
        }

        self._domain_metrics: dict[str, dict[str, float | str | list[str] | None]] = {}
        self._phase_cycle: tuple[str, ...] = ("Onboarding", "Aggregating", "Training", "Validating")

        self._audit_events: deque[AuditEvent] = deque(maxlen=100)
        self._audit_events.extend(
            [
                AuditEvent(
                    event_id="evt-001",
                    domain_id=None,
                    description="Secure aggregation round completed.",
                    level="info",
                ),
                AuditEvent(
                    event_id="evt-002",
                    domain_id=None,
                    description="Differential privacy budget regenerated.",
                    level="info",
                ),
            ]
        )
        self._timeline_step = 0
        self._meta_scarcity = 0.6
        self._ingest_log: deque[DataEntryRecord] = deque(maxlen=200)
        self._dataset_registry = dataset_registry
        self._dataset_domains: list[DomainSummary] = []
        self._dataset_overlay_map: dict[str, DomainSummary] = {}
        self.refresh_dataset_overlay()

    # Dataset overlay ----------------------------------------------------- #

    def set_dataset_registry(self, registry: DatasetRegistry | None) -> None:
        """Attach or replace the dataset registry."""
        self._dataset_registry = registry
        self.refresh_dataset_overlay()

    def refresh_dataset_overlay(self) -> None:
        """Regenerate dataset-driven domain summaries."""
        self._dataset_domains = []
        self._dataset_overlay_map = {}
        if not self._dataset_registry:
            return
        refreshed_at = self._dataset_registry.totals().get("refreshed_at")
        overlays: list[DomainSummary] = []
        for entry in self._dataset_registry.domain_breakdown():
            sector = self._infer_sector(entry.get("domain", "Data Hub"))
            accuracy = round(82.0 + min(entry.get("windows", 0) / 25.0, 8.0), 1)
            phase = "Streaming"
            summary = DomainSummary(
                domain_id=f"dataset-{entry.get('domain_id')}",
                name=str(entry.get("domain", "Dataset Domain")),
                sector=sector,
                phase=phase,
                accuracy=accuracy,
                delta=self._random.uniform(-0.2, 0.4),
                clients=[],
                compliance=DomainCompliance(
                    frameworks=self._default_frameworks(sector),
                    notes=f"{entry.get('datasets', 0)} datasets in this hub.",
                ),
                origin="dataset",
                dataset_count=int(entry.get("datasets", 0)),
                dataset_rows=int(entry.get("rows", 0)),
                dataset_windows=int(entry.get("windows", 0)),
                dataset_last_ingested=refreshed_at,
            )
            overlays.append(summary)
            self._dataset_overlay_map[summary.name.lower()] = summary
        self._dataset_domains = overlays

    def _dataset_overlay_for_name(self, name: str) -> DomainSummary | None:
        if not name:
            return None
        key = name.lower()
        if key in self._dataset_overlay_map:
            return self._dataset_overlay_map[key]
        # fallback fuzzy: try contains
        for overlay_key, summary in self._dataset_overlay_map.items():
            if overlay_key in key or key in overlay_key:
                return summary
        return None

    # --------------------------------------------------------------------- #
    # Utility mutators

    def _infer_sector(self, name: str) -> str:
        """Infer a sector label from a domain name."""

        lowered = name.lower()
        sector_map = {
            "finance": "Finance",
            "fintech": "Finance",
            "econom": "Economics",
            "health": "Healthcare",
            "med": "Healthcare",
            "pharma": "Biopharma",
            "tech": "Technology",
            "ed": "Education",
            "learn": "Education",
            "energy": "Energy",
            "climate": "Climate",
            "civic": "Civic",
            "agri": "Agriculture",
        }
        for key, sector in sector_map.items():
            if key in lowered:
                return sector
        return "General"

    def _default_frameworks(self, sector: str) -> list[str]:
        """Return default compliance frameworks for a sector."""

        defaults = {
            "Finance": ["SOC2", "PCI DSS"],
            "Healthcare": ["HIPAA", "ISO 27001"],
            "Biopharma": ["GxP"],
            "Technology": ["ISO 27017"],
            "Education": ["FERPA"],
            "Energy": ["ISO 50001"],
            "Economics": ["OECD Fair Data"],
            "Climate": ["TCFD"],
            "Civic": ["NIST Privacy"],
            "Agriculture": ["Global G.A.P."],
        }
        return defaults.get(sector, [])

    def _sync_domain_metrics(self, domain_objs: list) -> None:
        """Ensure internal metric cache aligns with onboarding domains."""

        active_ids = set()
        for domain in domain_objs:
            active_ids.add(domain.id)
            sector = self._infer_sector(domain.name)
            if domain.id not in self._domain_metrics:
                self._domain_metrics[domain.id] = {
                    "name": domain.name,
                    "sector": sector,
                    "phase": self._random.choice(self._phase_cycle),
                    "accuracy": round(self._random.uniform(82.0, 95.0), 1),
                    "delta": 0.0,
                    "frameworks": self._default_frameworks(sector),
                    "notes": None,
                }
            else:
                # Keep name and sector aligned with latest metadata.
                metrics = self._domain_metrics[domain.id]
                metrics["name"] = domain.name
                metrics["sector"] = sector
        # Drop cached entries for domains that no longer exist.
        for domain_id in list(self._domain_metrics.keys()):
            if domain_id not in active_ids:
                del self._domain_metrics[domain_id]
                self._paused_domains.discard(domain_id)

    def _update_domain_metrics(self, domain_objs: list) -> None:
        """Apply drift to cached metrics for active domains."""

        if self._global_paused:
            return
        for domain in domain_objs:
            metrics = self._domain_metrics.get(domain.id)
            if metrics is None or domain.id in self._paused_domains:
                continue
            jitter = self._random.uniform(-0.4, 0.6)
            current_accuracy = float(metrics["accuracy"])
            metrics["accuracy"] = max(70.0, min(99.9, current_accuracy + jitter))
            metrics["delta"] = jitter
            if self._random.random() > 0.94:
                metrics["phase"] = self._random.choice(self._phase_cycle)

    def _build_domain_summary(self, domain) -> DomainSummary:
        """Construct a DomainSummary from cached metrics and onboarding clients."""

        metrics = self._domain_metrics[domain.id]
        from scarcity.dashboard.onboarding import clients as onboarding_clients

        client_objs = onboarding_clients.list_clients(domain_id=domain.id)

        from scarcity.dashboard.onboarding.state import ClientState

        def _client_status(client) -> str:
            if client.id in self._paused_clients:
                return "paused"
            if client.state == ClientState.ACTIVE:
                return "active"
            if client.state == ClientState.SYNCING:
                return "syncing"
            if client.state == ClientState.REGISTERED:
                return "registering"
            if client.state == ClientState.QUARANTINED:
                return "quarantined"
            return client.state.value.lower()

        client_summaries = [
            ClientSummary(
                client_id=client.id,
                region=client.profile_class or client.domain_id.upper(),
                status=_client_status(client),
            )
            for client in client_objs
        ]
        compliance = DomainCompliance(
            frameworks=list(metrics.get("frameworks", [])),
            notes=metrics.get("notes"),
        )
        dataset_overlay = self._dataset_overlay_for_name(domain.name)
        dataset_count = dataset_overlay.dataset_count if dataset_overlay else 0
        dataset_rows = dataset_overlay.dataset_rows if dataset_overlay else 0
        dataset_windows = dataset_overlay.dataset_windows if dataset_overlay else 0
        dataset_last_ingested = dataset_overlay.dataset_last_ingested if dataset_overlay else None
        return DomainSummary(
            domain_id=domain.id,
            name=metrics.get("name", domain.name),
            sector=str(metrics.get("sector", "General")),
            phase=str(metrics.get("phase", "Onboarding")),
            accuracy=float(metrics.get("accuracy", 85.0)),
            delta=float(metrics.get("delta", 0.0)),
            clients=client_summaries,
            compliance=compliance,
            origin="scic",
            dataset_count=dataset_count,
            dataset_rows=dataset_rows,
            dataset_windows=dataset_windows,
            dataset_last_ingested=dataset_last_ingested,
        )

    def _step_kpis(self, mode: StakeholderMode) -> None:
        volatility = 0.6 if mode == "stakeholder" else 0.4
        for metric in self._kpi_state[mode]:
            noise = self._random.uniform(-volatility, volatility)
            metric.value = max(0.0, metric.value + noise)
            metric.delta = noise
            if noise > 0.05:
                metric.trend = "up"
            elif noise < -0.05:
                metric.trend = "down"
            else:
                metric.trend = "flat"

    def _step_domains(self, domain_objs: list) -> None:
        """Advance domain metrics."""

        self._update_domain_metrics(domain_objs)

    def _push_audit_event(self, description: str, domain_id: str | None = None, level: str = "info") -> None:
        event_id = f"evt-{int(time.time() * 1000)}"
        self._audit_events.appendleft(
            AuditEvent(
                event_id=event_id,
                domain_id=domain_id,
                description=description,
                level=level,
            )
        )

    # --------------------------------------------------------------------- #
    # Public APIs for REST handlers

    def get_summary(self, mode: StakeholderMode = "stakeholder") -> SummaryMetricsResponse:
        """Return updated KPI metrics for the requested mode."""

        if not self._global_paused:
            self._step_kpis(mode)

        return SummaryMetricsResponse(mode=mode, metrics=list(self._kpi_state[mode]))

    def get_domains(self, domain_id: str | None = None) -> DomainListResponse:
        """Return domain summary list."""

        from scarcity.dashboard.onboarding import domains as onboarding_domains

        domain_objs = onboarding_domains.list_domains()
        if not domain_objs:
            seeds = self._load_seed_domains()
            for seed in seeds:
                try:
                    onboarding_domains.create_domain(seed["name"], seed["description"])
                except ValueError:
                    continue
            domain_objs = onboarding_domains.list_domains()
        self._sync_domain_metrics(domain_objs)
        self._step_domains(domain_objs)
        summaries = [self._build_domain_summary(domain) for domain in domain_objs]
        summaries.extend(self._dataset_domains)
        selected = None
        if domain_id:
            for summary in summaries:
                if summary.domain_id == domain_id:
                    selected = summary
                    break
        return DomainListResponse(domains=summaries, selected_domain=selected)

    def _load_seed_domains(self) -> list[dict[str, str]]:
        """Return default domain definitions sourced from the onboarding config."""

        from scarcity.dashboard.onboarding import domains as onboarding_domains

        config_path = getattr(onboarding_domains, "CONFIG_PATH", None)
        if config_path:
            config_path = Path(config_path)
            if config_path.exists():
                try:
                    payload = json.loads(config_path.read_text(encoding="utf-8"))
                    seeds = []
                    for entry in payload:
                        if isinstance(entry, dict) and entry.get("name"):
                            seeds.append(
                                {
                                    "name": str(entry["name"]).strip(),
                                    "description": str(entry.get("description", "")).strip(),
                                }
                            )
                    return seeds
                except json.JSONDecodeError:
                    return []
        return []

    def get_risk_status(self) -> RiskStatusResponse:
        """Return current risk cards and audit events."""

        cards = [
            RiskCard(
                id="privacy",
                title="Privacy & Differential Privacy",
                status="green" if not self._global_paused else "amber",
                summary="ε=3.1 remaining budget; noise layer active.",
                badges=["DP", "Secure Aggregation"],
            ),
            RiskCard(
                id="aggregation",
                title="Secure Aggregation",
                status="green",
                summary="All recent rounds encrypted with AES-GCM.",
                badges=["KMS", "Zero Trust"],
            ),
            RiskCard(
                id="compliance",
                title="Compliance & Audit",
                status="green",
                summary="All domains within SLA; next audit due in 12 days.",
                badges=["GDPR", "HIPAA"],
            ),
        ]
        return RiskStatusResponse(cards=cards, audit_events=list(self._audit_events))

    def inject_drift(self, payload: DriftRequest) -> dict[str, str]:
        """Simulate concept drift injection."""

        target = payload.domain_id or "global"
        intensity = payload.intensity
        impact = intensity * self._random.uniform(1.0, 3.0)
        if payload.domain_id and payload.domain_id not in self._domain_metrics:
            from scarcity.dashboard.onboarding import domains as onboarding_domains

            domain_obj = onboarding_domains.get_domain(payload.domain_id)
            if domain_obj:
                self._sync_domain_metrics([domain_obj])
        if payload.domain_id and payload.domain_id in self._domain_metrics:
            metrics = self._domain_metrics[payload.domain_id]
            current_accuracy = float(metrics.get("accuracy", 85.0))
            metrics["accuracy"] = max(60.0, current_accuracy - impact)
            metrics["delta"] = -impact
            metrics["phase"] = "Investigating Drift"
            self._push_audit_event(
                description=f"Artificial drift injected at intensity {intensity:.2f}.",
                domain_id=payload.domain_id,
                level="warning",
            )
        else:
            self._push_audit_event(
                description=f"Global drift scenario triggered (intensity {intensity:.2f}).",
                level="warning",
            )
        return {"message": f"Drift injected for {target}", "intensity": f"{intensity:.2f}"}

    def apply_pause(self, payload: PauseRequest) -> dict[str, str]:
        """Pause or resume simulation scopes."""

        if payload.scope == "global":
            self._global_paused = payload.action == "pause"
        elif payload.scope == "domain" and payload.target_id:
            if payload.action == "pause":
                self._paused_domains.add(payload.target_id)
            else:
                self._paused_domains.discard(payload.target_id)
        elif payload.scope == "client" and payload.target_id:
            if payload.action == "pause":
                self._paused_clients.add(payload.target_id)
            else:
                self._paused_clients.discard(payload.target_id)
        else:
            return {"message": "Invalid pause request."}

        self._push_audit_event(
            description=f"{payload.scope.title()} {payload.action} requested.",
            domain_id=payload.target_id if payload.scope != "global" else None,
            level="info",
        )
        return {"message": f"{payload.scope.title()} {payload.action} processed."}

    def get_audit_for_domain(self, domain_id: str) -> list[AuditEvent]:
        """Filter audit events for a domain."""

        return [event for event in self._audit_events if event.domain_id == domain_id]

    def ingest_data(self, payload: DataEntry) -> dict[str, str]:
        """Ingest external data and nudge simulation state."""

        timestamp = datetime.utcnow().isoformat() + "Z"
        impact = payload.derived_delta()
        record = DataEntryRecord(
            domain_id=payload.domain_id,
            source=payload.source,
            metrics=payload.metrics,
            narrative=payload.narrative,
            timestamp=timestamp,
            impact=impact,
        )
        self._ingest_log.appendleft(record)

        if payload.domain_id and payload.domain_id not in self._domain_metrics:
            from scarcity.dashboard.onboarding import domains as onboarding_domains

            domain_obj = onboarding_domains.get_domain(payload.domain_id)
            if domain_obj:
                self._sync_domain_metrics([domain_obj])
        metrics = self._domain_metrics.get(payload.domain_id)
        if metrics:
            current_accuracy = float(metrics.get("accuracy", 85.0))
            metrics["accuracy"] = max(60.0, min(99.9, current_accuracy + impact))
            metrics["delta"] = impact
            self._push_audit_event(
                description=f"Data ingest adjusted {payload.domain_id} trajectory ({impact:+.2f}).",
                domain_id=payload.domain_id,
                level="info",
            )
        else:
            self._push_audit_event(
                description=f"Data ingest received for unknown domain {payload.domain_id}.",
                level="warning",
            )

        return {
            "message": "Data entry processed.",
            "domain": payload.domain_id,
            "impact": f"{impact:+.2f}",
            "timestamp": timestamp,
        }

    def recent_ingest(self, limit: int = 20) -> list[DataEntryRecord]:
        """Return recent ingested data entries."""

        return list(itertools.islice(self._ingest_log, 0, limit))

    def run_whatif(
        self,
        scenario_id: str,
        magnitude: float = 0.5,
        domain_id: str | None = None,
        horizon: int = 12,
    ) -> dict[str, object]:
        """Execute a lightweight what-if simulation returning projected metrics."""

        domains = self.get_domains(domain_id=domain_id).domains
        projections: list[dict[str, object]] = []
        for domain in domains:
            baseline = domain.accuracy
            projected = max(60.0, min(99.9, baseline + magnitude))
            trajectory = [
                {
                    "step": step,
                    "accuracy": max(60.0, min(99.9, baseline + magnitude * (step / horizon))),
                }
                for step in range(1, horizon + 1)
            ]
            projections.append(
                {
                    "domain_id": domain.domain_id,
                    "name": domain.name,
                    "baseline_accuracy": baseline,
                    "projected_accuracy": projected,
                    "trajectory": trajectory,
                }
            )
        return {
            "scenario_id": scenario_id,
            "magnitude": magnitude,
            "horizon": horizon,
            "domains": projections,
        }

    # --------------------------------------------------------------------- #
    # Introspection helpers

    @property
    def tick_seconds(self) -> float:
        """Return the current tick interval."""

        return self._tick_seconds

    @property
    def is_paused(self) -> bool:
        """Return whether the global simulation is paused."""

        return self._global_paused

    # --------------------------------------------------------------------- #
    # Websocket stream producers

    async def metrics_stream(self, mode: StakeholderMode = "stakeholder") -> AsyncIterator[dict]:
        """Yield KPI updates on an interval."""

        for step in itertools.count():
            if self._global_paused:
                await asyncio.sleep(self._tick_seconds)
                continue
            summary = self.get_summary(mode)
            payload = summary.model_dump()
            payload["sequence"] = step
            yield payload
            await asyncio.sleep(self._tick_seconds)

    async def gossip_stream(self) -> AsyncIterator[dict]:
        """Yield gossip events between clients."""

        from scarcity.dashboard.onboarding import clients as onboarding_clients

        for sequence in itertools.count():
            if self._global_paused:
                await asyncio.sleep(self._tick_seconds)
                continue
            client_objs = onboarding_clients.list_clients()
            active_clients = [client for client in client_objs if client.id not in self._paused_clients]
            if len(active_clients) < 2:
                await asyncio.sleep(self._tick_seconds)
                continue
            domain_groups: dict[str, list[str]] = {}
            for client in active_clients:
                domain_groups.setdefault(client.domain_id, []).append(client.id)
            same_domain_candidates = [ids for ids in domain_groups.values() if len(ids) >= 2]
            if same_domain_candidates:
                domain_clients = self._random.choice(same_domain_candidates)
                pair_ids = self._random.sample(domain_clients, 2)
            else:
                pair_ids = self._random.sample([client.id for client in active_clients], 2)
            accuracy_delta = self._random.uniform(-0.5, 1.5)
            payload = {
                "sequence": sequence,
                "from_client": pair_ids[0],
                "to_client": pair_ids[1],
                "accuracy_delta": round(accuracy_delta, 2),
                "timestamp": time.time(),
            }
            yield payload
            await asyncio.sleep(self._tick_seconds)

    async def timeline_stream(self) -> AsyncIterator[dict]:
        """Yield online learning timeline data."""

        for sequence in itertools.count():
            if self._global_paused:
                await asyncio.sleep(self._tick_seconds)
                continue
            self._timeline_step += 1
            loss = max(0.05, 1.6 * (0.98 ** self._timeline_step) + self._random.uniform(-0.02, 0.02))
            accuracy = min(0.99, 0.72 + self._timeline_step * 0.002 + self._random.uniform(-0.01, 0.01))
            payload = {
                "sequence": sequence,
                "step": self._timeline_step,
                "loss": round(loss, 4),
                "accuracy": round(accuracy, 4),
                "batch_event": self._timeline_step % 5 == 0,
                "drift": self._timeline_step % 37 == 0,
            }
            yield payload
            await asyncio.sleep(self._tick_seconds)

    async def meta_stream(self) -> AsyncIterator[dict]:
        """Yield meta-learning scenario data based on scarcity slider."""

        for sequence in itertools.count():
            if self._global_paused:
                await asyncio.sleep(self._tick_seconds)
                continue
            jitter = self._random.uniform(-0.05, 0.05)
            self._meta_scarcity = min(1.0, max(0.1, self._meta_scarcity + jitter))
            curve = [
                {"scarcity": round(level, 2), "meta_accuracy": round(0.65 + level * 0.3, 3)}
                for level in [0.1, 0.3, 0.5, 0.7, self._meta_scarcity, 1.0]
            ]
            payload = {
                "sequence": sequence,
                "scarcity_level": round(self._meta_scarcity, 2),
                "curve": curve,
            }
            yield payload
            await asyncio.sleep(self._tick_seconds)


__all__ = ["SimulationManager"]

