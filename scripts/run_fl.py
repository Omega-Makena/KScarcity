#!/usr/bin/env python
"""
Federated Learning CLI — start coordinator or client nodes.

Usage
-----
::

    # Start the coordinator (aggregation server)
    python scripts/run_fl.py coordinator --port 8765 --model hypothesis_ensemble

    # Start a client node that connects to coordinator
    python scripts/run_fl.py client --coordinator ws://10.0.0.1:8765 --node-id nairobi

    # Run a single sync round manually
    python scripts/run_fl.py round --model logistic

    # Watch a directory for new data and auto-trigger training
    python scripts/run_fl.py watch --dir data/uploads/ --node-id nairobi
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("fl_cli")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SCARCITY Federated Learning CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", help="Sub-command")

    # --- coordinator ---
    coord = sub.add_parser("coordinator", help="Start FL coordinator server")
    coord.add_argument("--host", default="0.0.0.0")
    coord.add_argument("--port", type=int, default=8765)
    coord.add_argument("--model", default="logistic", help="Model name from registry")
    coord.add_argument("--min-nodes", type=int, default=2)
    coord.add_argument("--max-wait", type=float, default=300.0)
    coord.add_argument("--aggregation", default="trimmed_mean")
    coord.add_argument("--auth-token", default=None)

    # --- client ---
    client = sub.add_parser("client", help="Start FL client node")
    client.add_argument("--coordinator", required=True, help="ws://host:port")
    client.add_argument("--node-id", required=True)
    client.add_argument("--port", type=int, default=8766, help="Local WS port")
    client.add_argument("--auth-token", default=None)

    # --- round ---
    rnd = sub.add_parser("round", help="Run a single sync round")
    rnd.add_argument("--model", default="logistic")
    rnd.add_argument("--lr", type=float, default=0.12)
    rnd.add_argument("--source", default="data/synthetic_kenya_policy/tweets.csv")

    # --- watch ---
    watch = sub.add_parser("watch", help="Watch directory for new data files")
    watch.add_argument("--dir", required=True, help="Directory to watch")
    watch.add_argument("--node-id", required=True)
    watch.add_argument("--poll-interval", type=float, default=5.0)
    watch.add_argument("--model", default="logistic")

    # --- status ---
    sub.add_parser("status", help="Show federation status")

    # --- models ---
    sub.add_parser("models", help="List available FL models")

    return parser


async def run_coordinator(args) -> None:
    """Start the FL coordinator with WebSocket transport."""
    from federated_databases.fl_config import FLOrchestratorConfig
    from federated_databases.fl_orchestrator import FLOrchestrator
    from federated_databases.scarcity_federation import get_scarcity_federation
    from scarcity.runtime import get_bus
    from scarcity.federation.ws_transport import WebSocketTransport, WSTransportConfig

    fm = get_scarcity_federation()
    bus = get_bus()

    # Ensure at least some nodes exist
    nodes = fm.control.list_nodes()
    if not nodes:
        logger.info("No nodes registered, creating default nodes...")
        fm.register_node("org_a")
        fm.register_node("org_b")

    # Start WebSocket transport
    ws_config = WSTransportConfig(
        host=args.host,
        port=args.port,
        auth_token=args.auth_token,
    )
    transport = WebSocketTransport(ws_config)

    async def handle_incoming(topic, payload):
        """Route incoming WebSocket packets to EventBus."""
        await bus.publish(topic, payload)

    transport.register_handler(handle_incoming)
    await transport.start()

    # Start orchestrator
    config = FLOrchestratorConfig(
        model_name=args.model,
        min_nodes_per_round=args.min_nodes,
        max_wait_seconds=args.max_wait,
        aggregation_method=args.aggregation,
        ws_host=args.host,
        ws_port=args.port,
        auth_token=args.auth_token,
    )
    orchestrator = FLOrchestrator(fm, bus, config)
    await orchestrator.start()

    # Subscribe to global_updated to broadcast via WebSocket
    async def broadcast_global(topic, payload):
        await transport.send("fl.global_updated", payload)

    bus.subscribe("fl.global_updated", broadcast_global)

    logger.info(
        f"\n{'='*60}\n"
        f"  FL Coordinator running on ws://{args.host}:{args.port}\n"
        f"  Model: {args.model}\n"
        f"  Min nodes per round: {args.min_nodes}\n"
        f"  Aggregation: {args.aggregation}\n"
        f"{'='*60}\n"
        f"  Waiting for client nodes to connect...\n"
    )

    # Keep running
    try:
        while True:
            await asyncio.sleep(10)
            stats = orchestrator.stats
            peers = transport.connected_peers + transport.connected_clients
            logger.info(
                f"[heartbeat] peers={peers}, "
                f"rounds={stats['rounds_completed']}, "
                f"updates={stats['total_updates_received']}, "
                f"loss={stats.get('last_global_loss', 'N/A')}"
            )
    except asyncio.CancelledError:
        pass
    finally:
        await orchestrator.stop()
        await transport.stop()


async def run_client(args) -> None:
    """Start an FL client node that connects to a coordinator."""
    from federated_databases.scarcity_federation import get_scarcity_federation
    from scarcity.runtime import get_bus
    from scarcity.federation.ws_transport import WebSocketTransport, WSTransportConfig

    fm = get_scarcity_federation()
    bus = get_bus()

    # Ensure node is registered
    try:
        fm.register_node(args.node_id)
    except Exception:
        pass

    # Connect to coordinator
    ws_config = WSTransportConfig(
        host="0.0.0.0",
        port=args.port,
        peer_endpoints=[args.coordinator],
        auth_token=args.auth_token,
    )
    transport = WebSocketTransport(ws_config)

    async def handle_incoming(topic, payload):
        await bus.publish(topic, payload)

    transport.register_handler(handle_incoming)
    await transport.start()

    # Listen for global updates
    async def on_global_update(topic, payload):
        logger.info(
            f"Received global model update: round={payload.get('round_number')}, "
            f"loss={payload.get('global_loss', 'N/A')}"
        )

    bus.subscribe("fl.global_updated", on_global_update)

    logger.info(
        f"\n{'='*60}\n"
        f"  FL Client '{args.node_id}' connected to {args.coordinator}\n"
        f"  Local port: {args.port}\n"
        f"{'='*60}\n"
    )

    try:
        while True:
            await asyncio.sleep(30)
    except asyncio.CancelledError:
        pass
    finally:
        await transport.stop()


def run_round(args) -> None:
    """Run a single sync round."""
    from federated_databases.scarcity_federation import get_scarcity_federation

    fm = get_scarcity_federation()
    nodes = fm.control.list_nodes()
    if not nodes:
        fm.register_node("org_a")
        fm.register_node("org_b")

    logger.info(f"Running sync round with model='{args.model}' ...")

    try:
        fm.ingest_live_batch(source_path=args.source)
    except Exception as e:
        logger.warning(f"Ingest failed (may not have data yet): {e}")

    try:
        result = fm.run_sync_round(
            learning_rate=args.lr,
            model_name=args.model,
            source_path=args.source,
        )
        logger.info(
            f"\nRound {result.round_number} complete:\n"
            f"  Participants: {result.participants}\n"
            f"  Total samples: {result.total_samples}\n"
            f"  Global loss: {result.global_loss:.4f}\n"
            f"  Gradient norm: {result.global_gradient_norm:.4f}\n"
        )
    except Exception as e:
        logger.error(f"Sync round failed: {e}")


async def run_watch(args) -> None:
    """Watch a directory for new CSV files and trigger training."""
    from scarcity.runtime import get_bus

    bus = get_bus()
    watch_dir = Path(args.dir)
    watch_dir.mkdir(parents=True, exist_ok=True)

    seen_files: set = set()
    logger.info(f"Watching {watch_dir} for new data files...")

    while True:
        for path in watch_dir.glob("*.csv"):
            if path.name not in seen_files:
                seen_files.add(path.name)
                logger.info(f"New file detected: {path.name}")
                await bus.publish(
                    "fl.data_ready",
                    {
                        "node_id": args.node_id,
                        "source_path": str(path),
                    },
                )
        await asyncio.sleep(args.poll_interval)


def run_status() -> None:
    """Show federation status."""
    from federated_databases.scarcity_federation import get_scarcity_federation

    fm = get_scarcity_federation()
    status = fm.get_status()

    print(f"\n{'='*50}")
    print(f"  Federation Status")
    print(f"{'='*50}")
    print(f"  Nodes: {status['node_count']}")
    for node in status.get("nodes", []):
        print(f"    - {node['node_id']} (samples: {node.get('sample_count', '?')})")
    latest = status.get("latest_round")
    if latest:
        print(f"  Latest round: #{latest.get('round_number')}")
        print(f"    Loss: {latest.get('global_loss', 'N/A')}")
        print(f"    Participants: {latest.get('participants', 'N/A')}")
    print(f"  Total rounds: {status.get('round_count', 0)}")
    print()


def run_models() -> None:
    """List available FL models."""
    from federated_databases.model_registry import FLModelRegistry

    models = FLModelRegistry.list_models()
    print(f"\n{'='*50}")
    print(f"  Available FL Models ({len(models)})")
    print(f"{'='*50}")
    for m in models:
        print(f"  - {m}")
    print()


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "coordinator":
        asyncio.run(run_coordinator(args))
    elif args.command == "client":
        asyncio.run(run_client(args))
    elif args.command == "round":
        run_round(args)
    elif args.command == "watch":
        asyncio.run(run_watch(args))
    elif args.command == "status":
        run_status()
    elif args.command == "models":
        run_models()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
