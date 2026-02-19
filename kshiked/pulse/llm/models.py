"""
KShield Pulse Model Manager

Manages Ollama model lifecycle:
- Check required models are available
- Pull missing models automatically
- Verify model capabilities
- Report GPU/VRAM status
- CLI interface for quick setup

Usage:
    # Programmatic
    from kshiked.pulse.llm.models import ModelManager
    
    manager = ModelManager()
    await manager.setup()  # Pull all required models
    status = await manager.status()  # Check health
    
    # CLI (from project root)
    python -m kshiked.pulse.llm.models --setup
    python -m kshiked.pulse.llm.models --status
"""

import asyncio
import json
import logging
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import aiohttp

from .config import (
    OllamaConfig,
    ModelProfile,
    MODEL_REGISTRY,
    DEFAULT_TASK_ROUTING,
    AnalysisTask,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelStatus:
    """Status of a single model."""
    name: str
    available: bool = False
    size_gb: float = 0.0
    parameter_count: str = ""
    quantization: str = ""
    tasks: List[str] = field(default_factory=list)


@dataclass
class SystemStatus:
    """Overall system health."""
    ollama_running: bool = False
    ollama_version: str = ""
    models: List[ModelStatus] = field(default_factory=list)
    gpu_available: bool = False
    gpu_name: str = ""
    vram_total_gb: float = 0.0
    vram_free_gb: float = 0.0
    ready: bool = False
    missing_models: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "KShield Pulse — Ollama System Status",
            "=" * 60,
            f"Ollama Server:  {'✓ Running' if self.ollama_running else '✗ NOT Running'}",
            f"Ollama Version: {self.ollama_version or 'Unknown'}",
            f"GPU:            {self.gpu_name or 'Not detected'}",
        ]
        if self.gpu_available:
            lines.append(
                f"VRAM:           {self.vram_free_gb:.1f} / {self.vram_total_gb:.1f} GB free"
            )
        
        lines.append("")
        lines.append("Models:")
        lines.append("-" * 50)
        
        for m in self.models:
            status = "✓" if m.available else "✗ MISSING"
            tasks = ", ".join(m.tasks[:3])
            size = f" ({m.size_gb:.1f} GB)" if m.size_gb > 0 else ""
            lines.append(f"  {status} {m.name}{size}  →  {tasks}")
        
        if self.missing_models:
            lines.append("")
            lines.append(f"⚠ Missing models: {', '.join(self.missing_models)}")
            lines.append("  Run: python -m kshiked.pulse.llm.models --setup")
        
        lines.append("")
        lines.append(
            f"Overall: {'✓ READY' if self.ready else '✗ NOT READY — models missing'}"
        )
        lines.append("=" * 60)
        
        return "\n".join(lines)


class ModelManager:
    """
    Manage Ollama models for KShield Pulse.
    
    Handles:
    - Health checking Ollama server
    - Listing available/required models
    - Pulling missing models
    - GPU/VRAM detection (via Ollama)
    - Readiness verification
    """

    def __init__(self, config: Optional[OllamaConfig] = None):
        self.config = config or OllamaConfig()
        self.base_url = self.config.base_url
        self._session: Optional[aiohttp.ClientSession] = None

    async def _ensure_session(self):
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(connect=10.0, total=600.0)
            self._session = aiohttp.ClientSession(timeout=timeout)

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None

    async def __aenter__(self):
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    # =========================================================================
    # Server Health
    # =========================================================================

    async def is_running(self) -> bool:
        """Check if Ollama server is running."""
        await self._ensure_session()
        try:
            async with self._session.get(f"{self.base_url}/api/tags") as resp:
                return resp.status == 200
        except Exception:
            return False

    async def get_version(self) -> str:
        """Get Ollama server version."""
        await self._ensure_session()
        try:
            async with self._session.get(f"{self.base_url}/api/version") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("version", "unknown")
        except Exception:
            pass
        return ""

    # =========================================================================
    # Model Queries
    # =========================================================================

    async def list_local_models(self) -> List[Dict[str, Any]]:
        """List all locally available models with metadata."""
        await self._ensure_session()
        try:
            async with self._session.get(f"{self.base_url}/api/tags") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("models", [])
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
        return []

    async def get_required_models(self) -> List[str]:
        """Get list of models required by current task routing config."""
        models = set()
        for task, model in self.config.task_routing.items():
            models.add(model)
        # Add embedding model
        models.add(self.config.embedding_model)
        return sorted(models)

    async def check_model_available(self, model_name: str) -> bool:
        """Check if a specific model is available locally."""
        models = await self.list_local_models()
        local_names = [m.get("name", "") for m in models]
        # Partial match (e.g. "llama3.1:8b" matches "llama3.1:8b-instruct-q4_0")
        return any(
            model_name in name or name.startswith(model_name.split(":")[0])
            for name in local_names
        )

    # =========================================================================
    # Model Management
    # =========================================================================

    async def pull_model(
        self,
        model_name: str,
        progress_callback: Optional[Any] = None,
    ) -> bool:
        """
        Pull a model from the Ollama registry.
        
        This can take several minutes for large models.
        """
        await self._ensure_session()
        logger.info(f"Pulling model: {model_name} (this may take a while)...")
        
        try:
            payload = {"name": model_name, "stream": True}
            async with self._session.post(
                f"{self.base_url}/api/pull", json=payload
            ) as resp:
                if resp.status != 200:
                    logger.error(f"Pull failed: HTTP {resp.status}")
                    return False
                
                # Stream progress
                last_status = ""
                async for line in resp.content:
                    try:
                        data = json.loads(line)
                        status = data.get("status", "")
                        if status != last_status:
                            logger.info(f"  {status}")
                            last_status = status
                            if progress_callback:
                                progress_callback(data)
                    except json.JSONDecodeError:
                        continue
                
                logger.info(f"✓ Model {model_name} pulled successfully")
                return True

        except Exception as e:
            logger.error(f"Pull error: {e}")
            return False

    async def setup(self, force: bool = False) -> List[str]:
        """
        Set up all required models.
        
        Checks what's needed and pulls missing models.
        
        Args:
            force: Pull even if model exists
            
        Returns:
            List of models that were pulled
        """
        if not await self.is_running():
            logger.error(
                "Ollama is not running! Start it with: ollama serve"
            )
            return []

        required = await self.get_required_models()
        pulled = []

        for model_name in required:
            available = await self.check_model_available(model_name)
            
            if available and not force:
                logger.info(f"✓ {model_name} already available")
                continue
            
            logger.info(f"Pulling {model_name}...")
            success = await self.pull_model(model_name)
            if success:
                pulled.append(model_name)
            else:
                logger.warning(f"✗ Failed to pull {model_name}")

        return pulled

    # =========================================================================
    # System Status
    # =========================================================================

    async def status(self) -> SystemStatus:
        """Get comprehensive system status."""
        sys_status = SystemStatus()
        
        # Check server
        sys_status.ollama_running = await self.is_running()
        if not sys_status.ollama_running:
            return sys_status
        
        sys_status.ollama_version = await self.get_version()
        
        # Get local models
        local_models = await self.list_local_models()
        local_names = {m.get("name", ""): m for m in local_models}
        
        # Check required models
        required = await self.get_required_models()
        task_map = self._build_task_map()
        
        for model_name in required:
            ms = ModelStatus(name=model_name)
            ms.tasks = task_map.get(model_name, [])
            
            # Check availability
            for local_name, info in local_names.items():
                if model_name in local_name or local_name.startswith(
                    model_name.split(":")[0]
                ):
                    ms.available = True
                    size_bytes = info.get("size", 0)
                    ms.size_gb = round(size_bytes / (1024**3), 1)
                    details = info.get("details", {})
                    ms.parameter_count = details.get("parameter_size", "")
                    ms.quantization = details.get("quantization_level", "")
                    break
            
            sys_status.models.append(ms)
            if not ms.available:
                sys_status.missing_models.append(model_name)

        # GPU detection (via Ollama PS endpoint if available)
        try:
            await self._ensure_session()
            async with self._session.get(f"{self.base_url}/api/ps") as resp:
                if resp.status == 200:
                    ps_data = await resp.json()
                    running = ps_data.get("models", [])
                    if running:
                        # If models are loaded on GPU, GPU is available
                        sys_status.gpu_available = True
        except Exception:
            pass

        # Also check via torch if available
        try:
            import torch
            if torch.cuda.is_available():
                sys_status.gpu_available = True
                sys_status.gpu_name = torch.cuda.get_device_name(0)
                sys_status.vram_total_gb = round(
                    torch.cuda.get_device_properties(0).total_memory / (1024**3), 1
                )
                sys_status.vram_free_gb = round(
                    (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0))
                    / (1024**3), 1
                )
        except ImportError:
            pass

        sys_status.ready = len(sys_status.missing_models) == 0
        return sys_status

    async def verify_model(self, model_name: str) -> Dict[str, Any]:
        """
        Verify a model works by sending a simple test prompt.
        
        Returns:
            {"model": name, "working": bool, "latency_ms": float, "error": str}
        """
        await self._ensure_session()
        import time
        
        start = time.monotonic()
        try:
            payload = {
                "model": model_name,
                "prompt": "Reply with exactly: {\"status\": \"ok\"}",
                "system": "Reply with valid JSON only.",
                "stream": False,
                "format": "json",
                "options": {"temperature": 0, "num_ctx": 256},
            }
            async with self._session.post(
                f"{self.base_url}/api/generate", json=payload
            ) as resp:
                elapsed = (time.monotonic() - start) * 1000
                if resp.status == 200:
                    return {
                        "model": model_name,
                        "working": True,
                        "latency_ms": round(elapsed, 1),
                        "error": "",
                    }
                return {
                    "model": model_name,
                    "working": False,
                    "latency_ms": round(elapsed, 1),
                    "error": f"HTTP {resp.status}",
                }
        except Exception as e:
            elapsed = (time.monotonic() - start) * 1000
            return {
                "model": model_name,
                "working": False,
                "latency_ms": round(elapsed, 1),
                "error": str(e),
            }

    # =========================================================================
    # Helpers
    # =========================================================================

    def _build_task_map(self) -> Dict[str, List[str]]:
        """Build model → tasks mapping."""
        task_map: Dict[str, List[str]] = {}
        for task, model in self.config.task_routing.items():
            task_map.setdefault(model, []).append(task.value)
        # Embedding model
        emb = self.config.embedding_model
        task_map.setdefault(emb, []).append("embedding")
        return task_map


# =============================================================================
# CLI Entry Point
# =============================================================================

async def _cli_main():
    """CLI entry point for model management."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="KShield Pulse — Ollama Model Manager"
    )
    parser.add_argument("--setup", action="store_true", help="Pull all required models")
    parser.add_argument("--status", action="store_true", help="Show system status")
    parser.add_argument("--verify", type=str, help="Verify a specific model")
    parser.add_argument("--pull", type=str, help="Pull a specific model")
    parser.add_argument(
        "--url", type=str, default="http://localhost:11434",
        help="Ollama server URL",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Default model override",
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    
    config = OllamaConfig(base_url=args.url)
    if args.model:
        config = OllamaConfig.single_model(args.model, base_url=args.url)
    
    async with ModelManager(config) as manager:
        if args.setup:
            pulled = await manager.setup()
            if pulled:
                print(f"\nPulled {len(pulled)} models: {', '.join(pulled)}")
            else:
                print("\nAll models already available ✓")
        
        elif args.status:
            status = await manager.status()
            print(status.summary())
        
        elif args.verify:
            result = await manager.verify_model(args.verify)
            if result["working"]:
                print(f"✓ {args.verify} — OK ({result['latency_ms']:.0f}ms)")
            else:
                print(f"✗ {args.verify} — FAILED: {result['error']}")
        
        elif args.pull:
            success = await manager.pull_model(args.pull)
            print(f"{'✓' if success else '✗'} {args.pull}")
        
        else:
            # Default: show status
            status = await manager.status()
            print(status.summary())


if __name__ == "__main__":
    asyncio.run(_cli_main())
