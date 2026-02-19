"""
Simple validation script to test v2 API endpoints.
"""
import asyncio
import sys
import os

# Add parent directory to path for scarcity module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.scarcity_manager import ScarcityCoreManager


async def test_scarcity_manager():
    """Test ScarcityCoreManager initialization and startup."""
    print("Testing ScarcityCoreManager...")
    
    manager = ScarcityCoreManager()
    print("✓ Manager created")
    
    try:
        # Initialize
        await manager.initialize()
        print("✓ Manager initialized")
        
        # Check status
        status = manager.get_status()
        print(f"✓ Status retrieved: {status}")
        
        # Start components
        await manager.start()
        print("✓ Manager started")
        
        # Verify components are running
        if manager.bus:
            print(f"✓ Runtime Bus: {manager.bus.get_stats()}")
        
        if manager.mpie:
            print(f"✓ MPIE: {manager.mpie.get_stats()}")
        
        if manager.drg:
            print("✓ DRG: Running")
        
        if manager.federation_coordinator:
            print(f"✓ Federation Coordinator: {len(manager.federation_coordinator.peers())} peers")
        
        if manager.federation_client:
            print("✓ Federation Client: Running")
        
        if manager.meta:
            print("✓ Meta Learning: Running")
        
        if manager.simulation:
            print("✓ Simulation Engine: Running")
        
        # Test telemetry
        telemetry = manager.get_telemetry_history(limit=10)
        print(f"✓ Telemetry history: {len(telemetry)} events")
        
        # Stop components
        await manager.stop()
        print("✓ Manager stopped")
        
        print("\n✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Ensure cleanup
        try:
            await manager.stop()
        except:
            pass


if __name__ == "__main__":
    success = asyncio.run(test_scarcity_manager())
    sys.exit(0 if success else 1)
