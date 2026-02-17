"""
Test v2 API endpoints with actual HTTP requests.
"""
import asyncio
import sys
import os
import httpx

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_URL = "http://localhost:8000/api/v2"


async def test_endpoints():
    """Test all v2 API endpoints."""
    print("Testing v2 API endpoints...")
    print("Note: This requires the backend server to be running on localhost:8000\n")
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        tests_passed = 0
        tests_failed = 0
        
        # Test Runtime endpoints
        print("Testing Runtime Bus endpoints...")
        try:
            response = await client.get(f"{BASE_URL}/runtime/status")
            if response.status_code == 200:
                print(f"  ✓ GET /runtime/status: {response.json()}")
                tests_passed += 1
            else:
                print(f"  ✗ GET /runtime/status: {response.status_code}")
                tests_failed += 1
        except Exception as e:
            print(f"  ✗ GET /runtime/status: {e}")
            tests_failed += 1
        
        try:
            response = await client.get(f"{BASE_URL}/runtime/topics")
            if response.status_code == 200:
                print(f"  ✓ GET /runtime/topics: {len(response.json())} topics")
                tests_passed += 1
            else:
                print(f"  ✗ GET /runtime/topics: {response.status_code}")
                tests_failed += 1
        except Exception as e:
            print(f"  ✗ GET /runtime/topics: {e}")
            tests_failed += 1
        
        # Test MPIE endpoints
        print("\nTesting MPIE endpoints...")
        try:
            response = await client.get(f"{BASE_URL}/mpie/status")
            if response.status_code == 200:
                print(f"  ✓ GET /mpie/status: {response.json()}")
                tests_passed += 1
            else:
                print(f"  ✗ GET /mpie/status: {response.status_code}")
                tests_failed += 1
        except Exception as e:
            print(f"  ✗ GET /mpie/status: {e}")
            tests_failed += 1
        
        try:
            response = await client.get(f"{BASE_URL}/mpie/store/nodes")
            if response.status_code == 200:
                print(f"  ✓ GET /mpie/store/nodes: {len(response.json())} nodes")
                tests_passed += 1
            else:
                print(f"  ✗ GET /mpie/store/nodes: {response.status_code}")
                tests_failed += 1
        except Exception as e:
            print(f"  ✗ GET /mpie/store/nodes: {e}")
            tests_failed += 1
        
        # Test DRG endpoints
        print("\nTesting DRG endpoints...")
        try:
            response = await client.get(f"{BASE_URL}/drg/status")
            if response.status_code == 200:
                print(f"  ✓ GET /drg/status: {response.json()}")
                tests_passed += 1
            else:
                print(f"  ✗ GET /drg/status: {response.status_code}")
                tests_failed += 1
        except Exception as e:
            print(f"  ✗ GET /drg/status: {e}")
            tests_failed += 1
        
        # Test Federation endpoints
        print("\nTesting Federation endpoints...")
        try:
            response = await client.get(f"{BASE_URL}/federation/peers")
            if response.status_code == 200:
                print(f"  ✓ GET /federation/peers: {len(response.json())} peers")
                tests_passed += 1
            else:
                print(f"  ✗ GET /federation/peers: {response.status_code}")
                tests_failed += 1
        except Exception as e:
            print(f"  ✗ GET /federation/peers: {e}")
            tests_failed += 1
        
        # Test Meta Learning endpoints
        print("\nTesting Meta Learning endpoints...")
        try:
            response = await client.get(f"{BASE_URL}/meta/domains")
            if response.status_code == 200:
                print(f"  ✓ GET /meta/domains: {len(response.json())} domains")
                tests_passed += 1
            else:
                print(f"  ✗ GET /meta/domains: {response.status_code}")
                tests_failed += 1
        except Exception as e:
            print(f"  ✗ GET /meta/domains: {e}")
            tests_failed += 1
        
        # Test Simulation endpoints
        print("\nTesting Simulation endpoints...")
        try:
            response = await client.get(f"{BASE_URL}/simulation/state")
            if response.status_code == 200:
                data = response.json()
                print(f"  ✓ GET /simulation/state: {len(data.get('nodes', []))} nodes, {len(data.get('edges', []))} edges")
                tests_passed += 1
            else:
                print(f"  ✗ GET /simulation/state: {response.status_code}")
                tests_failed += 1
        except Exception as e:
            print(f"  ✗ GET /simulation/state: {e}")
            tests_failed += 1
        
        # Test Health endpoints
        print("\nTesting Health endpoints...")
        try:
            response = await client.get(f"{BASE_URL}/health")
            if response.status_code == 200:
                print(f"  ✓ GET /health: {response.json()}")
                tests_passed += 1
            else:
                print(f"  ✗ GET /health: {response.status_code}")
                tests_failed += 1
        except Exception as e:
            print(f"  ✗ GET /health: {e}")
            tests_failed += 1
        
        # Summary
        print(f"\n{'='*50}")
        print(f"Tests passed: {tests_passed}")
        print(f"Tests failed: {tests_failed}")
        print(f"{'='*50}")
        
        return tests_failed == 0


if __name__ == "__main__":
    try:
        success = asyncio.run(test_endpoints())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nTest interrupted")
        sys.exit(1)
