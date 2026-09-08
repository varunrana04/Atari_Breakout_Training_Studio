import asyncio
import websockets
import json

async def spam_websocket(worker_id):
    uri = "ws://localhost:8001/ws/train"
    try:
        async with websockets.connect(uri) as ws:
            # Randomly start
            await ws.send(json.dumps({'cmd': 'start', 'hyperparams': {}}))
            # Just listen for a bit
            for _ in range(5):
                await ws.recv()
            # Randomly pause
            await ws.send(json.dumps({'cmd': 'pause'}))
            await asyncio.sleep(0.1)
            # Hard disconnect
            # (Context manager exits here)
        print(f"Worker {worker_id} connected, spammed, and violently disconnected.")
    except Exception as e:
        print(f"Worker {worker_id} failed: {e}")

async def main():
    print("Starting Project 2 WebSocket Stress Test...")
    tasks = [spam_websocket(i) for i in range(100)]
    await asyncio.gather(*tasks)
    print("Project 2 WebSocket Stress Test Completed.")

if __name__ == "__main__":
    asyncio.run(main())
