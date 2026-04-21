import asyncio
import json
import time
from async_workers import risk_worker, temporal_worker_placeholder, cluster_worker_placeholder

async def producer(filepath: str, queues: list[asyncio.Queue]):
    """Reads the log file and fans out the data to all worker queues."""
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                log_entry = json.loads(line)
                for q in queues:
                    await q.put(log_entry)

    # Shut down workers gracefully
    for q in queues:
        await q.put(None)

async def main():
    print("🚀 Booting High-Concurrency Evaluation Pipeline...")
    start_time = time.time()

    risk_queue = asyncio.Queue()
    temporal_queue = asyncio.Queue()
    cluster_queue = asyncio.Queue()
    
    queues = [risk_queue, temporal_queue, cluster_queue]
    risk_results = [] 

    workers = [
        asyncio.create_task(risk_worker(risk_queue, risk_results)),
        asyncio.create_task(temporal_worker_placeholder(temporal_queue)),
        asyncio.create_task(cluster_worker_placeholder(cluster_queue))
    ]

    await producer('dummy_logs.jsonl', queues)

    for q in queues:
        await q.join()

    await asyncio.gather(*workers)

    with open('final_risk_reports.json', 'w') as f:
        json.dump(risk_results, f, indent=2)

    end_time = time.time()
    print(f"✅ Pipeline Execution Complete in {end_time - start_time:.2f} seconds.")
    print(f"📊 Processed {len(risk_results)} records through the Risk Engine.")

if __name__ == "__main__":
    asyncio.run(main())