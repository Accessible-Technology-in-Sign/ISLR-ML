import os
import sys
import json
import requests
import threading
from queue import Queue
import argparse

# Function to send a POST request for each mp4 file
def send_request(file_name):
    url = "http://172.21.0.1"
    data = {"file_name": file_name}
    try:
        response = requests.post(url, json=data)
        print(f"Sent: {file_name}, Response: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to send {file_name}: {e}")

# Worker function for threads
def worker(q):
    while not q.empty():
        file_name = q.get()
        send_request(file_name)
        q.task_done()

# Main function to handle threading and processing
def main(folder_path, num_threads):
    # Get all mp4 files in the folder
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".mp4")]
    
    # Queue to hold the files
    q = Queue()
    for file in files:
        q.put(file)

    # Create threads
    threads = []
    for i in range(num_threads):
        t = threading.Thread(target=worker, args=(q,))
        t.start()
        threads.append(t)

    # Wait for the queue to be processed
    q.join()

    # Ensure all threads finish
    for t in threads:
        t.join()

if __name__ == "__main__":
    # CLI Argument parsing
    parser = argparse.ArgumentParser(description="Send MP4 file names to localhost:1234")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing mp4 files")
    parser.add_argument("--threads", type=int, default=32, help="Number of threads (default: 32)")

    args = parser.parse_args()
    
    # Run the main function with the folder path and the number of threads
    main(args.folder_path, args.threads)
