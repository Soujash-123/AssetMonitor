import pandas as pd
import requests
import json
import sys
import time

# API endpoint
API_URL = "http://127.0.0.1:5000/predict"

# Read CSV data
def read_csv_in_batches(file_path, batch_size=300):
    try:
        data = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)
    
    # Split data into chunks of 'batch_size'
    for i in range(0, data.shape[0], batch_size):
        yield data.iloc[i:i + batch_size].to_dict(orient="records")

# Send data to the API and print response
def send_data_to_api(batch):
    try:
        response = requests.post(API_URL, headers={"Content-Type": "application/json"}, data=json.dumps(batch))
        response.raise_for_status()  # Check for HTTP request errors
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error calling API: {e}")
        return {"error": str(e)}

# Main function
avg_time = []
def main(csv_file_path):
    for batch in read_csv_in_batches(csv_file_path):
        start = time.time()  # Start time before sending the batch
        result = send_data_to_api(batch)
        print("Batch Result:", json.dumps(result, indent=4))
        end = time.time()  # End time after receiving the response
        
        # Calculate and print time taken for this batch
        time_taken = end - start
        print(f"Time taken for this batch: {time_taken:.4f} seconds")
        
        avg_time.append(time_taken)  # Store time taken for calculating average later
    
    # Calculate and print average time per execution
    if avg_time:
        print("Average time per execution:", sum(avg_time) / len(avg_time))

# Replace 'input_data.csv' with your CSV file name
if __name__ == "__main__":
    main("dataset2.csv")
