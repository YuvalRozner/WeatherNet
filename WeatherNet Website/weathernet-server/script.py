import json

def main():
    # Create a simple dictionary to simulate some data
    data = {
        "message": "Hello from Python! Roznerrr",
        "status": "success"
    }
    
    # Convert the dictionary to a JSON string
    json_data = json.dumps(data)
    
    # Print the JSON string to stdout
    print(json_data)

if __name__ == "__main__":
    main() 