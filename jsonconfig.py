import os
import json

def create_json(directory):
    image_data = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                input_filename = file

                image_data.append({
                    "input_filename": input_filename,
                    "result": "yes"
                })

    return {"image_data": image_data}

if __name__ == "__main__":
    input_directory = "melanomapics\melanoma"

    # Create JSON data from the images
    json_data = create_json(input_directory)

    # Save the JSON data to a file named "image_data.json" in the same directory as the script
    with open("image_data.json", "w") as json_file:
        json.dump(json_data, json_file, indent=2)

    print("JSON file created successfully.")
