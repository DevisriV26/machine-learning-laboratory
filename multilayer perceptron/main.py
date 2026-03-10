import json
import os

# Kaggle API credentials
kaggle_dict = {
    "username": "Devisri",
    "key": "YOUR_KAGGLE_API_KEY"
}

# Create kaggle.json file
with open("kaggle.json", "w") as file:
    json.dump(kaggle_dict, file)

# Create kaggle folder if not exists
os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)

# Move kaggle.json to kaggle folder
os.replace("kaggle.json", os.path.expanduser("~/.kaggle/kaggle.json"))

print("Kaggle API configured successfully!")
