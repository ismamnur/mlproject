import os
import sys
import json
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging

@dataclass
class DataIngestionConfig:
    data_path: str  # Can be either a directory or a file path
    train_data_path: str = os.path.join('artifacts', "train_data.json")
    test_data_path: str = os.path.join('artifacts', "test_data.json")

class DataIngestion:
    def __init__(self, data_path):
        self.ingestion_config = DataIngestionConfig(data_path=data_path)

    def initiate_data_ingestion(self):
        logging.info("Data ingestion started")
        try:
            data_path = self.ingestion_config.data_path

            if os.path.isdir(data_path):
                # If the path is a directory, list files to find the JSON file
                files = os.listdir(data_path)
                json_files = [f for f in files if f.endswith('.json')]

                if not json_files:
                    raise FileNotFoundError("No JSON files found in the provided directory")

                # Use the first JSON file found
                json_file_path = os.path.join(data_path, json_files[0])
            elif os.path.isfile(data_path) and data_path.endswith('.json'):
                # If the path is a file, use it directly
                json_file_path = data_path
            else:
                raise ValueError("Provided path is neither a valid JSON file nor a directory containing JSON files")

            logging.info(f"Using JSON file: {json_file_path}")

            # Load the JSON data
            with open(json_file_path, 'r') as f:
                data = json.load(f)
            
            logging.info("JSON data loaded successfully")

            # Split the data into train and test sets
            train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

            # Ensure the directories exist for saving the split data
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.ingestion_config.test_data_path), exist_ok=True)

            # Save the train data
            with open(self.ingestion_config.train_data_path, 'w') as f:
                json.dump(train_data, f, indent=4)
            
            # Save the test data
            with open(self.ingestion_config.test_data_path, 'w') as f:
                json.dump(test_data, f, indent=4)
            
            logging.info("Train and test data saved successfully")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)
