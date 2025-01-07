import logging
import os
from datetime import datetime

LOG_FILE=f"{datetime.now().strftime('%Y-%m-%d')}.log"  #text file
LOG_PATH=os.path.join(os.getcwd(),"logs",LOG_FILE)
os.makedirs(LOG_PATH,exist_ok=True)

LOG_FILE_PATH=os.path.join(LOG_PATH,LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] [ %(levelname)s ] [ %(filename)s ] [ %(funcName)s ] [ %(message)s ]",
    level=logging.INFO,
)

# if __name__=="__main__":
#     logging.info("Logging has started")