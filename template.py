import os
from pathlib import Path
import logging


logging.basicConfig(level=logging.INFO, format="[%(asctime)s]: %(message)s:")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s]: %(message)s:")

# Embedded content of exception.py and logger.py
EXCEPTION_CONTENT = 'import sys\n\n\n# ________________ DEF THE ERROR MESSAGE ________________ #\ndef error_message_detail(error, error_detail: sys):\n    """\n    This function is to return a message regarding error details occuring in the execution of the code\n\n    """\n\n    # no interest in the 1st and 2nd items in the return of the exc_info()\n    _, _, exc_tb = error_detail.exc_info()\n\n    file_name = exc_tb.tb_frame.f_code.co_filename\n    error_message = "Error occured in the script, name: [{0}], line number: [{1}] error message: [{2}]".format(\n        file_name, exc_tb.tb_lineno, str(error)\n    )\n\n    return error_message\n\n\n# ________________ MAKE ERROR CAPTURE HANDLER ________________ #\nclass CustomException(Exception):\n    def __init__(self, error_message, error_detail: sys):\n        super().__init__(error_message)\n        self.error_message = error_message_detail(\n            error_message, error_detail=error_detail\n        )\n\n    def __str__(self):\n        return self.error_message\n'

LOGGER_CONTENT = 'import logging, os\nfrom datetime import datetime\n\nLOG_FILE = f"{datetime.now().strftime(\'%m _%d_%Y_%H_%M_%S\')}.log"\nlogs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)\nos.makedirs(logs_path, exist_ok=True)  # keep on appending the file\n\nLOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)\n\nlogging.basicConfig(\n    filename=LOG_FILE_PATH,\n    # recommended format\n    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",\n    level=logging.INFO,\n)\n'

# List of files and directories
list_of_files = [
    "input",
    "input/data",
    "input/viz",
    "output",
    "output/data",
    "output/viz",
    "notebook",
    "notebook/EDA.ipynb",
    "notebook/draft.ipynb",
    "logger.py",
    "exception.py",
    "test_1batch.py",
    "test_minibatch.py",
    "utils_1batch.py",
    "utils_data.py",
    "utils_minibatch.py",
    "requirements.txt",
]

for file_path in list_of_files:
    file_path = Path(file_path)

    # If it's a directory or doesn't have a dot (assuming it's a directory)
    if file_path.is_dir() or "." not in file_path.name:
        if not file_path.exists():
            os.makedirs(file_path, exist_ok=True)
            logging.info(f"Created directory: {file_path}")
        else:
            logging.info(
                f"Directory {file_path} already exists => re-creating ignored."
            )
    else:
        file_dir, file_name = os.path.split(file_path)

        if file_dir != "" and not Path(file_dir).exists():
            os.makedirs(file_dir, exist_ok=True)
            logging.info(f"Created directory: {file_dir} for the file {file_name}")

        # If the file does not exist or its size is 0
        if not file_path.exists() or file_path.stat().st_size == 0:
            with open(file_path, "w") as file:
                # Write the content of exception.py if the file is exception.py
                if file_name == "exception.py":
                    file.write(EXCEPTION_CONTENT)
                # Write the content of logger.py if the file is logger.py
                elif file_name == "logger.py":
                    file.write(LOGGER_CONTENT)
                # Else, just create an empty file
                else:
                    pass
            logging.info(f"Created file: {file_path}")
        else:
            logging.info(
                f"File {file_path} already exists and is not empty => re-creating ignored."
            )

if __name__ == "__main__":
    print("Project structure generated successfully!")
