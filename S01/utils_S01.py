import logging
import os

def configure_logging(log_file_name: str) -> None:
    """Configure logging to write to both console and file."""
    log_file_path = os.path.join(os.path.dirname(__file__), "logs_S01", log_file_name)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),  # Console logging
            logging.FileHandler(log_file_path)  # File logging
        ]
    )
