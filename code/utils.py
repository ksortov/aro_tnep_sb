import logging

# Create logger
logger = logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG)  # Set the lowest level to capture all messages

# Create file handler
file_handler = logging.FileHandler('my_log_file.log')
file_handler.setLevel(logging.DEBUG)  # Log all levels to the file

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Show INFO+ messages in the console

# Create formatter and set it for both handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add both handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)