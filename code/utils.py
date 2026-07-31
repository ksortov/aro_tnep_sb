import logging
import sys
import requests
import traceback
import psutil
import threading
import time

# Create logger
logger = logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG)  # Set the lowest level to capture all messages

# Create file handler
file_handler = logging.FileHandler('my_log_file.log', mode='w')
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


def notify_mobile(topic="kevin_aro_tnep_job_0919", title="Execution Update", message="Job finished", priority="high", tags="white_check_mark"):
    """Sends a notification to a mobile phone via ntfy.sh."""
    try:
        # Sanitize Title header to ASCII to prevent latin-1 HTTP header encoding errors in urllib3
        safe_title = title.encode('ascii', 'ignore').decode('ascii').strip()
        headers = {
            "Title": safe_title if safe_title else "Execution Update",
            "Priority": priority,
            "Tags": tags
        }
        requests.post(
            f"https://ntfy.sh/{topic}",
            data=message.encode('utf-8'),
            headers=headers
        )
    except Exception as e:
        logger.error(f"Failed to send ntfy notification: {e}")


def setup_ntfy_exception_handler(topic="kevin_aro_tnep_job_0919", script_name="multi_year_aro_tnep.py"):
    """Sets up a global excepthook to automatically send an urgent ntfy alert if the script crashes."""
    def handle_exception(exctype, value, tb):
        error_details = "".join(traceback.format_exception(exctype, value, tb))
        short_error = f"Error Type: {exctype.__name__}\nMessage: {value}\n\nTraceback:\n{error_details[-300:]}"
        notify_mobile(
            topic=topic,
            title=f"{script_name} FAILED",
            message=short_error,
            priority="urgent",
            tags="x,warning"
        )
        sys.__excepthook__(exctype, value, tb)

    sys.excepthook = handle_exception

class MemoryTracker:
    """Monitors peak RAM usage (in MB/GB) of Python and child solver processes."""
    def __init__(self, interval=0.5):
        self.interval = interval
        self.peak_bytes = 0
        self._running = False
        self._thread = None
    def _monitor(self):
        current_proc = psutil.Process()
        while self._running:
            try:
                # Sum memory of Python process + GAMS/solver child processes
                mem = current_proc.memory_info().rss
                for child in current_proc.children(recursive=True):
                    mem += child.memory_info().rss
                if mem > self.peak_bytes:
                    self.peak_bytes = mem
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            time.sleep(self.interval)
    def start(self):
        self.peak_bytes = 0
        self._running = True
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()
    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        return self.get_peak_formatted()
    def get_peak_mb(self):
        return round(self.peak_bytes / (1024 * 1024), 2)
    def get_peak_formatted(self):
        mb = self.get_peak_mb()
        if mb >= 1024:
            return f"{mb / 1024:.2f} GB ({mb:.0f} MB)"
        return f"{mb:.1f} MB"

