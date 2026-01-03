from logfire.testing import CaptureLogfire
import inspect

try:
    # Try to see if it's a dataclass requiring arguments
    print(f"CaptureLogfire signature: {inspect.signature(CaptureLogfire.__init__)}")
    # Try to see if there is a helper method
    if hasattr(CaptureLogfire, 'capfire'):
        print("CaptureLogfire has .capfire")
except Exception as e:
    print(f"Error: {e}")
