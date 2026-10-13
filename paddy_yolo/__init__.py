"""Paddy-YOLO: An accurate method for rice pest detection."""

from paddy_yolo.modules.register import register_paddy_modules
from paddy_yolo.compat import install_metrics_compatibility

__version__ = "0.1.0"

# Automatically register custom Paddy-YOLO modules into Ultralytics on import
install_metrics_compatibility()
register_paddy_modules()

__all__ = ["register_paddy_modules", "__version__"]
