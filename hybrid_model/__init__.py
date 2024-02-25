import logging
from .model import HybridModel


__all__ = ['HybridModel']

# Configure logging at the package level
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# Optionally, create a logger for your package
logger = logging.getLogger(__name__)
logger.info("Logging is configured for the hybrid_model package.")