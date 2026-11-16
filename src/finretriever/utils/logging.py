"""Logging configuration.

Single entry point for setting up structured stdlib logging. Scripts call
``configure_logging()`` once at startup so every module gets consistent output.
"""

# TODO(v1): `configure_logging(level="INFO")` setting a sensible format and a
#           single StreamHandler.
# TODO(v1): `get_logger(name)` thin wrapper around logging.getLogger.
