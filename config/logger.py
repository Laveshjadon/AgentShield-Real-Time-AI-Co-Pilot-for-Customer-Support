"""Sets up structured logging for the project."""

import logging
import sys
import os
import structlog  


structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)


logging.basicConfig(
    format="%(message)s",
    stream=sys.stdout,
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
)


def get_logger(module_name: str):
    """returns the structlog thingy so we can log properly"""
    return structlog.get_logger(f"agentshield.{module_name}")


if __name__ == "__main__":
    logger = get_logger("test")
    logger.info("structured_log_test", key="value", number=42)
    logger.warning("warning_test", module="config.logger")
    logger.error("error_test", code=500)
    print("Logger test complete!")
