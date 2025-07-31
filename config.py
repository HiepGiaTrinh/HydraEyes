"""
Configuration settings for H2 Factory Camera Monitoring System
"""

import os


class Config:
    """Base configuration class"""

    SECRET_KEY = os.environ.get("SECRET_KEY") or "dev-secret-key-change-in-production"

    # Model paths
    DIGITAL_MODEL_PATH = os.environ.get("DIGITAL_MODEL_PATH", "best.pt")
    ANALOG_MODEL_PATH = os.environ.get("ANALOG_MODEL_PATH", "gauge_reader_web/models")

    # Server settings
    HOST = os.environ.get("HOST", "0.0.0.0")
    PORT = int(os.environ.get("PORT", 5000))


class DevelopmentConfig(Config):
    """Development configuration"""

    DEBUG = True
    TESTING = False


class ProductionConfig(Config):
    """Production configuration"""

    DEBUG = False
    TESTING = False

    # Use stronger secret key in production
    SECRET_KEY = os.environ.get("SECRET_KEY")
    if not SECRET_KEY:
        raise ValueError("No SECRET_KEY set for production environment")


class TestingConfig(Config):
    """Testing configuration"""

    DEBUG = False
    TESTING = True


# Configuration dictionary
config = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
    "default": DevelopmentConfig,
}
