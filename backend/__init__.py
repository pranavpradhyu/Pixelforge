"""Application factory."""
from flask import Flask

from config import Config


def create_app(config: type[Config] = Config) -> Flask:
    app = Flask(
        __name__,
        static_folder="../frontend/static",
        template_folder="../frontend/templates",
    )
    app.config.from_object(config)

    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    from backend.routes import bp
    app.register_blueprint(bp)

    # opportunistic cleanup of expired files on boot
    from backend import storage
    storage.sweep_expired()

    return app
