from flask import Flask
from flask_cors import CORS
from config import config

def create_app(config_name='default'):
    app = Flask(__name__)
    app.config.from_object(config[config_name])

    CORS(app)

    from .routes.main import main as main_blueprint
    app.register_blueprint(main_blueprint)

    return app
