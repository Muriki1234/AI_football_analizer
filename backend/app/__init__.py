from flask import Flask
from flask_cors import CORS
from config import config

def create_app(config_name='default'):
    app = Flask(__name__)
    app.config.from_object(config[config_name])

    CORS(app)

    from .routes.main import main as main_blueprint
    app.register_blueprint(main_blueprint)

    from .routes.api import api as api_blueprint
    app.register_blueprint(api_blueprint)

    from .routes.analysis import analysis as analysis_blueprint
    app.register_blueprint(analysis_blueprint)

    return app
