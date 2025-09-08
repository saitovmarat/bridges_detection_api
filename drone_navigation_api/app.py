from typing import Type
from flask import Flask
from .api.depth_server_http import blueprint as depth_blueprint
from .api.command_server_udp import run_command_server_udp
from .settings import Config, DevConfig


def create_depth_http_server(config_object: Type[Config] = DevConfig):
    app = Flask(__name__)
    app.config.from_object(config_object)
    app.register_blueprint(depth_blueprint)       
    return app


def create_command_udp_server(host: str = "0.0.0.0", port: int = 9999):
    run_command_server_udp(host, port)
