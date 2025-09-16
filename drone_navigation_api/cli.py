import click
from sympy import threaded
from .app import create_command_udp_server, create_depth_http_server


@click.group()
def cli():
    """CLI для запуска серверов BridgeVision."""
    pass


@cli.command()
@click.option('--host', default='0.0.0.0', help='Host для сервера')
@click.option('--port', default=9999, type=int, help='Порт UDP-сервера')
def udp(host, port):
    """Запускает UDP-сервер для приёма кадров и отправки детекций."""
    click.echo(f"🚀 Запуск UDP-сервера на {host}:{port}")
    create_command_udp_server(host=host, port=port)


@cli.command()
@click.option('--host', default='0.0.0.0', help='Host для сервера')
@click.option('--port', default=5000, type=int, help='Порт REST API')
@click.option('--debug', is_flag=True, help='Режим отладки')
def rest(host, port, debug):
    """Запускает REST API сервер."""
    app = create_depth_http_server()
    click.echo(f"🌐 Запуск REST API на http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    cli()
