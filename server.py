from waitress import serve
import os
from app import server
from threading import Timer
import webbrowser

port=8049


def open_browser():
    if not os.environ.get("WERKZEUG_RUN_MAIN"):  # to prevent to open twice
        
        webbrowser.open_new(f'http://127.0.0.1:{port}/')

Timer(1, open_browser).start()
serve(server, port=port)
