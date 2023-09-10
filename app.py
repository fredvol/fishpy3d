import dash
import dash_bootstrap_components as dbc
import os
from threading import Timer
import webbrowser


# # for live
# pages_folder = os.path.join(os.path.dirname(__name__), "pages")  # for live

# for build
# pages_folder = os.getcwd() + "/pages/"  # for exe
# print("cwd:", os.getcwd())


app = dash.Dash(
    __name__,
    use_pages=True,
    pages_folder=os.getcwd() + "/pages/",
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    assets_folder=os.getcwd() + "/assets/",
)
server = app.server

app.layout = dbc.Container(
    [dash.page_container],
    fluid=True,
)

port = 8049


def open_browser():
    if not os.environ.get("WERKZEUG_RUN_MAIN"):  # to prevent to open twice
        webbrowser.open_new(f"http://127.0.0.1:{port}/")


if __name__ == "__main__":
    Timer(1, open_browser).start()

    app.run_server(debug=True, port=port)
    # app.run_server(host="0.0.0.0", debug=True)
