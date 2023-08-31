import dash
import dash_bootstrap_components as dbc
import os


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

if __name__ == "__main__":
    app.run_server(debug=True)
