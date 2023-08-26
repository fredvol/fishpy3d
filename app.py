import dash
import dash_bootstrap_components as dbc
import os

__version__ = "2.0.0"
pages_folder = os.path.join(os.path.dirname(__name__), "pages")  # for live
# pages_folder = os.getcwd() + "/pages/"  # for exe
# print("pages is:", pages_folder)
# print("cwd:", os.getcwd())

app = dash.Dash(
    __name__,
    use_pages=True,
    pages_folder=pages_folder,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)
server = app.server

navbar = dbc.NavbarSimple(
    dbc.DropdownMenu(
        [
            dbc.DropdownMenuItem(page["name"], href=page["path"])
            for page in dash.page_registry.values()
            if page["module"] != "pages.not_found_404"
        ],
        nav=True,
        label="Different model :",
    ),
    brand=f"FishPy v{__version__}",
    color="primary",
    dark=True,
    className="py-1",
)

app.layout = dbc.Container(
    [navbar, dash.page_container],
    fluid=True,
)

if __name__ == "__main__":
    app.run_server(debug=True)
