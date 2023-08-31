from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
import dash

dash.register_page(__name__)

layout = html.Div(
    [
        html.Br(),
        html.H2("FishPy : Select your model"),
        dbc.Stack(
            [
                html.Div(
                    dcc.Link(
                        dbc.Button(
                            page["name"],
                            size="lg",
                            className="me-1",
                        ),
                        href=page["relative_path"],
                    )
                )
                for page in dash.page_registry.values()
                if page["module"] != "/pages/.not_found_404"
            ],
            gap=2,
        ),
    ]
)
