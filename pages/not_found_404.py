from dash import Dash, html, dcc
import dash

dash.register_page(__name__)

layout = html.Div(
    [
        html.H2("Select your model"),
        html.Div(
            [
                html.Div(
                    dcc.Link(
                        f"{page['name']} - {page['path']}", href=page["relative_path"]
                    )
                )
                for page in dash.page_registry.values()
                if page["module"] != "pages.not_found_404"
            ]
        ),
    ]
)
