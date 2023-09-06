from setuptools import find_packages
from cx_Freeze import setup, Executable


options = {
    "build_exe": {
        "include_files": [
            "pages/",
            "data/",
            "assets/",
            "model.py",
            "model_3d.py",
            "fish_plot.py",
            "fish_plot_3d.py",
            "app_components_3d.py",
            "app_components.py",
        ],
        "includes": [
            "cx_Logging",
            "idna",
        ],
        "packages": [
            "asyncio",
            "flask",
            "jinja2",
            "dash",
            "plotly",
            "waitress",
            "dash_daq",
            "jsonpickle",
            "openpyxl",
        ],
        "excludes": ["tkinter"],
    }
}


executables = [
    Executable(
        "server.py", base="console", icon="Icone_fish.ico"
    )  # targetName="halliburton_dash_rig.exe")
]

setup(
    name="fishpy3d_app",
    packages=find_packages(),
    version="2.0.0",
    description="fishpy_app",
    executables=executables,
    options=options,
)
