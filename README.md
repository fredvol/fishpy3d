# Fishpy
09/07/2023  - Luc, Fred

This tools is a VPP for the FishKite project.

## Overview
This app is a Dash app. it is accessible via a browser ( Firefox,Chromium, Chrome)


## requirement 
 * python 3.10 +
 Several file of requirement exist :
  * requirement.txt  : the lightest only good for production
  * requirement_prod.txt  : the actual production env with version 
  * requirement_dev.txt  : env for developement
 
## activate env
 * Prod
    & c:/Users/fred/Kite_fish/fishpy/.venv/Scripts/Activate.ps1
 * Dev
    & c:/Users/fred/Kite_fish/fishpy3d/.venv_dev/Scripts/Activate.ps1


## Run the app 

The start point is : app.py

    c:; cd 'c:\Users\fred\Kite_fish\fishpy3d'; & 'c:\Users\fred\Kite_fish\fishpy3d\.venv_dev\Scripts\python.exe' 'c:\Users\fred\.vscode\extensions\ms-python.python-2023.16.0\pythonFiles\lib\python\debugpy\adapter/../..\debugpy\launcher' '61355' '--' 'C:\Users\fred\Kite_fish\fishpy3d\app.py' 

## create exe 
run
     python setup.py build

Uncomment in pages/app_2D.py

    # sys.path.append(os.getcwd())
    # print("open browser at http://127.0.0.1:8049/)
    # print("--")

Uncommemnt in app.py
    pages_folder = os.getcwd() + "/pages/"  # for exe
    print("cwd:", os.getcwd())