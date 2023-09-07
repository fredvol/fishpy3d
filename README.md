# fishpy
fish


## requirement 
 * python 3.10 +
 
# activate env
    & c:/Users/fred/Kite_fish/fishpy/.venv/Scripts/Activate.ps1


# create exe 
run
     python setup.py build

Uncomment in pages/app_2D.py

    # sys.path.append(os.getcwd())
    # print("open browser at http://127.0.0.1:8049/)
    # print("--")

Uncommemnt in app.py
    pages_folder = os.getcwd() + "/pages/"  # for exe
    print("cwd:", os.getcwd())