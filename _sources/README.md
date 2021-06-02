# Assigments

# How to get started
Install a virtual environment on the highest level of this project folder by typing ```python -m venv ML2``` in your command line (accessed through such tools as Git Bash, Windows Power Shell, or the Command Prompt). Please make sure, that you do this as administrator, otherwise you may face issues. Especially for Windows user using the PowerShell you should before continuing check you ```Get-ExecutionPolicy```. If that returns anything else than ```Unrestricted``` you may want to set it to ```Set-ExecutionPolicy Unrestricted```. This enables you later on to activate visablely the virtual enviroment. Windows users should then proceed to activate this environment by inserting ```.\ML2\Scripts\activate.ps1``` to their command line. As a Windows user using PowerShell you should now see a green (ML2) at the beginning of your command line. This signals you, that you virtual environment is activated. For macOS use ```source ./ML2/bin/activate``` instead. Thereafter we can install all the required packages with ```pip install -r requirements.txt```. Doing so will install all the packages defined in the ```requirements.txt``` file. Once we are done with this environment (i.e. the packages have been installed) we can deactivate it by entering ```./ML2/Scripts/deactivate.bat```.

# Access to Wharton

1. Start WinSCP
2. Log in using creds
3. Creating virtual environment with older python version using `/usr/local/sas/grid/python3-3.8.3/bin/python3 -m venv --copies <env_name>`.
4. Activate virtual environment with `source ~/<env_name>/bin/activate`.