# Steel Energy Streamlit — beginner's Windows setup

You need no Streamlit account, SAS account, API key or GitHub account. You need **Windows, Python, a browser**, and an internet connection only to install Python packages the first time.

## 1. Download and unpack
Download `Steel_Energy_Streamlit_Standalone.zip`. Right-click → **Extract All...** → choose Desktop. Open the extracted **Steel_Energy_Streamlit_Standalone** folder. You should see `app.py`, `train_model.py`, `requirements.txt`, `models`, and `data`. **Do not work from inside the ZIP preview.**

## 2. Install Python (once)
Open https://www.python.org/downloads/ and install a supported Python version (Python 3.12 or 3.13 recommended for a beginner). On a traditional Windows installer, enable **Add Python to PATH** if offered. Close and reopen Command Prompt after installation.

Press Windows key, type **cmd**, open **Command Prompt** (not PowerShell), type:

    py --version

You should see a Python version. If `py` is unavailable, try `python --version` and use `python` in place of `py` below.

## 3. Open Command Prompt directly in the project folder
Open the unzipped project folder in File Explorer. Click the **address bar** at the top, type `cmd`, and press Enter. A black Command Prompt window should open inside this folder. This avoids having to type the folder's path manually.

## 4. Create an isolated environment (once)
In that Command Prompt, type and press Enter:

    py -m venv .venv

If `py` did not work in step 2, use `python -m venv .venv` instead.

## 5. Activate it (every time you reopen Command Prompt)

    .venv\Scripts\activate.bat

Look for `(.venv)` at the start of the prompt. This step works in **Command Prompt**; PowerShell uses a different activation command.

## 6. Install the libraries (once)

    python -m pip install -r requirements.txt

This installs Streamlit, pandas, scikit-learn, numpy and joblib in the local environment. You need internet for this step.

## 7. Launch the application

    python -m streamlit run app.py

It usually opens automatically in your browser at `http://localhost:8501`. Otherwise type that URL into the browser manually. No account is needed.

## 8. Try an example
Select a built-in **SAS test-period example** in the app and click **Generate predictions**. You should see a kWh estimate, operating load class, and chart of class probabilities. Try manual inputs next.

## 9. Stop or restart
To stop the application, return to Command Prompt and press **Ctrl+C**. On a later day, open a new cmd in the folder, run `.venv\Scripts\activate.bat`, then `python -m streamlit run app.py`. You do not reinstall everything.

## 10. OPTIONAL: reproduce the Python classifier locally
A trained `models/python_random_forest.joblib` is already included. You do NOT need to train it just to run the app. If your professor wants you to demonstrate the Python training workflow, with the same environment activated run:

    python train_model.py

This trains a **new Python Random Forest** from the included raw CSV, performs the chronological 70/30 split, and regenerates the model, test metrics and demo examples. It does **not** recreate the exact SAS Viya Random Forest.

## Common beginner mistakes
- **'py' is not recognized:** Try `python --version`. If neither works, install Python and reopen Command Prompt.
- **'No module named streamlit':** Activate `.venv` and run `python -m pip install -r requirements.txt`.
- **'can't open file app.py':** Command Prompt is in the wrong folder. Open the folder in Explorer and enter `cmd` in its address bar again.
- **Port 8501 in use:** Run `python -m streamlit run app.py --server.port 8502` then open `http://localhost:8502`.
- **Windows asks about network access:** For a local-only demonstration, do not expose the app to the public internet.
- **An older joblib model causes a compatibility warning:** Run `python train_model.py` inside your installed environment to rebuild the Python classifier.

## Academic integrity: wording to use
"The regression component deploys the coefficient equation exported from the SAS Studio Forward Selection model. The classification component is an independent Python Random Forest reproduction trained with comparable inputs and the same chronological split; it is not the registered SAS Viya forest. SAS modelling and classification evaluation results are reported separately from Python model validation."

## Limits
- No SAS dependency for normal use.
- A model trained on 2018 industrial data is not an operationally validated industrial forecast.
- The app is not proof of SAS forest deployment. The SAS Random Forest remains registered/published separately in SAS Model Manager.
