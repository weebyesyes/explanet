
This project provides a front-end for exploring exoplanet data. Follow the instructions below to get the development environment running locally.

## Prerequisites

- [Node.js](https://nodejs.org/) (the version specified in `package.json`'s `engines`, or the latest LTS release)
- [npm](https://www.npmjs.com/)
- [Python 3.10](https://www.python.org/downloads/release/python-3100/) (required for CSV submission features)

## Installing dependencies

1. Clone this repository and open it in your terminal.
2. Install JavaScript dependencies:
   ```bash
   npm install
   ```

## Running the development server

Start the Next.js development server with:

```bash
npm run dev
```

The app will be available at [http://localhost:3000](http://localhost:3000).

## CSV submission requirements (Python virtual environment)

Uploading CSV files requires a local Python environment. The application expects Python 3.10, so make sure you create and activate a Python 3.10 virtual environment before attempting to submit CSV data. If the environment is missing, the app will warn you.

### Creating a Python 3.10 virtual environment

> **Important:** Use Python 3.10 specifically. If you have multiple Python versions installed, replace `python3.10` with the path to your Python 3.10 executable.
>
> On Windows, the commands below work in Git Bash, PowerShell, or Command Prompt (see the activation steps).

Run the following commands from the project root:

```bash
# Create the virtual environment in a `.venv` folder
python3.10 -m venv .venv

# Activate the virtual environment (choose the command for your shell)
# macOS/Linux (bash/zsh/Git Bash)
source .venv/bin/activate

# macOS/Linux (fish)
source .venv/bin/activate.fish

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Windows (Command Prompt)
.\.venv\Scripts\activate.bat

# Windows (Git Bash)
source .venv/Scripts/activate
```

Once activated, install the required Python packages. The CSV pipeline relies on the libraries used in `exo_infer.py`:

```bash
pip install pandas numpy scikit-learn lightgbm joblib

# Optional but recommended if you plan to use the full model bundle
pip install catboost xgboost
```

If youd like to try the version we have used (should not be necessary in python 3.10), use these:
```
pip install "numpy==1.23.5" "scipy==1.9.3" "pandas==1.5.3"
pip install "scikit-learn==1.2.2"
pip install "lightgbm==3.3.5" "xgboost==1.7.6" "catboost==1.2"
```

If a `requirements.txt` file is added later, prefer installing from that file:

```bash
pip install -r requirements.txt
```

When you are finished working, deactivate the environment with:

```bash
deactivate
```

## Submitting CSV files

After activating the `.venv` and installing the dependencies, restart the development server if it was running. The CSV submission functionality will now have access to the required Python runtime.

## Troubleshooting

- If `npm run dev` fails, ensure dependencies were installed with `npm install` and the correct Node.js version is in use.
- If CSV upload complains about missing Python, confirm that the `.venv` is activated and you are using Python 3.10.
- If you encounter Python package errors, re-run the `pip install` commands from within the activated `.venv`.

## Additional tips

- To keep your environment clean, use `.venv` consistently and avoid committing it to source control (it is typically listed in `.gitignore`).
- Re-run `pip install` whenever Python dependencies change.
