# Deep Learning for Predicting Space Weather Events Using NASA Data

## How to run in VS Code

1. Open this folder in VS Code.
2. Open the VS Code terminal.
3. Run:

```bash
python -m venv .venv
```

4. Activate the virtual environment:

Windows PowerShell:

```bash
.venv\Scripts\activate
```

Mac/Linux:

```bash
source .venv/bin/activate
```

5. Install the packages:

```bash
pip install -r requirements.txt
```

6. Run the full project:

```bash
python main.py
```

## Screenshot order

1. VS Code project folders
2. data/space_weather_dataset.csv opened
3. Terminal after running preprocess.py
4. LSTM model summary from train_model.py
5. Training epochs from train_model.py
6. Evaluation metrics from evaluate.py
7. outputs/loss_curve.png
8. outputs/prediction_vs_actual.png
9. outputs/prediction_scatter.png
10. models/space_weather_lstm.keras
11. outputs folder showing generated files
12. Final terminal output after predict.py

## Individual commands

```bash
python src/create_dataset.py
python src/preprocess.py
python src/train_model.py
python src/evaluate.py
python src/visualize.py
python src/predict.py
```
