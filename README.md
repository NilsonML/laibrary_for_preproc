# MPIIGaze preprocessing

Small Python library for exploratory analysis and preprocessing of the **MPIIGaze** gaze-estimation dataset (Max Planck Institute for Informatics) annotations. One call runs the whole pipeline and produces plots, a Markdown report and train/test arrays for a model that predicts **where on the screen a person is looking from their head pose**.

**Stack:** Python · pandas · NumPy · scikit-learn · matplotlib / seaborn

## Pipeline

| Step | What it does | Output |
|---|---|---|
| 1. Load & clean | Parses `annotation.txt` (41 columns: eye landmarks, on-screen gaze, 3D gaze, head pose, eye centers), drops duplicates, fills gaps | DataFrame |
| 2. Correlations | Correlation matrix of gaze and head-pose features | `correlation_matrix.png` |
| 3. Scatter plots | Pairwise relations of key features | `scatter_plots.png` |
| 4. Clustering | K-means (k=3) on standardized on-screen gaze points | `clustering_result.png` |
| 5. Distributions | Histograms and summary statistics | `distributions.png` |
| 6. Outliers | IQR filter on key features | — |
| 7. Modeling prep | Head pose → gaze (x, y); train/test split, then scaling fitted on train only | `X_train, X_test, y_train, y_test` |
| 8. Report | Everything above in Markdown | `report_A.md` |

## Usage

```bash
pip install .
```

```python
from mpiigaze_processor import MPIIGazeProcessor

proc = MPIIGazeProcessor(base_path="MPIIGaze", output_dir="results")
proc.process_complete_analysis(participant="p00", day="day01")

# or step by step
df = proc.load_and_clean_data("p00", "day01")
X_train, X_test, y_train, y_test = proc.prepare_for_modeling(proc.remove_outliers(df))
pred_pixels = proc.scaler_y.inverse_transform(y_test)   # back to screen coordinates
```

The dataset is expected in `MPIIGaze/Data/Original/<participant>/<day>/annotation.txt`.

## Notes and limitations

- Scalers are fitted **after** the train/test split, on the training part only — fitting them on all data would leak test statistics.
- The split is random by row. Consecutive frames of one recording are nearly identical, so for an honest estimate split by **participant** (leave-one-person-out) instead.
- One participant/day is processed per call.

Originally written for a data-preprocessing competition task.
