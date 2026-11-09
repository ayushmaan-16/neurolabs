# Upload a different pretrained model yourself

Use [bernardozschaber/titanic-federated-dp-mlp on Hugging Face](https://huggingface.co/bernardozschaber/titanic-federated-dp-mlp).
It is an educational Titanic passenger classifier, different from the Iris and heat-equation
examples. The author supplies the weights, architecture and preprocessing statistics.

The actual checkpoint has **978 parameters** and this architecture:

```text
12 input values -> 32 neurons + ReLU -> 16 neurons + ReLU -> 2 linear outputs
```

That is three dense layers, two hidden layers, and four columns of neurons including the
input. It fits the current importer. The model card says 986 parameters; summing the actual
weight/bias tensors gives 978. This guide uses pinned source revision
`f6761cc2249fafe1fa037bd9bfddb1516764bd7b` so future repository changes do not change the exercise.

The two outputs are **logits** (scores), not probabilities. Index 0 means did not survive;
index 1 means survived. The larger score determines the predicted class. For example,
`[-1.15, -1.75]` selects class 0 because -1.15 is larger. This exercise inspects computations;
it does not evaluate prediction accuracy.

## 1. Open PowerShell in your neuroLabs project

For the existing installation on this computer:

```powershell
cd "C:\Users\ayush\OneDrive\Documents\ChatGPT\NeuroLens"
```

The commands below use the project's existing `.venv` and the new helper
`examples/export_titanic_checkpoint.py`. You do not need to install Hugging Face's Python
client, pandas, or an ONNX converter. On a fresh checkout, first follow the repository's
setup instructions (`uv sync --locked`, `npm ci`, `npm run build`).

## 2. Download two files from Hugging Face

Run these commands in that terminal:

```powershell
New-Item -ItemType Directory -Force -Path ".local\titanic-demo" | Out-Null
$checkpointBase = "https://huggingface.co/bernardozschaber/titanic-federated-dp-mlp/resolve/f6761cc2249fafe1fa037bd9bfddb1516764bd7b"
curl.exe --fail --location "$checkpointBase/model.safetensors" --output ".local\titanic-demo\model.safetensors"
curl.exe --fail --location "$checkpointBase/preprocessing.json" --output ".local\titanic-demo\preprocessing.json"
```

Alternatively, open the pinned revision's
[Files page](https://huggingface.co/bernardozschaber/titanic-federated-dp-mlp/tree/f6761cc2249fafe1fa037bd9bfddb1516764bd7b),
download these same two files, and place them in `.local\titanic-demo`.

| File | What it contains |
|---|---|
| `model.safetensors` | The trained connection weights and neuron biases |
| `preprocessing.json` | Feature order, title encoding, missing-value defaults, and training means/standard deviations |

The helper reconstructs the exact small architecture documented in the author's
[inference.py](https://huggingface.co/bernardozschaber/titanic-federated-dp-mlp/blob/f6761cc2249fafe1fa037bd9bfddb1516764bd7b/inference.py).
You do not need to download or run that Python file.

Optional download checks using `Get-FileHash <file> -Algorithm SHA256`:

```text
model.safetensors: F6FEEDB9726AA2023AF5CD1B06332DE6470315021A00F0856778F49FE5EEF5EE
preprocessing.json: BA69B4F07B50EED52A77775B08B05EE392867E89F69A8C93735C067B1FD35D7F
```

## 3. Prepare the upload ZIP

Run this as one line:

```powershell
.\.venv\Scripts\python.exe .\examples\export_titanic_checkpoint.py .\.local\titanic-demo\model.safetensors .\.local\titanic-demo\preprocessing.json .\.local\titanic-demo\titanic.zip
```

The helper loads the trained parameters into the matching architecture, preprocesses eight
fictional passenger profiles from `examples/titanic-passengers.csv`, and checks the copied
model against the source checkpoint's forward equations. It then verifies the package
with neuroLabs' runtime. There is no training step.

Expect messages including `978 parameters`, verification `status: passed`, and
`sample_count: 8`. On the tested local CPU, maximum absolute error was zero; tiny float32
rounding differences on other hardware may still pass the importer's fixed tolerance.

It creates these two files:

| File | Where you use it |
|---|---|
| `.local\titanic-demo\titanic.zip` | **Import model**: architecture, trained weights, metadata and verification examples |
| `.local\titanic-demo\titanic-inputs.csv` | **Diagnostics → Load CSV**: eight preprocessed rows, each with exactly 12 numeric values |

Upload the generated ZIP without extracting or repacking it. Zipping the two downloaded
source files yourself will not create a neuroLabs package.

## 4. Upload it through the website

If the app is already running, open [your local neuroLabs](http://127.0.0.1:8000).
Otherwise, keep another PowerShell terminal open in the project and run:

```powershell
.\.venv\Scripts\python.exe -m uvicorn neurolabs.api:create_app --factory --host 127.0.0.1 --port 8000
```

Then:

1. Click **Import model** in the model collection.
2. Under **Model package (.zip)**, choose `.local\titanic-demo\titanic.zip`.
3. Review the preview: **12 inputs**, **978 parameters**, task **classification**,
   dense shapes **32 x 12**, **16 x 32**, **2 x 16**, two ReLU operations, and output
   verification passed on **8 supplied examples**.
4. Click **Import into workspace**.
5. Select the imported **Titanic survival MLP** if it is not already selected.
6. Open **Architecture**, then click **Run & trace**. You should see 12 input neurons,
   32 neurons, 16 neurons, and 2 output neurons. **All connections** can show all 928 weights.

The first fictional passenger's preprocessed vector is already the default input.
The two original output scores should be approximately **[-1.148907, -1.745681]**,
so the predicted class is **0**. These are expected model scores, not known ground truth.

## 5. Check neuron inactivity across a batch

1. Open **Diagnostics**.
2. Click **Load CSV** and choose `.local\titanic-demo\titanic-inputs.csv`.
3. Compare the original model with itself, then click **Analyze batch**.
4. Check that it says **8 inputs analyzed**. Output changes should be zero.
5. For this pinned checkpoint and these eight profiles, expect **1 ReLU neuron zero on
   every input**, in the second hidden layer (`op03`). Select that layer, filter inactive
   neurons, and click **Inspect in 3D**.

This is an observation about eight examples. It is a reason to investigate that neuron's
weighted sum and input coverage, not sufficient evidence to remove it. Use a representative
validation batch for an engineering diagnosis.

## 6. Make a controlled edit and compare revisions

This deliberate change demonstrates that your edits affect real computation:

1. Open **Weights**, choose **op05.bias** (the two output biases), and select index **1**,
   the second cell.
2. In the Inspector, change its value from approximately **-0.30738723** to
   **0.69261277**, then stage the edit. This adds approximately 1 to output score 1.
3. Name the revision **Output class 1 bias +1** and click **Save revision**.
4. Reopen **Diagnostics**. Load the same `titanic-inputs.csv` again if necessary.
5. Compare the current revision with **Original model**, then click **Analyze batch**.

Expected results, allowing tiny float32 rounding differences:

- First output score unchanged; second output score increases by about **1** for every row.
- Mean absolute output-coordinate change about **0.5**; largest change about **1**.
- **4 of 8** predicted classes flip. This does not establish whether those changes are correct.
- Hidden-layer activations and their inactivity count stay unchanged because the edit is
  after those layers.

Use **Inspect in 3D** on a changed case, then restore **Original model** from revision
history when finished. This is a controlled experiment, not a recommended accuracy fix.

## 7. Try passenger inputs of your own

Copy `examples/titanic-passengers.csv` to `.local\titanic-demo\my-passengers.csv` and edit
the rows. Keep the header and provide 2-256 rows. Use `male`/`female` for Sex, `C`/`Q`/`S`
for Embarked, and quote names containing commas. Blank Age/Fare/Embarked use the source's
defaults. Names such as `Example, Mr. Alex` supply a title used in feature engineering.

Run the converter with your CSV:

```powershell
.\.venv\Scripts\python.exe .\examples\export_titanic_checkpoint.py .\.local\titanic-demo\model.safetensors .\.local\titanic-demo\preprocessing.json .\.local\titanic-demo\my-titanic.zip --passengers .\.local\titanic-demo\my-passengers.csv
```

Use `my-titanic-inputs.csv` for Diagnostics on your existing imported model. To run one
of its cases in 3D, click **Inspect in 3D** in the diagnostics results, or paste one complete
numeric row into **Input vector · 12 values** and click **Apply input**. You do not need
to import another model just to change the inputs. Re-importing `my-titanic.zip` would
instead create a separate workspace model using the original downloaded weights.

The 12 features, in order, are `Pclass, Sex, Age, SibSp, Parch, Fare, Title, FamilySize,
IsAlone, Embarked_C, Embarked_Q, Embarked_S`. The helper derives the family/title/port
features, then applies `(value - training_mean) / training_std` to every feature. The
input boxes expect these standardized numbers. The raw passenger CSV, including its
text and header, cannot go directly into Diagnostics.

## If something fails

- **Windows file picker stays on "Working on it...":** the native dialog may be
  waiting on Explorer/OneDrive while browsing this cloud-synced project. Press Esc
  to cancel if it responds. Copy only the ZIP and input CSV to a small local folder
  outside OneDrive, then paste the full ZIP path into the picker's **File name** box
  instead of browsing the project. On this installation, copies are available at
  `C:\Users\ayush\AppData\Local\neuroLabs\uploads\titanic\titanic.zip` and
  `C:\Users\ayush\AppData\Local\neuroLabs\uploads\titanic\titanic-inputs.csv`.
  If the dialog cannot be dismissed, ending Chrome through Task Manager is a last
  resort and may lose unsaved browser work. An unselected file has not reached
  neuroLabs' package validation yet.
- **Python path not found:** open the terminal in the project folder and check that
  `.venv` exists. On a fresh installation run the README setup first.
- **No module named neurolabs:** use the project's `.venv\Scripts\python.exe`, not a
  different system Python. Run `uv sync --locked` if this environment is incomplete.
- **Download/conversion error:** confirm both source files downloaded successfully;
  an HTML error page renamed to `.safetensors` is not a checkpoint.
- **Invalid package:** choose the converter's `titanic.zip`, rather than the source
  checkpoint or a manually zipped Hugging Face repository.
- **Wrong number of input values:** use one numeric row of the generated inputs CSV,
  with all 12 columns and no labels/header. Do not standardize it again.
- **Results differ after experimenting:** restore the imported original, discard any
  unsaved draft, and reload the eight-profile CSV before checking the reference values.

The converter is specific to this checkpoint. Another supported MLP can be packaged too,
but its architecture, parameter mapping and preprocessing must be checked separately.
