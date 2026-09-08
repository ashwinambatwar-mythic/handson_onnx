ML example

A small end to end example that trains a scikit learn model, converts it to the ONNX format and then runs predictions with the ONNX runtime. The model is a random forest classifier trained on the classic iris flower data set.


What is in this folder

train.py trains a random forest classifier on the iris data set and saves the fitted model with joblib to output/model.pkl using compression level 9.

convert.py loads output/model.pkl, describes the input as a float tensor with 4 columns and an unknown number of rows, converts the model with skl2onnx and writes output/model.onnx.

inference.py loads output/model.onnx with the ONNX runtime, prints the input and output names of the graph, feeds three sample flower measurements as float32 values and prints the predicted classes.

test_joblib.py is a side experiment that measures how long a heavy factorial computation takes when it is spread over a different number of cpu cores with joblib Parallel. It is not part of the ONNX flow.

requirements.txt lists the python packages needed, which are scikit learn, onnx, skl2onnx, joblib and onnxruntime.

output holds the generated artifacts, that is model.pkl from training and model.onnx from conversion.


Requirements

Python 3.10 or newer. A virtual environment is recommended.


Setup

1. Create a virtual environment with python3 and the venv module, for example python3 with the argument m venv and the folder name venv.

2. Activate the environment. On linux or mac use the source command on venv bin activate. On windows use venv Scripts activate.

3. Install the packages with pip install and the argument r pointing at requirements.txt.


How to run

Run every command from inside this folder, because all three scripts use paths that are relative to it.

1. Train the model with python train.py. This writes output/model.pkl.

2. Convert the model with python convert.py. This reads output/model.pkl and writes output/model.onnx.

3. Run inference with python inference.py. This prints the graph input name, the graph output name and one predicted class for each of the three sample rows.


About the input data

Each row given to the model has 4 values in this order, sepal length, sepal width, petal length and petal width, all in centimeters. The ONNX runtime expects float32 values, so the sample array in inference.py is cast before it is passed in. The predicted labels are 0 for setosa, 1 for versicolor and 2 for virginica.


Why use ONNX here

The pickle file produced by joblib can only be loaded back into a matching python environment with the same scikit learn version. The ONNX file stores the model as a portable computation graph instead, so it can be loaded by any ONNX runtime, in python or in another language, without scikit learn being installed. This makes the model easier to deploy and keeps the serving side small.


Notes

The training script does not set a seed for the classifier, so the trained forest and the resulting files can differ slightly between runs. The split of the data is fixed because train test split is called with a random state of 0.
