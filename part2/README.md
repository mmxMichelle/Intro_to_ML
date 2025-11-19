# README

# Neural Network Coursework Project

This repository contains the full implementation for a two-part
neural-network coursework assignment:

1.  **Part 1 --- From-scratch NumPy Neural Network Library**\
    Includes a custom multilayer network, activation functions, loss
    layers, gradient-descent training loop, preprocessing tools, and
    example usage on the *Iris* dataset.

2.  **Part 2 --- California Housing Price Regression (PyTorch)**\
    A full machine-learning pipeline including preprocessing, training,
    early stopping, hyperparameter search, evaluation, saving/loading a
    trained model, and data-exploration utilities.

Both parts run entirely from the project root directory.

------------------------------------------------------------------------

## Project Structure

    project_root/
    │
    ├── part1_nn_lib.py
    ├── part2_house_value_regression.py
    ├── part2_model.pickle        # Saved model (generated after training)
    ├── data_exploration.py
    ├── requirements.txt
    │
    ├── iris.dat                  # Dataset for Part 1
    └── src/
         ├── housing.csv          # California housing dataset
         └── median_house_value_distribution.png

------------------------------------------------------------------------

# 1. Installation

You can set up the environment with a single command:

``` bash
pip install -r requirements.txt
```

This installs all dependencies, including:

-   NumPy, Pandas, SciPy\
-   Scikit-learn\
-   PyTorch\
-   Matplotlib

Once the environment is ready, you can run each assignment part
independently.

------------------------------------------------------------------------

# 2. Running Part 1 --- NumPy Neural Network (Iris Dataset)

The Part 1 implementation is fully self-contained inside
**`part1_nn_lib.py`**, including an `example_main()` function that:

-   Loads and shuffles the *Iris* dataset
-   Preprocesses inputs using a min--max normalizer
-   Builds a multilayer network
-   Trains it using mean-squared error
-   Prints the validation loss

### ▶ Run Part 1

Simply run:

``` bash
python part1_nn_lib.py
```

The script will automatically:

-   Read `iris.dat` from the project root
-   Train the neural network
-   Display the validation loss at the end


------------------------------------------------------------------------

# 3. Running Part 2 --- Housing Price Regression (PyTorch)

Part 2 provides a full deep-learning pipeline using the California
housing dataset located in `src/housing.csv`.

Running the script performs:

1.  Loading and splitting the data
2.  Hyperparameter search (hidden layers + learning rate)
3.  Training the best model with early stopping
4.  Evaluating on a test set (RMSE, MAE, R²)
5.  Saving the trained model into `part2_model.pickle`

### ▶ Run Part 2

Use:

``` bash
python part2_house_value_regression.py
```

The script automatically:

-   Reads `src/housing.csv`
-   Performs 5-fold cross-validated hyperparameter tuning
-   Trains the best model
-   Prints performance metrics
-   Saves the final model into `part2_model.pickle`

Because hyperparameter search can take time, you'll see progress printed
clearly as the training proceeds.

------------------------------------------------------------------------

# 4. Loading the Saved Regressor (Part 2)

After training completes, you can load the saved model in any Python
session:

``` python
from part2_house_value_regression import load_regressor

model = load_regressor()
```

You can then make predictions:

``` python
import pandas as pd

sample = pd.DataFrame({...})  # your input features
predictions = model.predict(sample)
print(predictions)
```

Everything needed for preprocessing (normalization, binarizers, stored
min/max values) is already embedded inside the saved model.

------------------------------------------------------------------------

# 5. Running Data Exploration Script

To generate the histogram `median_house_value_distribution.png`, run:

``` bash
python data_exploration.py
```

This will:

-   Load `src/housing.csv`
-   Plot the distribution of `median_house_value`
-   Save the image directly into the `src/` folder


------------------------------------------------------------------------

