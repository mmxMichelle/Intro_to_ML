Before running the code, one should check if requirements are met, if not, in your environment, run

、、、
pip install -r requirements.txt
、、、

To run the full pipeline, under the root directory, run

、、、
python data_exploration.py
python nn.py
python svmrbf.py
、、、

The first command will visualize the data structure with feature distributions, correlation heatmaps, PCAs, pairplots,
and 2D feature plots. The command will not show those plots since the number of plots is large. It will save the plots
under /results/<n>D directory.
The second command will train and test a 3 layer MLP on kryptonite data and save the loss curves and model performances
of each dimension under /results/<n>D directory. It will also create a summary_all.csv that summarizes results for each
dimension. The estimated running time is 1 hour and 20 minutes.
The third command will train and test an SVM RBF model on kryptonite data and save the model performances of each
dimension under /results/<n>D directory. It will also create a summary_all_SVMRBF.csv that summarizes results for each
dimension. The estimated running time is above 48 hours.

For simpler data illustration, one can simply check the data_exploration_notebook.ipynb, which draws each data
exploration plot for 10 dimensional data cell by cell.
For simpler MLP results reproduction, one can run the following command under the root directory, which would use the
best MLP model for each dimension to predict kryptonite data labels.
However, due to time limit, we are unable to save SVM RBF best models for reproduction, but running the svmrbf.py should
give similar results to those provided in the essay.

、、、
python nn_reproduce.py
、、、