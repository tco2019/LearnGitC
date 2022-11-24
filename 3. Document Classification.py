#!/usr/bin/env python
# coding: utf-8

# ## Testing the Automated ML model
# 
# In this notebook you will be loading the best machine learning model trained using Automated ML, and use it to assign clusters to a series of new COVID-19 articles.

# ### Loading the latest model trained with Automated ML

# We'll start off by importing the necessary modules and checking the Azure ML SDK version

# In[ ]:


import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix

from azureml.core import Workspace, Experiment, Dataset, VERSION
from azureml.train.automl.run import AutoMLRun
from azureml.interpret import ExplanationClient

from azureml.widgets import RunDetails
from interpret_community.widget import ExplanationDashboard
from raiwidgets import ExplanationDashboard

print("Azure ML SDK Version: ", VERSION)


# We first need to load our workspace, and use that to retrieve our Automated ML experiment

# In[ ]:


# Load the workspace from a configuration file
ws = Workspace.from_config()

# Get a reference to our automated ml experiment
exp = Experiment(ws, 'COVID19_Classification')


# We now need to retrieve our latest Automated ML run, and its corresponding best model

# In[ ]:


# Retrieve a list of all the experiment's runs
runs = list(exp.get_runs()) 

# Pick the latest run
raw_run = runs[len(runs)-1]

# Convert it to an AutoMLRun object in order to retrieve its best model
automl_run = AutoMLRun(exp, raw_run.id)

# Retrieve the best run and its corresponding model
best_run, best_model = automl_run.get_output()


# ### Analyzing the metrics calculated while training the model

# After having retrieved the best performing run, let's examine some of its metrics using the SDK's `RunDetails` widget. Analyze the various metrics of your model, including *Precision-Recall*, *ROC*, *Lift Curve*, *Gain Curve*, and *Calibration Curve*.
# 
# Analyze the *Confusion Matrix* and see which clusters are correctly identified by the model, and which have a higher likelihood ofbeing misclassified.
# 
# Experiment with the *Feature Importance* and analyze the relative importance of the top K features.

# In[ ]:


RunDetails(best_run).show()


# ### Running the model on a new dataset
# 
# First we'll load the dataset we had previously prepared for testing, and convert it to a Pandas data frame.

# In[ ]:


# Retrieve the dataset from the workspace
test_ds = Dataset.get_by_name(ws, 'COVID19Articles_Test_Vectors')

# Convert it to a standard pandas data frame
test_df = test_ds.to_pandas_dataframe()

# Examine a sample of 5 documents
test_df.sample(5)


# Now we can use the *best_model* to assign clusters to the test documents

# In[ ]:


# Save the true values of the clusters
true_clusters = test_df['cluster']

# Keep all features except the label column
features_df = test_df.drop(columns=['cluster'])

# Predict the clusters for each document and display them
best_model.predict(features_df)


# We can compare the true clusters with the predicted ones by using a confusion matrix - notice the true positive values on the diagonal

# In[ ]:


plot_confusion_matrix(best_model, features_df, true_clusters)


# ### Interpreting and explaining the model
# 
# By default, Automated ML also explains the machine learning models it trains. We will download and examine the explanations for our *best_model*.

# In[ ]:


# Use an ExplanationClient for accesing the best run's model explanations
client = ExplanationClient.from_run(best_run)

# Download the engineered explanations in their raw form
engineered_explanations = client.download_model_explanation(raw=True)

# Retrieve the dataset used for training the model - it will be needed when visualizing the explanations
training_df = Dataset.get_by_name(ws, 'COVID19Articles_Train').to_pandas_dataframe()


# We will use an `ExplanationDashboard` to visualize the engineered explanations. For best results it needs to be presented with the same dataset used for training the model.
# 
# Analyze the *Aggregate Feature Importance* to identify the top predictive features. Select a feature and analyze how individual values of that feature impact prediction results. Switch to the *Individual Feature Importance & What-If* and explore the feature importance plots for individual points.

# In[ ]:


from raiwidgets import ExplanationDashboard
ExplanationDashboard(engineered_explanations, best_model, dataset=training_df.drop(columns='cluster'), true_y=training_df['cluster'])


# In[ ]:




