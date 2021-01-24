# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
In this project we use the UCI Bank Marketing dataset, related with direct marketing campaigns of a Portuguese banking institution. The data originated in the paper: S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, Elsevier, 62:22-31, June 2014.

According to UCI documentation (https://archive.ics.uci.edu/ml/datasets/Bank+Marketing), the bank used phone calls for its marketing campaign. The desired outcome for the bank was for the client to subscribe a term deposit. In this project we use two different approaches to predict the outcome variable (y), indicating wether the client will subscribe the term deposit or not. The table below shows the first five reccords.

<p><img src="./auxiliary/dataset head.png" alt="First five records of the dataset" title="Dataset Head" />
Note: in this table is missing the "duration" which appeared as elypses on the ds.take(5).to_pandas_dataframe() command</p>

The best performing model was a voting ensamble from the AutoML run, achieving an accuracy of 0.9197. However, I am hesitant to accept this result, since the AutoML also alerted us about class balance issues, which might cause a false high accuracy. Maybe F1 would have been a better metric in this case.

## Scikit-learn Pipeline
The first experiment consist on using hyperdrive to search for optimal parameters for a logistic regression. The first step is creating a compute cluster to run it. Following lab's specifications I chose a "Standard_D2_V2" virtual machine with a maximum of 4 nodes.

I then specified a parameter sampler and an early-stop policy (the order is not important). For the parameter sampler I used random sampling on the inverse of the regularization strength (C) and the maximum number of iterations taken for the solver to converge (max_iter) as suggested in the starter code. All other parameters were left as default (<a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html">see documentation</a>).

For C I specified a uniform distribution between 0.01 and 2. Regularization is used to prevent overfitting. It penalizes high coefficients by adding a regularization term (L2 by default in sklearn) multiplied by a parameter lambda to the loss function, which we seek to minimize. This C is the inverse of lambda, therefore the smaller the C, the stronger the penalty. In all my tests, the top performing models had C values between 0.7 and 1.4.

For the maximum number of iterations I specified a choice distribution in the set (50,75,100,150,200). This parameter is not important as long as convergence is reached within the number of iterations explicited. In all my tests convergence was reached, so sampling this parameter was not necessary given that the default value is 100 and even with a value of 50 the solver converged every time. Theoretically this parameter could be use to avoid waisting time in models that will not converge, no matter how long we let the solver run.

Regarding the early stopping policy I used a BanditPolicy with a slack_factor of 0.2, an evaluation interval of 1, and a delay evaluation of 5. This means that any training run with an accuracy 20% lower than the maximum reported at that interval is terminated without completing it. The delay evaluation parameter indicates that the evaluation will start after 5 intervals to avoid premature termination. This policy is used to avoid waisting resources on a combination of hyperparameters that will not result in a useful model at the end. In all my tests this policy was never applied.

The next step is to create an estimator using the train.py script. This script loads data from the internet, cleans them using a custom function, and splits them into train and test using an sklearn function. It is also responsible of parsing the arguments selected by the sampler, defining the model to be trained and keep track of the chosen metric (acccuracy). 

I then established the configurations for the training runs using HyperDriveConfig. In it I specified the estimator previously defined, added the sampling and the early termination policy, specified the primary metric (accuracy in this case) and its goal (maximization). Also, I specified the maximum total runs to be 100 and a maximum concurrent number of runs in 4. In retrospective, maybe 100 runs was too much, while the 4 concurrent runs are linked to the compute cluster I created for the experiment.

With all the sets ready, I submitted the experiment to be executed and called for RunDetails(exp).show() to be able to monitor it within the notebook.

Finally, once the run concluded, I retrieved the best run, saved it to the output folder and registered the model.

Screenshots of all these steps are available in <a href="./auxiliary/hyperdrive.pptx">hyperdrive</a> file in the auxiliary folder. Also availabel is the <a href="./auxiliary/pruebasHyper.ipynb">notebook</a> used to create those screenshots.

## AutoML
For the AutoML part, I first import, clean and split the data inside the notebook using the same functions as before.Then I configured the run with a 30 minutes time limit and 5 cross validations. No further hyperparameter specifications were needed. At the end of the run, I retrieved the best model, saved it and registered it. Screenshots of all these steps are available in <a href="./auxiliary/automl.pptx">hyperdrive</a> file in the auxiliary folder. Also availabel is the <a href="./auxiliary/pruebaAutoML.ipynb">notebook</a> used to create those screenshots.

As I mentioned before, AutoML detected a problem with class imbalance and sugested stop training an correcting it. It did not detect any problem with missing values or high cardinality. It then proceeded to train different models. The best model was a Voting Ensamble with accuracy = 0.9197. Once finished training, AutoML proceeded with the explanation of the best model. In the above mentioned PowerPoint are screenshots of these results.

The most influential feature acording to AutoML is duration. The documentation says "duration: last contact duration, in seconds (numeric)" and further explicits that this is a feature unknown at the time of prediction. Thus, this model might be a good one for explaining why things happened the way they did, but it cannot anticipate future outcomes. This variable should have been removed from the dataset at the time of cleaning it.

Another factor to keep in mind is the <a href="./auxiliary/confusion matrix.png">confusion matrix</a>. It shows that, while the relaation of true negatives to false negatives is 18839 to 805, the relation of true positives to false positives is 1404 to 1028. This is the effect of the class imbalance problem and one of the reasons the accuracy was so high, the models simply learned that the most likely outcome was false.

## Pipeline comparison
The two experiments I run gave quite similar accuracies, 91% for the logistic regression in the HyperDrive and 92% for the voting ensable in the AutoML. However, once again these metrics might be misleading. If these results were correct, unless I am planning to deploy this models to classify a huge ammount of records, a 1% difference doesn´t seem relevant. 

That said, what is relevant is the ammount of work saved by the AutoML experiment compared to setting the HyperDrive pipeline. Another point is the algorithm chosen by AutoML. A voting ensamble combines multiple instances of ML algorithms. Each algorithm works independently and their results are combined using a majority or weighted vote (in classification problems). It is to be expected that a combination of different methods yields a better solution to a problem and thus improves predictive power of the model.

## Future work
For improving this project I would:
1) Remove the "duration" feature if the model is to be used for prediction. First, this feature is unknown at the time of prediction. Second, even if I knew it, it does not depend on anything the decision-makers can control, thus I don´t see the use of it (except for academic purposes). One might say that age, for example, is not under the decision-makers control either; however, they can target their advertising campaign to specific age ranges, while they cannot force potential customers to stay longer on the phone.
2) As mentioned before, class imbalance needs to be addressed. Either we change the primary metrics to F1, or we use oversampling methods on the positive label examples to reduce the bias caused by the imbalance. One alternative is to use the SMOTE or ADASYN sampling methods in the <a href="https://imbalanced-learn.readthedocs.io/en/stable/">imbalanced-learn</a> library.
3) Not relevant to the outcome of the model but for better use of resources, I would remove the sampling on maximum number of iterations for the reasons mentioned in the Scikit-learn Pipeline section.

## Final remarks
I recorded the entire notebook running a shorter version of the experiment that can be reached <a href="https://1drv.ms/v/s!AmnIa5DZWnqEwXXjV20jwhHdQzl7?e=HHcYNr">here</a> in OneDrive, and the <a href="./auxiliary/for-video.ipynb">notebook</a> used to create it is in the auxiliary folder.

## Proof of cluster clean up
I did delete the cluster at the end of the notebook, nevertheless I am attaching here prove of that.
<p><img src="./auxiliary/deleting cluster.png" alt="Screencapture of cluster cleanup" title="Cluster cleanup" /></p>
