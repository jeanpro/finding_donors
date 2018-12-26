# Udacity's Finding Donors project

Udacity's Data Scientist Nanodegree's first project. Predicting donors for a NGO that found out that people with income >$50K are more likely to give donations. We are using supervised learning over 1994 Census data.

## Methodology

1) Exploring the data against the label `income`

2) Feature Exploration

  * **age**: continuous. 
  * **workclass**: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked. 
  * **education**: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool. 
  * **education-num**: continuous. 
  * **marital-status**: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse. 
  * **occupation**: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces. 
  * **relationship**: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried. 
  * **race**: Black, White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other. 
  * **sex**: Female, Male. 
  * **capital-gain**: continuous. 
  * **capital-loss**: continuous. 
  * **hours-per-week**: continuous. 
  * **native-country**: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.

3) Transforming Skewed Continuous Features

4) Normalizing Numerical Features

5) Data Processing: one-hot-encoding

6) Shuffle and Split Data

7) Evaluating Model Performance using a naive predictor

8) Analyzing the data and testing against multiple supervised learning algorithms including ensemble methods:

* Gaussian Naive Bayes
* Random Forest
* AdaBoosting Classifer
* SVM

9) Optimizing the model hyper parameters using `GridSeachCV`

10) Feature selection


## Libraries:

* Numpy
* Pandas
* Sci-kit Learn

## Results:

After choosing the best model and performing an optimization over hyper parameters using `GridSearchCV` we got a model with the below results:

|     Metric     | Unoptimized Model | Optimized Model |
| :------------: | :---------------: | :-------------: | 
| Accuracy Score |      0.8576       |    0.8677       |
| F-score        |      0.7246       |    0.7452       |

**Model**: Decision Tree (weak classifier) on AdaBoosting ensemble method with the below parameters:

* `n_estimators`: 300
* `learning_rate`: 1.5
* `algorithm`: SAMME.R
