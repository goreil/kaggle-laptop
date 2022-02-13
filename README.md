# kaggle-laptop
<img src=Laptop.jpg alt=Picture of my Laptop width="300">

Since I bought a Laptop, I wanted to check if I got scammed or not by applying my newly aquired Machine Learning knowledge. Specifically given the Specs of my Laptop, how much should it have cost?

## Used Dataset
<a href=https://www.kaggle.com/ionaskel/laptop-prices> A Kaggle Dataset </a> about the Characteristics and Price for 1300 laptop models.

# Process
1. Data Preperation
2. Building the Model
3. Evaluation

## Data Preparation

**Target column:** `Price_euros`


**Feature columns:**
1. Drop Columns that contain NA-Values
2. Columns with categorical data, Cardinality < 10
3. Columns with numerical data

**My Laptop data:** For the given columns, I looked up the specs of my Laptop.

## Model Building
The tests have been done with a `XGBRegressor`-Model since that is the most accurate one I learned so far.
1. Create a scoring method `get_score` to evaluate quality of the model based on Mean Average Error (MAE)
2. Run a loop that determines the best parameters (`n_estimators`, `learning_rate`) for this model.

### Final Model
* `XGBRegressor`
* `n-estimators = 200`
* `learning_rate = 0.05`
* Mean Average Error: 285.74 €

## Evaluation
The Model evaluates my Laptop at Price: 1727.37 €, but I paid barely 800€. Since that is a huge price discrepancy, we can assume that probably the Model is bad.

### Future Work
1. Better Data Cleaning:
* Analyse if Dataset actually contains reasonable prices.
* Analyse which columns that might cause overfitting.
* Include Columns with NA-Values.
* Learn from other peoples Work.

2. Better Model Creation:
* Learn about more Models and how they work.
* Find a better fitting model.
* Adjust paramters for lower Mean Average Error.



