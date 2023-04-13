# Have I been scammed?
<p align="center">
    <img src=laptop.jpg width="300">
</p>

Since I bought a laptop, I wanted to check if I got scammed or not by applying my newly aquired machine learning knowledge. Given the specs of my laptop, how much should it have cost?

## Used Dataset
<a href=https://www.kaggle.com/ionaskel/laptop-prices> A Kaggle dataset </a> about the characteristics and price for 1300 laptop models.

# Process
1. Data preperation
2. Building the model
3. Evaluation

## Data Preparation
**Target column:** `Price_euros`

**Feature columns:**
1. Drop columns that contain NA-Values
2. Columns with categorical data, cardinality < 10
3. Columns with numerical data

**My laptop data:** For the given columns, I looked up the specs of my laptop.

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
The model estimates the price of my laptop to be 1727.37€, but I paid barely 800€. This could indicate that the model is bad.

### Future Work
1. Better Data Cleaning:
  * Analyse if dataset actually contains reasonable prices.
  * Analyse which columns that might cause overfitting.
  * Include columns with NA-values.
  * Learn from other peoples Work.

2. Better Model Creation:
  * Learn about more models and how they work.
  * Find a better fitting model.
  * Adjust paramters for lower Mean Average Error.
