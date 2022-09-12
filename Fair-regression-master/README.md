Implementation of "A Convex Framework for Fair Regression"

# Summary
A rich family of fairness metrics for regression models that take the form of a fairness regularizer is applied to the standard loss functions for linear and logistic regression. The family of fairness metrics covers the spectrum from group fairness to individual fairness along with intermediate fairness notion. By varying the weight on the fairness regularizer, the efficient frontier of the accuracy-fairness tradeoff is obtained and the severity of this trade-off is computed via a numerical quantity called Price of Fairness (PoF).

# Requirements
python_version = "3.6" <br />
[packages] <br />
numpy==1.18.2 <br />
pandas==1.0.3 <br />
cvxpy==1.0.31 <br />
sklearn==0.22.2.post1 <br />
matplotlib==3.2.1 <br />
xlrd==1.2.0 <br />

# Results
Paper(Left) vs Our implementation(Right)

<img src="https://github.com/ManishChandra12/Fair-regression/blob/master/output_paper/community.png" width="404" height="324"> <img src="https://github.com/ManishChandra12/Fair-regression/blob/master/output/community.png" width="454" height="350">
<img src="https://github.com/ManishChandra12/Fair-regression/blob/master/output_paper/compas.png" width="404" height="324"> <img src="https://github.com/ManishChandra12/Fair-regression/blob/master/output/compas.png" width="454" height="350">
<img src="https://github.com/ManishChandra12/Fair-regression/blob/master/output_paper/adult.png" width="404" height="324"> <img src="https://github.com/ManishChandra12/Fair-regression/blob/master/output/adult.png" width="454" height="350">
<img src="https://github.com/ManishChandra12/Fair-regression/blob/master/output_paper/default.png" width="404" height="324"> <img src="https://github.com/ManishChandra12/Fair-regression/blob/master/output/default.png" width="454" height="350">

# Remaining
1. The ```Law School``` dataset that we managed to have access to, is a much concise version of what the authors used. Therefore, the result we obtained for this concise version of dataset is different from the author's and hence, isn't shown in the above results.
2. Because of the unavailability of the ```Sentencing``` dataset, experimentation with it couldn't be performed.

# NOTE
1. The paper doesn't use all the cross-pairs, but rather, random sampling is done for choosing the cross-pairs. In our experiments, we found that some datasets are quite sensitive to which random pairs are chosen and hence the slight difference in the paper's and our results.
2. Experimenting with various values of lambdas to get smoother curves couldn't be performed because for the large datasets, the time to run the experiments on our local machines was quite large (~7-8 hours with 7 cores).
3. For ```Communities and Crime``` dataset, the paper says that two groups are formed based on the percentage of Black people, White people, Indians, Asians and Hispanics in a community. However, per capita incomes for these groups are considered for forming groups.

## Team Members
1. Sharik A (19CS60D04)
2. Manish Chandra (19CS60A01)
3. Anju Punuru (19CS60R07)
4. Kunal Devanand Zodape (19CS60R13)
5. Anirban Saha (19CS60R50)
6. Hasmita Kurre (19CS60R67)


## Project Setup
1. Clone the repo
2. Install pipenv
```
pip install pipenv
```
3. cd to the project directory
4. Create the virtual environment
```
pipenv install --skip-lock
```
5. Activate the virtual environment
```
pipenv shell
```

### Preprocessing
```
python3 -m src.preprocess_compas
```
Replace ```preprocess_compas``` with ```preprocess_adult```, ```preprocess_default``` or ```preprocess_community``` for 'Adult', 'Default' and 'Communities and Crime' datasets respectively.

### Generating Accuracy vs Fairness
```
python3 -m src.frontier --dataset=compas --proc=<number of cores to use>
```
Replace ```compas``` with ```adult```, ```lawschool```, ```default``` or ```community``` for 'Adult', 'Default' and 'Communities and Crime' datasets respectively.

### Generating Price of Fairness Bar Graph
```
python3 -m src.pof --dataset=compas --proc=<number of cores to use>
```
Replace ```compas``` with ```adult```, ```lawschool```, ```default``` or ```community``` for 'Adult', 'Default' and 'Communities and Crime' datasets respectively.

The final plots will be saved inside output/
