# Classification-and-Regression-in-Machine-Learning

CSE574 Introduction to Machine Learning
Programming Assignment 1
Classication and Regression
Due Date: March 14th 2018
Maximum Score: 100
Note A zipped le containing skeleton Python script les and data is provided. Note that for each problem,
you need to write code in the specied function withing the Python script le. Do not use any Python
libraries/toolboxes, built-in functions, or external tools/libraries that directly perform classi-
cation, regression, function tting, or ridge regression. Using any external code will result in 0
points for the corresponding problem.
Evaluation We will evaluate your code by executing script.py le, which will internally call the problem
specic functions. Also submit an assignment report (pdf le) summarizing your ndings. In the problem
statements below, the portions under REPORT heading need to be discussed in the assignment report.
Data Sets Two data sets are provided:
1. A 2D sample data set in the le \sample.pickle" along with the class assignment.
2. A medical data set is provided in the le \diabetes.pickle" along with the target assignment. The input
variables correspond to measurements (physical, physiological, and blood related) for a given patient
and the target variable corresponds to the level of diabetic condition in the patient. It contains:
 xtrain (242  64) and ytrain (242  1) for training.
 xtest (200  64) and ytest (200  1) for testing.
Submission You are required to submit a single le called proj1.zip using UBLearns.
File proj1.zip must contain 2 les: report.pdf and script.py.
 Submit your report in a pdf format. Please indicate the team members on the top of the report.
 The code le should contain all implemented functions. Please do not change the name of the le.
Using UBLearns Submission: In the groups page of the UBLearns website you will see groups
called \CSE574 Project Group x". Please choose any available group number for your group and join
the group. All project group members must join the same group. Please do not join any other group on
UBLearns that you are not part of. You should submit one solution per group through the groups page.
Problem 1: Experiment with Gaussian Discriminators (10 code +
10 report = 20 points)
Implement Linear Discriminant Analysis (LDA) and Quadratic Discriminant Analysis (QDA). Im-
plement two functions in Python: ldaLearn and qdaLearn which take a training data set (a feature matrix
1
and labels) and return the means and covariance matrix (or matrices). Implement two functions ldaTest
and qdaTest which return the true labels for a given test data set and the accuracy using the true labels for
the test data. The format of arguments and the outputs is provided in the base code.
REPORT 1.
Train both methods using the sample training data (sample train). Report the accuracy of LDA and QDA
on the provided test data set (sample test). Also, plot the discriminating boundary for linear and quadratic
discriminators. The code to plot the boundaries is already provided in the base code. Explain why there is
a dierence in the two boundaries.
Problem 2: Experiment with Linear Regression (5 code + 5 report
= 10 Points)
Implement ordinary least squares method to estimate regression parameters by minimizing the squared loss.
J(w) =
1
2
XN
i=1
(yi 􀀀 w>xi)2 (1)
In matrix-vector notation, the loss function can be written as:
J(w) =
1
2
(y 􀀀 Xw)>(y 􀀀 Xw) (2)
where X is the input data matrix, y is the target vector, and w is the weight vector for regression.
Note that this is same as maximizing the log-likelihood in the Bayesian setting. You need to implement
the function learnOLERegression. Also implement the function testOLERegression to apply the learnt
weights for prediction on both training and testing data and to calculate the mean squared error (MSE):
MSE =
1
N
XN
i=1
(yi 􀀀 w>xi)2 (3)
REPORT 2.
Calculate and report the MSE for training and test data for two cases: rst, without using an intercept (or
bias) term, and second with using an intercept. Which one is better?
Problem 3: Experiment with Ridge Regression (10 code + 10 re-
port = 20 Points)
Implement parameter estimation for ridge regression by minimizing the regularized squared loss as follows:
J(w) =
1
2
XN
i=1
(yi 􀀀 w>xi)2 +
1
2
w>w (4)
In matrix-vector notation, the squared loss can be written as:
J(w) =
1
2
(y 􀀀 Xw)>(y 􀀀 Xw) +
1
2
w>w (5)
You need to implement it in the function learnRidgeRegression.
REPORT 3.
Calculate and report the MSE for training and test data using ridge regression parameters using the the
testOLERegression function that you implemented in Problem 2. Use data with intercept. Plot the errors
on train and test data for dierent values of . Vary  from 0 (no regularization) to 1 in steps of 0.01.
Compare the relative magnitudes of weights learnt using OLE (Problem 2) and weights learnt using ridge
regression. Compare the two approaches in terms of errors on train and test data. What is the optimal value
for  and why?
2
Problem 4: Using Gradient Descent for Ridge Regression Learning
(20 code + 5 report = 25 Points)
As discussed in class, regression parameters can be calculated directly using analytical expressions (as in
Problem 2 and 3). To avoid computation of (X>X)􀀀1, another option is to use gradient descent to minimize
the loss function (or to maximize the log-likelihood) function. In this problem, you have to implement the
gradient descent procedure for estimating the weights w.
You need to use the minimize function (from the scipy library). You need to implement a function
regressionObjVal to compute the regularized squared error (See (4)) and its gradient with respect to w.
In the main script, this objective function will be used within the minimizer.
REPORT 4.
Plot the errors on train and test data obtained by using the gradient descent based learning by varying the
regularization parameter . Compare with the results obtained in Problem 3.
Problem 5: Non-linear Regression (10 code + 5 report = 15 Points)
In this problem we will investigate the impact of using higher order polynomials for the input features. For
this problem use the third variable as the only input variable:
x t r a i n = x t r a i n [ : , 2 ]
x t e s t = x t e s t [ : , 2 ]
Implement the function mapNonLinear which converts a single attribute x into a vector of p attributes,
1; x; x2; : : : ; xp.
REPORT 5.
Using the  = 0 and the optimal value of  found in Problem 3, train ridge regression weights using the
non-linear mapping of the data. Vary p from 0 to 6. Note that p = 0 means using a horizontal line as the
regression line, p = 1 is the same as linear ridge regression. Compute the errors on train and test data.
Compare the results for both values of . What is the optimal value of p in terms of test error in each
setting? Plot the curve for the optimal value of p for both values of  and compare.
Problem 6: Interpreting Results (0 code + 10 report = 10 points)
Using the results obtained for previous 4 problems, make nal recommendations for anyone using regression
for predicting diabetes level using the input features.
REPORT 6.
Compare the various approaches in terms of training and testing error. What metric should be used to
choose the best setting?
3
