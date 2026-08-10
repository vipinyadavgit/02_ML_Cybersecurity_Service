1. Understand the problem statement in detail. On the basis of this, you will able to identify which are the input variables and what is the output variable.
2. Import the libraries
3. Load the dataset
4. EDA (these arent the fixed series of steps to be followed, it can vary, you can change te series of steps depending on the problem statement and dataset)
	a. Basic dataset check:
		i. df.shape - dimension of rows and columns
		ii. df.info() - it will print the information of the dataset i.e. datatypes, column names, number of not null values
		iii. df.describe() - Generate descriptive statistics.
		iv. print df.head() - gives first 5 rows / df.tail() - gives last 5 rows.
	b. Delete all the columns that have all the values in it as unique or all the values in it as same.- 
		because it doesnt add any value in the data science- drop it using df.drop("column name")
	c. Handling null values:
		i. Check if the are null values - df.isnull().sum()
		ii. Ways to handle null values:
	    		- Delete the rows that have null value using dropna() (When there are small number of missing rows)
      			- Delete columns using df.drop() when there are more than 50% of the values in a particular column as null
     			- If the column has categorical datatype: Impute/Replace with mode
			- if the column has continuous numerical dataype: Impute/replace with mean
			- if the column has discrete numerical dataype: Impute/replace with median
	d. Handling duplicates:
		i. Check the duplicate rows in dataset - df.duplicated().sum()
		ii. Ways to handle duplicate values:
			- Remove duplicate using df.drop_duplicates()
	e. Split the dataset into categorical and numerical columns
		i. cat = df.select_dtypes(include = "object").columns
		ii. num = df.select_dtypes(exclude= "object").columns
	f. Handling error values: Error values can be anything (?, @, etc)
		i. Replace the error value with null using df.replace("?", np.NaN). 
		ii. Once all the errors are converted into Null, apply step c.
	g. Handling Outliers:
		i. Detect outliers using IQR method:
			- Find Q1 = df.quantile(0.25), Q3= df.quantile(0.75)
			- Find IQR =  Q3 - Q1
			- LB = Q1 - 1.5*IQR, UB = Q3 + 1.5*IQR
			- Remove all the values less than LB and greater than UB
	h. Categorical to numerical conversion:
		i. Label encoding: When we have a large number of nominal columns.	using sklearn.preprocessing.LabelEncoder()
		   Disadv: It can be misleading to algorithms
		ii. .map() -Best when there are only a few categories and you want to manually map the encodings
		   Disadv: Not suitable when we have many categories in the dataset
		iii. Ordinal encoding: When categories have natural order of ranking
		   Disadv: it has be done manually
		iv. One hot encoding: Each category becomes a seperate column
	i. Univariate analysis (analysis done on single variable)and bivariate analysis (analysis done on 2 variables)
		i. Univariate- Plot histogram of every/major columns, Check for the unique values in each column
		ii. Bivariate - Sales vs (TV, Radio, Newspaper), correlation analysis, pair plot analyis
5. ML model building:
	i. Import the models whichever you will be building
	ii. Divide the dataset into X (input), y (output) variables. X will have all the input data, y will have all the output data points
	iii. Further divide into X_train, X_test, y_train, y_test
	iv. Train our model using X_train, y_train
	v. Pass X_test as an input to the newly created model, predict y_pred
	vi. Model evaluation: Compare y_pred with y_test and find the following metrics:
			a. Regression: MAE, MSE, R2-score, RMSE
			b. Classification: confusion matrix, accuracy, precision, recall, F1-Score

>	================================================================================================
###		Choosing right ML Algorithm.
----------------------------------------------------------------------------------------------------
	
Choosing the right ML algorithm:
1. There is no universally best algo.
2. the model depends on business problem, requirements, computational constraints, validation performance. 

	Steps:
	a. Understanding business problem
	b. type of ML Problem to apply
	c. Understand the dataset
	d. Choose different algos to apply
	e. train multiple models
	f. Hyperpareter tuning
	g. cross validation
	h. Compare the model metrices
	i. Choose the best model






































