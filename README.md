Stock Price Prediction Using Machine Learning:

Overview:
This project demonstrates the application of machine learning to predict stock prices, using the Tesla stock dataset. The goal is to predict whether buying a stock on a given day will be profitable based on historical stock data. Several machine learning models are trained and evaluated to identify the best-performing model.

Dataset:
The dataset used for this project is Tesla stock price data, including OHLC (Open, High, Low, Close) prices and trading volume from January 1, 2010, to December 31, 2017. This data is available in CSV format.

Features:
Open: Opening price of the stock for the day.
High: Highest price of the stock during the day.
Low: Lowest price of the stock during the day.
Close: Closing price of the stock for the day.
Volume: Total number of shares traded.
is_quarter_end: Indicates whether the date is the end of a financial quarter.
open-close: Difference between the opening and closing prices.
low-high: Difference between the lowest and highest prices.
Target:
target: Binary target feature (1 for profitable, 0 for not profitable) indicating whether the closing price for the next day will be higher than the current day.

Project Structure:

├── data/
│   └── Tesla.csv                # Dataset file
├── src/
│   └── stock_price_prediction.py # Main script to run the model
├── notebooks/
│   └── EDA.ipynb                # Jupyter notebook with Exploratory Data Analysis (EDA)
├── README.md                    # Project documentation (this file)
└── requirements.txt             # Required Python libraries

Libraries Used
Pandas: For data manipulation and analysis.
NumPy: For numerical operations.
Matplotlib/Seaborn: For data visualization.
Scikit-learn: For data preprocessing, model building, and evaluation.
XGBoost: For the extreme gradient boosting machine learning algorithm.
Exploratory Data Analysis (EDA)
EDA is performed to explore stock price trends and identify useful patterns:

Visualized the closing price trend of Tesla stocks.
Checked for missing values and redundant data.
Explored the distribution and outliers of continuous variables (OHLC and Volume).
Added new features derived from existing ones (such as quarter-end indicator, open-close difference, etc.).
Feature Engineering
Created new features like:
open-close: Difference between opening and closing price.
low-high: Difference between lowest and highest price.
is_quarter_end: Feature indicating if the day was the end of a financial quarter.
The target column was created to signal whether the stock price would increase the next day.
Model Training

Three machine learning models were trained on the dataset:
Logistic Regression
Support Vector Machine (SVM)
XGBoost

Model Evaluation:
The models were evaluated based on the ROC-AUC score on both the training and validation sets. The XGBoost model performed best in terms of training accuracy, but overfitted the data. Logistic Regression was more balanced between training and validation performance.

Performance
Model	Training ROC-AUC	Validation ROC-AUC
Logistic Regression	0.52	0.54
Support Vector Machine	0.47	0.45
XGBoost	0.96	0.57
The XGBoost model had the highest training accuracy but showed signs of overfitting.

Usage
Prerequisites
Ensure you have Python 3.x installed and install the required packages:


pip install -r requirements.txt
Running the Model
Clone this repository:
 [code]
git clone https://github.com/username/stock-price-prediction.git
cd stock-price-prediction
Place the dataset (Tesla.csv) in the data/ directory.
Run the main script:
 [code]
python src/stock_price_prediction.py
Output
The script will output the ROC-AUC score and display a confusion matrix for the selected model (Logistic Regression, SVM, or XGBoost).

Conclusion:
This project applies machine learning to predict stock prices using Tesla stock data. While XGBoost performed the best on the training set, logistic regression provided more balanced performance across both training and validation sets.

Improvements:
Collect more recent data to improve model training and accuracy.
Tune hyperparameters for better performance and reduce overfitting.
Experiment with more advanced machine learning or deep learning models.
