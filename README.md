# forecast-demand-modeling


Achievements of project:
•	This project was implemented in Ural Federal University in Yekaterinburg, Russia.
•	Won “Best project within direction of the Summer University 2024” amongst a crowd of 400 students coming from 14 different countries.
•	This award was given (signed by) by the Minister of Science and Higher Education of the Russian Federation.


Introduction
Predicting future values from past data is an essential task in many fields, including economics, finance, inventory management, and more. Our project's main goal is to employ the ARIMA (AutoRegressive Integrated Moving Average) technique to create a reliable forecasting model. The objective is to make more informed decisions by precisely predicting future data points based on historical observations.
Objective
This project's main goal is to develop an ARIMA-based forecasting model that can produce accurate predictions from any time series of data. Preprocessing the data, dividing it into train and test datasets, choosing custom made ARIMA parameters, training the model, and predicting using test data are all part of this procedure. 


Methodology 
1. Gathering and Preparing Data:
•	Data Collection: Gathering historical information pertinent to the area of interest is the first phase in the process. This might include sales data, temperature observations, stock prices, etc.
•	Preparing data: To guarantee data quality, preprocessing is necessary. This covers processing missing values, eliminating anomalies, and bringing the data into conformity. In order to make the time series stationary, which is vital for ARIMA modeling, we additionally implemented differencing.
2. Development of the ARIMA Model:
•	Selecting a Model: The three parameters that define ARIMA models are p (the order of the autoregressive component), d (the degree of differencing), and q (it is the order of the moving average part). To ascertain these characteristics, we employed methods like the Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF) plots.
•	Model Training: Using the preprocessed data, we trained the ARIMA model after determining the parameters. In order to reduce the prediction error, the parameters were optimized throughout the training process by combining the model to the past data.
3. Assessment of the Model:
•	Performance Metrics: We utilized RMSE and MAE to assess how well our ARIMA model performed. These measures shed light on the typical size of the forecast mistakes. We obtained good results from our tests using a range of datasets.


Uniqueness of our Project:
Our project stands out due to its unique capability to automate machine learning operations, significantly streamlining the forecasting process. By leveraging automated machine learning (AutoML) techniques, the model autonomously handles tasks such as data preprocessing, feature selection, parameter tuning, and model evaluation, eliminating the need for extensive manual intervention. Once the user inputs a single dataset, the system processes it, trains the ARIMA model, and generates forecasts. These predictions are then displayed in real time on an interactive dashboard. The seamless integration of automation and real-time visualization ensures that users, regardless of their technical expertise, can easily access and benefit from advanced predictive analytics.
