'''from flask import Flask, request, jsonify, send_file, render_template
import pandas as pd
import os
from flask_cors import CORS
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Function to clean and split data
def clean_and_split_data(df, target_column):
    df.fillna(method='ffill', inplace=True)  # Fill missing values
    df.drop_duplicates(inplace=True)  # Drop duplicates
    train_size = int(len(df) * 0.8)
    train_data = df[:train_size]  # 80% of data for training
    test_data = df[train_size:].drop(columns=[target_column])  # 20% of data for testing, excluding target column
    return train_data, test_data

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/process', methods=['POST'])
def process_data():
    file = request.files['file']
    df = pd.read_csv(file)
    target_column = request.form['target_column']

    try:
        train_data, test_data = clean_and_split_data(df, target_column)
        
        # Save the cleaned data to CSV files
        cleaned_train_path = 'cleaned_train_data.csv'
        cleaned_test_path = 'cleaned_test_data.csv'
        train_data.to_csv(cleaned_train_path, index=False)
        test_data.to_csv(cleaned_test_path, index=False)

        # Return success message
        return jsonify({
            'message': 'Data processed successfully.',
            'cleaned_train_data': cleaned_train_path,
            'cleaned_test_data': cleaned_test_path
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    try:
        filepath = os.path.join(os.getcwd(), filename)
        return send_file(filepath, as_attachment=True)
    except FileNotFoundError:
        return jsonify({'error': 'File not found.'}), 404

@app.route('/train-arima', methods=['POST'])
def train_arima():
    try:
        # Load the cleaned training data
        train_data = pd.read_csv('cleaned_train_data.csv')

        # Assume the target column is the last column in the DataFrame
        target_column = train_data.columns[-1]
        y_train = train_data[target_column]

        # Fit ARIMA model
        model = ARIMA(y_train, order=(5, 1, 0))
        model_fit = model.fit()

        # Forecast the training data
        y_train_pred = model_fit.predict(start=0, end=len(y_train) - 1, dynamic=False)

        # Calculate evaluation metrics
        mae = mean_absolute_error(y_train, y_train_pred)
        rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

        # Return the evaluation metrics
        return jsonify({
            'message': 'ARIMA model trained successfully.',
            'mae': mae,
            'rmse': rmse
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
'''
'''from flask import Flask, request, jsonify, send_file, render_template
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/process', methods=['POST'])
def process_file():
    try:
        file = request.files['file']
        target_column = request.form['target_column']
        df = pd.read_csv(file)
        
        # Splitting the data into train and test
        train_data = df.iloc[:int(0.8 * len(df))]
        test_data = df.iloc[int(0.8 * len(df)):]
        
        # Save cleaned train and test data
        train_data.to_csv('cleaned_train_data.csv', index=False)
        test_data.to_csv('cleaned_test_data.csv', index=False)

        return jsonify({'message': 'File processed and split into train and test data.'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/train-arima', methods=['POST'])
def train_arima():
    try:
        # Load the cleaned training data
        train_data = pd.read_csv('cleaned_train_data.csv')
        test_data = pd.read_csv('cleaned_test_data.csv')

        # Assume the target column is the last column in the DataFrame
        target_column = train_data.columns[-1]
        y_train = train_data[target_column].dropna()
        y_test = test_data[target_column].dropna()

        if len(y_train) == 0 or len(y_test) == 0:
            raise ValueError("Training or testing data is empty after removing NaN values.")

        # Fit ARIMA model
        model = ARIMA(y_train, order=(5, 1, 0))
        model_fit = model.fit()

        # Forecast the test data
        y_test_pred = model_fit.forecast(steps=len(y_test))

        # Calculate evaluation metrics
        mae = mean_absolute_error(y_test, y_test_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100
        nrmse = rmse / (y_test.max() - y_test.min())

        # Ensure metrics are not NaN
        mae = np.nan_to_num(mae, nan=0.0)
        rmse = np.nan_to_num(rmse, nan=0.0)
        mape = np.nan_to_num(mape, nan=0.0)
        nrmse = np.nan_to_num(nrmse, nan=0.0)

        # Return the evaluation metrics
        return jsonify({
            'message': 'ARIMA model trained and forecasted successfully.',
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'nrmse': nrmse
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
'''

'''from flask import Flask, request, jsonify, send_file, render_template
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/process', methods=['POST'])
def process_file():
    try:
        file = request.files['file']
        target_column = request.form['target_column']
        df = pd.read_csv(file)
        
        # Splitting the data into train and test
        train_data = df.iloc[:int(0.8 * len(df))]
        test_data = df.iloc[int(0.8 * len(df)):]
        
        # Save cleaned train and test data
        train_data.to_csv('cleaned_train_data.csv', index=False)
        test_data.to_csv('cleaned_test_data.csv', index=False)

        return jsonify({'message': 'File processed and split into train and test data.'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/train-arima', methods=['POST'])
def train_arima():
    try:
        # Load the cleaned training data
        train_data = pd.read_csv('cleaned_train_data.csv')
        test_data = pd.read_csv('cleaned_test_data.csv')

        # Assume the target column is the last column in the DataFrame
        target_column = train_data.columns[-1]
        y_train = train_data[target_column].dropna()
        y_test = test_data[target_column].dropna()

        if len(y_train) == 0 or len(y_test) == 0:
            raise ValueError("Training or testing data is empty after removing NaN values.")

        # Fit ARIMA model
        model = ARIMA(y_train, order=(5, 1, 0))
        model_fit = model.fit()

        # Forecast the test data
        y_test_pred = model_fit.forecast(steps=len(y_test))

        # Calculate evaluation metrics
        mae = mean_absolute_error(y_test, y_test_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100
        nrmse = rmse / (y_test.max() - y_test.min())

        # Ensure metrics are not NaN
        mae = np.nan_to_num(mae, nan=0.0)
        rmse = np.nan_to_num(rmse, nan=0.0)
        mape = np.nan_to_num(mape, nan=0.0)
        nrmse = np.nan_to_num(nrmse, nan=0.0)

        # Save the forecast results
        forecast_df = pd.DataFrame({
            'actual': y_test,
            'predicted': y_test_pred
        })
        forecast_df.to_csv('forecast_results.csv', index=False)

        # Return the evaluation metrics
        return jsonify({
            'message': 'ARIMA model trained and forecasted successfully.',
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'nrmse': nrmse
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(filename, as_attachment=True)

@app.route('/plot/line')
def plot_line():
    df = pd.read_csv('forecast_results.csv')

    plt.figure(figsize=(12, 6))
    plt.plot(df['actual'], label='Actual', color='blue')
    plt.plot(df['predicted'], label='Forecasted', color='red')
    plt.legend()
    plt.title('Actual vs Forecasted Values')
    plt.xlabel('Date')
    plt.ylabel('Values')

    # Save plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return render_template('plot.html', img_data=img_base64)

@app.route('/plot/bar')
def plot_bar():
    train_data = pd.read_csv('cleaned_train_data.csv')
    test_data = pd.read_csv('cleaned_test_data.csv')

    plt.figure(figsize=(12, 6))
    sns.histplot(train_data.iloc[:, -1], color='blue', label='Train Data', kde=True)
    sns.histplot(test_data.iloc[:, -1], color='red', label='Test Data', kde=True)
    plt.legend()
    plt.title('Distribution of Train and Test Data')
    plt.xlabel('Values')
    plt.ylabel('Frequency')

    # Save plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return render_template('plot.html', img_data=img_base64)

@app.route('/plot/pie')
def plot_pie():
    df = pd.read_csv('forecast_results.csv')

    mae = mean_absolute_error(df['actual'], df['predicted'])
    rmse = np.sqrt(mean_squared_error(df['actual'], df['predicted']))
    mape = np.mean(np.abs((df['actual'] - df['predicted']) / df['actual'])) * 100
    nrmse = rmse / (df['actual'].max() - df['actual'].min())

    sizes = [mae, rmse, mape, nrmse]
    labels = ['MAE', 'RMSE', 'MAPE', 'NRMSE']
    colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']

    plt.figure(figsize=(10, 7))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    plt.title('Error Metrics Distribution')

    # Save plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return render_template('plot.html', img_data=img_base64)

@app.route('/plot/gauge')
def plot_gauge():
    # For gauge chart, we will create a custom gauge using matplotlib

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ['Poor', 'Average', 'Good', 'Excellent']
    colors = ['red', 'orange', 'yellow', 'green']
    metrics = ['MAE', 'RMSE', 'MAPE', 'NRMSE']
    values = [0.2, 0.4, 0.6, 0.8, 1.0]

    ax.barh(metrics, values, color=colors, edgecolor='black')
    for index, value in enumerate(values):
        ax.text(value, index, f'{value:.2f}', va='center', ha='left')

    ax.set_xlim(0, 1)
    ax.set_title('Gauge of Forecasted Data')

    # Save plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return render_template('plot.html', img_data=img_base64)

@app.route('/plot/histogram')
def plot_histogram():
    df = pd.read_csv('forecast_results.csv')

    plt.figure(figsize=(12, 6))
    plt.hist(df['actual'], bins=20, alpha=0.5, label='Actual', color='blue')
    plt.hist(df['predicted'], bins=20, alpha=0.5, label='Forecasted', color='red')
    plt.legend()
    plt.title('Histogram of Actual vs Forecasted Values')
    plt.xlabel('Values')
    plt.ylabel('Frequency')

    # Save plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return render_template('plot.html', img_data=img_base64)

if __name__ == '__main__':
    app.run(debug=True)
'''
'''
VERY IMP
from flask import Flask, request, jsonify, send_file, render_template
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/process', methods=['POST'])
def process_file():
    try:
        file = request.files['file']
        target_column = request.form['target_column']
        df = pd.read_csv(file)
        
        # Splitting the data into train and test
        train_data = df.iloc[:int(0.8 * len(df))]
        test_data = df.iloc[int(0.8 * len(df)):]
        
        # Save cleaned train and test data
        train_data.to_csv('cleaned_train_data.csv', index=False)
        test_data.to_csv('cleaned_test_data.csv', index=False)

        return jsonify({'message': 'File processed and split into train and test data.'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/train-arima', methods=['POST'])
def train_arima():
    try:
        # Load the cleaned training data
        train_data = pd.read_csv('cleaned_train_data.csv')
        test_data = pd.read_csv('cleaned_test_data.csv')

        # Assume the target column is the last column in the DataFrame
        target_column = train_data.columns[-1]
        y_train = train_data[target_column].dropna()
        y_test = test_data[target_column].dropna()

        if len(y_train) == 0 or len(y_test) == 0:
            raise ValueError("Training or testing data is empty after removing NaN values.")

        # Fit ARIMA model
        model = ARIMA(y_train, order=(5, 1, 0))
        model_fit = model.fit()

        # Forecast the test data
        y_test_pred = model_fit.forecast(steps=len(y_test))

        # Calculate evaluation metrics
        mae = mean_absolute_error(y_test, y_test_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100
        nrmse = rmse / (y_test.max() - y_test.min())

        # Ensure metrics are not NaN
        mae = np.nan_to_num(mae, nan=0.0)
        rmse = np.nan_to_num(rmse, nan=0.0)
        mape = np.nan_to_num(mape, nan=0.0)
        nrmse = np.nan_to_num(nrmse, nan=0.0)

        # Save the forecast results
        forecast_df = pd.DataFrame({
            'actual': y_test,
            'predicted': y_test_pred
        })
        forecast_df.to_csv('forecast_results.csv', index=False)

        # Return the evaluation metrics
        return jsonify({
            'message': 'ARIMA model trained and forecasted successfully.',
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'nrmse': nrmse
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(filename, as_attachment=True)

def plot_to_img():
    """Converts a Matplotlib plot to a base64-encoded image."""
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    img_data = base64.b64encode(buffer.read()).decode('utf-8')
    buffer.close()
    plt.close()
    return img_data

@app.route('/plot/<plot_type>')
def plot(plot_type):
    try:
        # Load the forecast results
        forecast_df = pd.read_csv('forecast_results.csv')

        # Generate the requested plot
        if plot_type == 'line':
            plt.figure(figsize=(10, 6))
            plt.plot(forecast_df['actual'], label='Actual')
            plt.plot(forecast_df['predicted'], label='Forecast', linestyle='--')
            plt.legend()
            plt.title('Actual vs Forecast')
        elif plot_type == 'bar':
            plt.figure(figsize=(10, 6))
            forecast_df.plot(kind='bar', stacked=True)
            plt.title('Train and Test Data')
        elif plot_type == 'pie':
            plt.figure(figsize=(8, 8))
            sizes = [forecast_df['actual'].sum(), forecast_df['predicted'].sum()]
            labels = ['Actual', 'Forecast']
            plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
            plt.title('Actual vs Forecast Distribution')
        elif plot_type == 'gauge':
            fig, ax = plt.subplots(figsize=(10, 6))
            size = 200  # Size of the gauge chart
            min_val, max_val = 0, forecast_df['actual'].max()
            actual_val = forecast_df['predicted'].iloc[-1]
            theta = np.linspace(0, 2*np.pi, 100)
            x1, y1 = size*np.cos(theta), size*np.sin(theta)
            x2, y2 = [0, size*np.cos(np.pi * actual_val / max_val)], [0, size*np.sin(np.pi * actual_val / max_val)]
            ax.plot(x1, y1)
            ax.plot(x2, y2, color='r')
            ax.set_xlim(-size, size)
            ax.set_ylim(-size, size)
            plt.title('Forecasted Value Gauge')
        elif plot_type == 'histogram':
            plt.figure(figsize=(10, 6))
            forecast_df['predicted'].plot(kind='hist', bins=30, alpha=0.7)
            plt.title('Forecasted Data Histogram')

        img_data = plot_to_img()
        return render_template('plot.html', img_data=img_data)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
'''

from flask import Flask, request, jsonify, send_file, render_template
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import logging
from dash import Dash, dcc, html
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output
import io

app = Flask(__name__)
dash_app = Dash(__name__, server=app, url_base_pathname='/dash/')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/process', methods=['POST'])
def process_file():
    try:
        file = request.files['file']
        target_column = request.form['target_column']
        df = pd.read_csv(file)
        
        logging.info('File uploaded successfully.')

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in the uploaded file.")

        # Splitting the data into train and test
        train_data = df.iloc[:int(0.8 * len(df))]
        test_data = df.iloc[int(0.8 * len(df)):]
        
        # Save cleaned train and test data
        train_data.to_csv('cleaned_train_data.csv', index=False)
        test_data.to_csv('cleaned_test_data.csv', index=False)

        return jsonify({'message': 'File processed and split into train and test data.'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/train-arima', methods=['POST'])
def train_arima():
    try:
        # Load the cleaned training data
        train_data = pd.read_csv('cleaned_train_data.csv')
        test_data = pd.read_csv('cleaned_test_data.csv')

        # Assume the target column is the last column in the DataFrame
        target_column = train_data.columns[-1]
        y_train = train_data[target_column].dropna()
        y_test = test_data[target_column].dropna()

        if len(y_train) == 0 or len(y_test) == 0:
            raise ValueError("Training or testing data is empty after removing NaN values.")

        # Fit ARIMA model
        model = ARIMA(y_train, order=(5, 1, 0))
        model_fit = model.fit()

        # Forecast the test data + next 3 months (assuming monthly data and 3*30 days for 3 months)
        forecast_steps = len(y_test) + 90
        y_test_pred = model_fit.forecast(steps=forecast_steps)

        # Calculate evaluation metrics for the test period only
        y_test_pred_test_period = y_test_pred[:len(y_test)]
        mae = mean_absolute_error(y_test, y_test_pred_test_period)
        rmse = np.sqrt(mean_squared_error(y_test, y_test_pred_test_period))
        mape = np.mean(np.abs((y_test - y_test_pred_test_period) / y_test)) * 100
        nrmse = rmse / (y_test.max() - y_test.min())

        # Ensure metrics are not NaN
        mae = np.nan_to_num(mae, nan=0.0)
        rmse = np.nan_to_num(rmse, nan=0.0)
        mape = np.nan_to_num(mape, nan=0.0)
        nrmse = np.nan_to_num(nrmse, nan=0.0)

        # Save the forecast results
        forecast_df = pd.DataFrame({
            'actual': pd.concat([y_test, pd.Series([None] * (forecast_steps - len(y_test)))]).reset_index(drop=True),
            'predicted': y_test_pred
        })
        forecast_df.to_csv('forecast_results.csv', index=False)

        # Return the evaluation metrics
        return jsonify({
            'message': 'ARIMA model trained and forecasted successfully.',
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'nrmse': nrmse
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/download/<filename>')
def download_file(filename):
    return send_file(filename, as_attachment=True)

@app.route('/dash')
def render_dash():
    return dash_app.index()

# Dash App Layout
dash_app.layout = html.Div([
    html.H1("Data Visualization Dashboard"),
    dcc.Dropdown(id='plot-type', options=[
        {'label': 'Line Plot', 'value': 'line'},
        {'label': 'Bar Plot', 'value': 'bar'},
        {'label': 'Pie Chart', 'value': 'pie'},
        {'label': 'Gauge Chart', 'value': 'gauge'},
        {'label': 'Histogram', 'value': 'histogram'}
    ], value='line', clearable=False),
    dcc.Dropdown(id='filter-column'),
    dcc.Dropdown(id='filter-id'),
    dcc.Graph(id='graph-output')
])

@dash_app.callback(
    Output('filter-column', 'options'),
    Output('filter-column', 'value'),
    Output('filter-id', 'options'),
    Output('filter-id', 'value'),
    Input('plot-type', 'value')
)
def update_filters(plot_type):
    train_data = pd.read_csv('cleaned_train_data.csv')
    columns = [{'label': col, 'value': col} for col in train_data.columns]
    first_col = train_data.columns[0]
    ids = [{'label': str(id), 'value': id} for id in train_data[first_col].unique()]
    return columns, first_col, ids, ids[0]['value']

@dash_app.callback(
    Output('graph-output', 'figure'),
    Input('plot-type', 'value'),
    Input('filter-column', 'value'),
    Input('filter-id', 'value')
)

def update_graph(plot_type, filter_column, filter_id):
    train_data = pd.read_csv('cleaned_train_data.csv')
    if filter_id:
        train_data = train_data[train_data[filter_column] == filter_id]
    forecast_df = pd.read_csv('forecast_results.csv')

    if plot_type == 'line':
        fig = px.line(forecast_df, x=forecast_df.index, y=['actual', 'predicted'], labels={'value': 'Value', 'index': 'Index'})
    elif plot_type == 'bar':
        fig = px.bar(forecast_df, x=forecast_df.index, y=['actual', 'predicted'], barmode='group')
    elif plot_type == 'pie':
        values = [forecast_df['actual'].sum(), forecast_df['predicted'].sum()]
        labels = ['Actual', 'Forecast']
        fig = px.pie(values=values, names=labels)
    elif plot_type == 'gauge':
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=forecast_df['predicted'].iloc[-1],
            title={'text': "Forecasted Value"},
            gauge={'axis': {'range': [None, forecast_df['actual'].max()]}}
        ))
    elif plot_type == 'histogram':
        fig = px.histogram(forecast_df, x='predicted')
    else:
        fig = go.Figure()

    return fig


if __name__ == '__main__':
    app.run(debug=True)
