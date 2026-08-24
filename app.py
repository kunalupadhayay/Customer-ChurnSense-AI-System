from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)

# Load and prepare the model
data = pd.read_csv('churn.csv')

# Also load raw copy for template-driven UI (preserve original values)
raw = pd.read_csv('churn.csv')

# Handle missing values just like in the notebook
data.fillna(data.mean(numeric_only=True), inplace=True)

# Prepare UI choices and numeric limits from raw data
raw.fillna(raw.mean(numeric_only=True), inplace=True)
try:
    state_options = sorted(raw['State'].dropna().unique().tolist())
except Exception:
    state_options = []

from pandas.api import types as pdtypes
limits = {}
for col in raw.columns:
    if col == 'Churn?':
        continue
    try:
        if pdtypes.is_numeric_dtype(raw[col]):
            col_min = raw[col].min()
            col_max = raw[col].max()
            # Cast to int when both are integers
            if float(col_min).is_integer() and float(col_max).is_integer():
                col_min = int(col_min)
                col_max = int(col_max)
            limits[col] = {'min': col_min, 'max': col_max}
    except Exception:
        continue

# Encode categorical features
import pandas.api.types as ptypes
encoders = {}
for col in data.columns:
    if (data[col].dtype == 'object' or ptypes.is_string_dtype(data[col])) and col != 'Churn?':
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        encoders[col] = le

# Encode target variable
target_le = LabelEncoder()
data['Churn?'] = target_le.fit_transform(data['Churn?'].astype(str))

# Prepare features and target
X = data.drop('Churn?', axis=1)
y = data['Churn?']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_scaled, y)

@app.route('/')
def index():
    return render_template('index.html', state_options=state_options, limits=limits)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Accept either JSON body or form-encoded data
        req_data = request.get_json(silent=True)
        if not req_data:
            req_data = request.form.to_dict()

        input_df = pd.DataFrame([req_data])

        # Ensure correct column order and include any missing columns
        input_df = input_df.reindex(columns=X.columns)
        
        # Encode categorical inputs from the web form
        for col in input_df.columns:
            # Fill missing values: categorical -> most common known class, numeric -> median
            if input_df[col].isnull().any():
                if col in encoders:
                    input_df[col].fillna(encoders[col].classes_[0], inplace=True)
                else:
                    # fallback to median of training data for numeric columns
                    try:
                        input_df[col] = input_df[col].fillna(X[col].median())
                    except Exception:
                        input_df[col].fillna(0, inplace=True)

            if col in encoders:
                le = encoders[col]
                known_classes = set([str(c) for c in le.classes_])
                # Handle unseen labels by mapping to the first known class
                input_df[col] = input_df[col].apply(lambda x: str(x) if str(x) in known_classes else str(le.classes_[0]))
                input_df[col] = le.transform(input_df[col].astype(str))
                
        # Scale inputs
        input_scaled = scaler.transform(input_df)
        
        # Predict
        prediction = model.predict(input_scaled)
        
        # Map prediction back to original class label
        result_class = target_le.inverse_transform(prediction)[0]
        
        # Check if the result indicates churn (adjusting for 'True.' string format in original data)
        is_churn = 1 if 'True' in str(result_class) else 0
        
        return jsonify({'prediction': is_churn})
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)