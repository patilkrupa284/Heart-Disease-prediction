from flask import Flask, request, render_template
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load the dataset
url = r"C:\Users\KRUPA\Desktop\heart_new\heart.csv"
data = pd.read_csv(url)

# Ensure the first row contains column names
# If the file doesn't have a header, you can specify column names manually
# Example: column_names = ['age', 'sex', 'cp', ...]
# data = pd.read_csv(url, names=column_names, header=None)

# Drop any non-numeric columns or columns that are not needed for training
# Adjust this step based on your dataset structure
# For example, if 'target' is the last column, you can drop it directly
data = data.dropna()  # Drop any rows with missing values
X = data.drop('target', axis=1)
y = data['target']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_inputs = [float(x) for x in request.form.values()]
        user_inputs_scaled = scaler.transform([user_inputs])
        prediction = model.predict(user_inputs_scaled)[0]
        if prediction == 1:
            result = "Based on the information provided, you are likely to have heart disease."
        else:
            result = "Based on the information provided, you are unlikely to have heart disease."
        return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
