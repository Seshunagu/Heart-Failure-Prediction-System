This project is a Heart Failure Prediction System that uses a hybrid machine learning approach combining an Artificial Neural Network (ANN) for feature selection and a Decision Tree (DT) for final classification. The aim is to assist doctors and patients in predicting the risk of heart failure using clinical inputs.

System Features: The system has a frontend form that collects 13 medical input features from the user. The backend processes these inputs using a trained hybrid model. The prediction result is displayed in the console. Additionally, all inputs and prediction results are stored in an SQLite database for future reference.

Machine Learning Model: The dataset used is the Cleveland Heart Disease Dataset from the UCI Repository. The system uses an Artificial Neural Network (Multi-Layer Perceptron) to select the most relevant 8 features out of 13. These features are then passed to a Decision Tree classifier for the final prediction.

**Model Performance:**

ANN Accuracy: 99.71 percent

Decision Tree Accuracy: 98.54 percent

**How to Run:**
Clone the repository from GitHub.

Install the required Python dependencies using pip.

Run the main backend script (app.py).

Open a browser and navigate to http://localhost:5000.

Enter the medical values into the form and submit.

View the prediction result in the console.

All data will be saved in the SQLite database.

Technologies Used: Python for backend development, Flask as the web framework, Scikit-learn for machine learning models, SQLite for database storage, and HTML/CSS for the frontend interface.

**Folder Structure:**

model: contains the trained ANN and DT model files

templates: includes the HTML files for the web interface

static: used optionally for CSS or JavaScript

app.py: the main backend script

database.db: the SQLite database file

requirements.txt: lists the Python packages needed

README.txt: contains project documentation
