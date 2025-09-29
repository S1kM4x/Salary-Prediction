# Salary Prediction

Machine learning project for predicting salaries based on professional and demographic characteristics using regression techniques.

## Description

This project implements a salary prediction model that analyzes various factors such as years of experience, education level, age, gender, and geographic location to estimate a professional's salary. It includes an interactive web application developed with Streamlit for real-time predictions.

## Features

- **Complete Exploratory Data Analysis (EDA)**
- **Trained and optimized machine learning models**
- **Interactive web application** with Streamlit
- **Visualizations** of correlations and patterns
- **Robust data preprocessing**
- **Real-time predictions**

## Project Structure

```
Salary-Prediction/
│
├── data         
│   └── Salary_Data.csv  
├── models         
│   └── model_salary.joblib      
├── notebooks         
│   ├── 01_eda.ipynb    
│   └── 02_eda.ipynb
├── streamlit         
│   └── app.py 
├── .gitignore
├── README.md
└── requirements.txt

```

## Dataset

The dataset used comes from [Kaggle - Salary Prediction Dataset](https://www.kaggle.com/datasets/wardabilal/salary-prediction-dataset/data) and contains information about:

- **Age**: Employee's age
- **Gender**: Gender (Male/Female)
- **Education Level**: Educational level (Bachelor's, Master's, PhD)
- **Job Title**: Job position title
- **Years of Experience**: Years of work experience
- **Salary**: Salary (target variable)
- **Country**: Country of residence
- **Race**: Ethnicity

## 🚀 Installation

### Prerequisites

- Python 3.11
- pip

### Installation Steps

1. Clone this repository:
```Terminal
git clone https://github.com/S1kM4x/Salary-Prediction.git
cd Salary-Prediction
```

2. Install dependencies:
```Terminal
pip install -r requirements.txt
```

*Note: If a `requirements.txt` file doesn't exist, install the following libraries:*
```Terminal
pip install pandas numpy scikit-learn matplotlib seaborn streamlit joblib
```

## 💻 Usage

### Run the Web Application

To start the interactive Streamlit application:

```Terminal
streamlit run app.py
```

The application will automatically open in your browser at `http://localhost:8501`

### Run the Notebook

Open the `01_eda.ipynb` notebook in Jupyter Notebook or JupyterLab:
Open the `02_eda.ipynb` notebook in Jupyter Notebook or JupyterLab:

```Terminal
jupyter notebook 01_eda.ipynb
jupyter notebook 02_eda.ipynb
```

## 🧠 Methodology

1. **Data Loading and Exploration**: Initial dataset analysis
2. **Data Cleaning**: Handling null values and outliers
3. **Feature Engineering**: Encoding categorical variables
4. **Data Split**: Train/Test split
5. **Model Training**: Testing different regression algorithms
6. **Evaluation**: Performance metrics (R², RMSE, MAE)
7. **Optimization**: Hyperparameter tuning
8. **Deployment**: Web application with Streamlit

## 📈 Models Used

The project explores different machine learning algorithms:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost

## 🎨 Application Demo

The `app.py` application allows:

- Input employee characteristics through an intuitive form
- Get instant salary predictions
- Visualize factors influencing the prediction
- User-friendly and responsive interface

## 📊 Results

The final model achieves competitive performance metrics in salary prediction, considering multiple professional and demographic variables.

*(Add your specific metrics here when available)*

## 🛠️ Technologies Used

- **Python**: Main language
- **Pandas**: Data manipulation
- **NumPy**: Numerical operations
- **Scikit-learn**: Machine learning models
- **Matplotlib/Seaborn**: Visualizations
- **Streamlit**: Interactive web application
- **Joblit**: Model serialization

## 👤 Author

**S1kM4x**

- GitHub: [@S1kM4x](https://github.com/S1kM4x)
- Project Link: [https://github.com/S1kM4x/Salary-Prediction](https://github.com/S1kM4x/Salary-Prediction)

## 🙏 Acknowledgments

- Dataset provided by [Warda Bilal](https://www.kaggle.com/wardabilal) on Kaggle
- Thanks to the open-source community for the amazing tools and libraries
