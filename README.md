# Real Estate Price Prediction - Bengaluru

A machine learning project that predicts house prices in Bengaluru, India using various features like location, square footage, number of bedrooms, bathrooms, and more.

## 📊 Project Overview

This project analyzes the Bengaluru real estate market and builds a predictive model to estimate house prices based on various property characteristics. The model uses a comprehensive dataset of over 13,000 property listings from different areas of Bengaluru.

## 🏠 Dataset Features

The dataset includes the following features:
- **area_type**: Type of area (Super built-up, Built-up, Plot, etc.)
- **availability**: Availability status (Ready To Move, specific dates)
- **location**: Property location (various areas in Bengaluru)
- **size**: Number of bedrooms (BHK - Bedroom, Hall, Kitchen)
- **society**: Housing society name
- **total_sqft**: Total square footage
- **bath**: Number of bathrooms
- **balcony**: Number of balconies
- **price**: Price in lakhs (target variable)

## 🚀 Features

- **Data Analysis**: Comprehensive exploratory data analysis with visualizations
- **Feature Engineering**: Advanced feature engineering including:
  - Location encoding (one-hot encoding for 240+ locations)
  - Size extraction (BHK conversion)
  - Outlier detection and removal
  - Data cleaning and preprocessing
- **Machine Learning Models**: Multiple model comparison including:
  - Linear Regression
  - Lasso Regression
  - Decision Tree Regressor
- **Model Evaluation**: Cross-validation and performance metrics
- **Price Prediction**: Interactive function to predict house prices
- **Model Persistence**: Saved trained model for future use

## 📁 Project Structure

```
Real_estate_price_prediction/
├── Project.ipynb                    # Main Jupyter notebook with analysis
├── Bengaluru_House_Data.csv         # Dataset (13,320 records)
├── Bengaluru_House_Prices_Model.pickle  # Trained model
├── columns.json                     # Feature columns for model input
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Git ignore file
└── README.md                        # This file
```

## 🛠️ Installation

### Prerequisites

- Python 3.7+
- Jupyter Notebook or JupyterLab

### Setup

1. Clone the repository:
```bash
git clone [<repository-url>](https://github.com/Ayush-Repositories/Real-Estate-Price-Prediction)
cd Real_estate_price_prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Open the Jupyter notebook:
```bash
jupyter notebook Project.ipynb
```

## 📖 Usage

### Running the Analysis

1. Open `Project.ipynb` in Jupyter Notebook
2. Run all cells to perform the complete analysis
3. The notebook includes:
   - Data loading and exploration
   - Data preprocessing and feature engineering
   - Model training and evaluation
   - Price prediction examples

### Making Predictions

Use the trained model to predict house prices:

```python
# Load the model
import pickle
with open('Bengaluru_House_Prices_Model.pickle', 'rb') as f:
    model = pickle.load(f)

# Load feature columns
import json
with open('columns.json', 'r') as f:
    columns = json.load(f)

# Predict price for a property
def predict_price(location, sqft, bath, bhk):
    loc_index = np.where(X.columns==location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if(loc_index>=0):
        x[loc_index] = 1
    return model.predict([x])[0]

# Example usage
price = predict_price('Indira Nagar', 1000, 2, 2)
print(f"Predicted price: {price:.2f} lakhs")
```

## 📈 Model Performance

The project compares multiple machine learning algorithms:

- **Linear Regression**: Baseline model with feature scaling
- **Lasso Regression**: Regularized linear model
- **Decision Tree**: Non-linear model for complex patterns

The models are evaluated using cross-validation to ensure robust performance estimates.

## 🔍 Key Insights

- **Location Impact**: Location is a significant factor in price determination
- **Size Correlation**: Strong correlation between square footage and price
- **BHK Effect**: Number of bedrooms significantly affects pricing
- **Market Trends**: Ready-to-move properties command different pricing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👨‍💻 Author

**Ayush** - Real Estate Price Prediction Project

## 🙏 Acknowledgments

- Dataset source: Bengaluru real estate market data
- Libraries: NumPy, Pandas, Matplotlib, Scikit-learn
- Community: Open source machine learning community

## 📞 Contact

For questions or suggestions, please open an issue in the repository.

---

**Note**: This model is trained on historical data and should be used as a reference. Real estate prices are influenced by many factors and market conditions change over time. 
