import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def download_stock_data(ticker, start_date, end_date):
    """Download stock data using yfinance with auto_adjust=False"""
    if isinstance(ticker, list):
        ticker = ticker[0]
    
    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
    
    # Flatten MultiIndex columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    return data


def create_additional_features(stock_data):
    """Create technical indicators from price data"""
    df = stock_data.copy()
    
    # Asegurarse de que tenemos Series, no DataFrames
    adj_close = df['Adj Close']
    if isinstance(adj_close, pd.DataFrame):
        adj_close = adj_close.iloc[:, 0]
    
    volume = df['Volume']
    if isinstance(volume, pd.DataFrame):
        volume = volume.iloc[:, 0]
    
    # Crear nuevo DataFrame con solo las columnas necesarias
    result = pd.DataFrame(index=df.index)
    result['Adj Close'] = adj_close
    result['Volume'] = volume
    
    # Retornos semanales (reduce ruido)
    result['Weekly_Return'] = adj_close.pct_change(5)
    
    # Volatilidad (desviación estándar de retornos)
    result['Volatility'] = adj_close.pct_change().rolling(window=20).std()
    
    # Media móvil (tendencia)
    sma_20 = adj_close.rolling(window=20).mean()
    sma_50 = adj_close.rolling(window=50).mean()
    result['SMA_20'] = sma_20
    result['SMA_50'] = sma_50
    result['Price_vs_SMA'] = (adj_close - sma_20) / sma_20
    
    # Momentum (cambio de precio relativo)
    result['Momentum'] = adj_close.pct_change(5)  # 5-day momentum
    
    # Volumen relativo
    vol_mean = volume.rolling(window=20).mean()
    result['Volume_Ratio'] = volume / vol_mean
    
    return result.dropna()


def prepare_data_for_ml(stock_data):
    """Prepare features and target with lag structure"""
    df = stock_data.copy()
    
    # Target: Retorno semanal forward-looking (lo que queremos predecir)
    df['Target'] = df['Weekly_Return'].shift(-5)  # Retorno de la próxima semana
    
    return df.dropna()


def train_model(model, X_train, y_train):
    """Train the model"""
    model.fit(X_train, y_train)
    return model


def predict_future_returns(model, X_test):
    """Predict WEEKLY returns from trained model"""
    predictions = model.predict(X_test)
    # Retorna el promedio de predicciones
    return np.mean(predictions)


def get_model_confidence(model, X_test, y_test):
    """Calculate model confidence using R² score"""
    from sklearn.metrics import r2_score
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    
    # Clamp entre 0.01 y 1 para mejor interpretación
    # Si R² es negativo o muy bajo, usar mínimo de 0.01
    r2 = max(0.01, min(1.0, r2))
    
    return r2


def generate_investor_views(ticker, start_date, end_date, model_type='Gradient Boosting'):
    """
    Generates stock RETURN predictions using weekly returns.
    
    :param str ticker: ticker to generate investor views for
    :param start_date: start date for training in form 'YYYY-MM-DD'
    :param end_date: end date for training in form 'YYYY-MM-DD'
    :param model_type: 'Gradient Boosting' (recommended), 'Random Forest', or 'Linear Regression'
    :return: tuple with predicted ANNUAL RETURNS and model's confidence (R²)
    """
    try:
        # 1. Download and prepare data
        stock_data = download_stock_data(ticker, start_date, end_date)
        ml_stock_data = create_additional_features(stock_data)
        ml_stock_data = prepare_data_for_ml(ml_stock_data)
        
        # 2. Prepare X and y
        feature_cols = ['Volatility', 'Price_vs_SMA', 'Momentum', 'Volume_Ratio', 
                        'SMA_20', 'SMA_50']
        X = ml_stock_data[feature_cols].copy()
        y = ml_stock_data['Target'].copy()
        
        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)
        
        # Remove NaN values
        valid_idx = ~np.isnan(y.values)
        X = X[valid_idx].reset_index(drop=True)
        y = y[valid_idx].reset_index(drop=True)
        
        if len(X) < 100:
            print(f"⚠️ Insufficient data for {ticker} ({len(X)} samples)")
            return 0.0, 0.1
        
        # 3. Train-test split (70-30 para más datos en test)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, shuffle=True
        )
        
        # Handle any remaining NaN
        imputer = SimpleImputer(strategy='mean')
        X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=feature_cols)
        X_test = pd.DataFrame(imputer.transform(X_test), columns=feature_cols)
        
        # 4. Select and train model
        if model_type == 'Random Forest':
            model = RandomForestRegressor(n_estimators=100, max_depth=10, 
                                        random_state=42, n_jobs=-1)
        elif model_type == 'Gradient Boosting':
            model = GradientBoostingRegressor(n_estimators=100, max_depth=5,
                                            learning_rate=0.1, subsample=0.8,
                                            random_state=42)
        else:
            model = LinearRegression()
        
        trained_model = train_model(model, X_train, y_train)
        
        # 5. Predict weekly returns and annualize
        predicted_weekly_return = predict_future_returns(trained_model, X_test)
        
        # Annualize: (1 + r_weekly)^52 - 1
        predicted_annual_return = (1 + predicted_weekly_return) ** 52 - 1
        
        # Clip to reasonable bounds
        predicted_annual_return = np.clip(predicted_annual_return, -0.5, 2.0)
        
        # 6. Get confidence
        confidence = get_model_confidence(trained_model, X_test, y_test)
        
        print(f"  ✅ {ticker}: Weekly Return: {predicted_weekly_return*100:.3f}%, "
              f"Annual Return: {predicted_annual_return*100:.2f}%, "
              f"Confidence (R²): {confidence:.3f}, "
              f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        return predicted_annual_return, confidence
        
    except Exception as e:
        print(f"  ⚠️ Error processing {ticker}: {str(e)}")
        import traceback
        traceback.print_exc()
        return 0.0, 0.1