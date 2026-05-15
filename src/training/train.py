import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import joblib
import os

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def create_sample_data():
    """Create sample customer data"""
    print("📊 Creating sample customer data...")
    
    n_customers = 5000
    data = {
        'tenure_months': np.random.randint(1, 60, n_customers),
        'monthly_charges': np.random.uniform(20, 150, n_customers),
        'num_support_tickets': np.random.poisson(2, n_customers),
        'satisfaction_score': np.random.uniform(1, 5, n_customers),
        'contract_years': np.random.choice([0, 1, 2], n_customers),
    }
    
    df = pd.DataFrame(data)
    
    # Create churn label
    churn_score = (
        (df['satisfaction_score'] < 2.5) * 0.4 +
        (df['monthly_charges'] > 100) * 0.3 +
        (df['num_support_tickets'] > 3) * 0.3
    )
    df['churn'] = (churn_score > 0.5).astype(int)
    
    print(f"✅ Created {n_customers} customers")
    return df

def train_models():
    """
    Train multiple models and track in MLflow
    """
    print("\n" + "="*50)
    print("🚀 STARTING EXPERIMENT TRACKING")
    print("="*50)
    
    # Set up MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("churn_prediction")
    
    # Create data
    df = create_sample_data()
    feature_cols = ['tenure_months', 'monthly_charges', 'num_support_tickets', 
                    'satisfaction_score', 'contract_years']
    X = df[feature_cols]
    y = df['churn']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )
    
    # Define models to try
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED),
        "Logistic Regression": LogisticRegression(random_state=RANDOM_SEED, max_iter=1000)
    }
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\n🌲 Training {model_name}...")
        
        # Start MLflow run
        with mlflow.start_run(run_name=model_name):
            # Log parameters
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("random_seed", RANDOM_SEED)
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predict and evaluate
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            
            # Log the model
            mlflow.sklearn.log_model(model, model_name.replace(" ", "_").lower())
            
            results[model_name] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            }
            
            print(f"   ✅ Accuracy: {accuracy:.4f}")
            print(f"   ✅ F1 Score: {f1:.4f}")
    
    # Save best model
    best_model_name = max(results, key=lambda x: results[x]['accuracy'])
    best_model = models[best_model_name]
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/best_churn_model.pkl')
    
    print("\n" + "="*50)
    print("🏆 BEST MODEL:", best_model_name)
    print(f"   Accuracy: {results[best_model_name]['accuracy']:.4f}")
    print("="*50)
    
    return results

if __name__ == "__main__":
    results = train_models()
    print("\n📊 To view experiments, run: mlflow ui")
    print("   Then open http://localhost:5000 in your browser")