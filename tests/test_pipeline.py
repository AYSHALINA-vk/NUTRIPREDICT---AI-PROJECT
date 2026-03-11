# tests/test_pipeline.py - For pytest in CI
import pytest
import pandas as pd
from sklearn.metrics import accuracy_score

# Mock data for fast tests
@pytest.fixture
def mock_df():
    return pd.DataFrame({
        'Calories (kcal)': [100, 200],
        'Protein (g)': [10, 20],
        'Protein (g)_risk': [1, 0]  # Mock label
    })

def test_data_ingestion(mock_df):
    assert len(mock_df) > 0, "Ingestion failed"
    assert 'Protein (g)' in mock_df.columns

# Mock model train/test
def test_model_accuracy(mock_df):
    from sklearn.ensemble import RandomForestClassifier
    X = mock_df[['Calories (kcal)', 'Protein (g)']]
    y = mock_df['Protein (g)_risk']
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    model = RandomForestClassifier().fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    assert acc >= 0.5, "Model accuracy too low"

# Run: pytest tests/test_pipeline.py -vpip install pytest