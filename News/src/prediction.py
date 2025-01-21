def get_predictions(model, X_test):
    """
    Get predicted labels and probabilities.
    
    Args: 
        model (sklearn.linear_model): Trained model for prediction
        X_test (pandas.core.frame.DataFrame): Test data to make predictions on
        
    Returns:
        prediction (array): Predicted labels (0 or 1)
        probabilities (array): Predicted probabilities for each class (real/fake)
    """
    # Get predicted labels (0 or 1)
    pred = model.predict(X_test)
    
    # Get predicted probabilities for each class
    probabilities = model.predict_proba(X_test)
    
    return pred, probabilities
