def explain_prediction(merchant_data):
    """
    Generate human-readable explanations for merchant status.
    """
    reasons = []
    
    # Recency check
    if merchant_data['recency'] > 30:
        reasons.append("High recency: The merchant hasn't transacted in over 30 days, indicating inactivity.")
    elif merchant_data['recency'] > 15:
        reasons.append("Moderate recency: Activity has slowed down in the last two weeks.")
        
    # Failure rate check
    if merchant_data['failure_rate'] > 0.15:
        reasons.append(f"High failure rate ({merchant_data['failure_rate']:.1%}): Technical issues or payment declines are affecting retention.")
        
    # Frequency/Engagement
    if merchant_data['frequency'] < 5:
        reasons.append("Low frequency: Merchant has very few transactions, showing low engagement with the platform.")
        
    # Value
    if merchant_data['monetary'] > 1000:
        reasons.append("High value: This is a significant revenue contributor.")
        
    if not reasons:
        return "Merchant is active and healthy with no immediate risk factors identified."
    
    return " ".join(reasons)
