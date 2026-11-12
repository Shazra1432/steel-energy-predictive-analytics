"""SAS PROC GLMSELECT scoring equation, not a newly trained Python regression."""
COEFFICIENTS = {
    'usage_lag1': 1.04055516102348,
    'usage_lag4': -0.02734953591693,
    'usage_lag96': 0.07731772957606,
    'usage_lag672': 0.09437119994865,
    'hour_sin': 1.12543767722858,
    'hour_cos': -0.93900333488321,
    'is_weekend': -1.97408248403902,
    'usage_lag1_sq': -0.00237915617447,
}
INTERCEPT = -0.01697242092658

def predict_kwh(features):
    """Compute the exact exported linear scoring formula (no clipping)."""
    return INTERCEPT + sum(c * float(features[k]) for k, c in COEFFICIENTS.items())
