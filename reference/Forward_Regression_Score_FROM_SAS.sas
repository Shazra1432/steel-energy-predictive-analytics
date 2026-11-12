*****************************************;
** SAS Scoring Code for PROC GLMSelect;
*****************************************;

label P_Usage_kWh = 'Predicted: Usage_kWh' ;
drop _LMR_BAD;
_LMR_BAD=0;

*** Check interval variables for missing values;
if nmiss(usage_lag1,usage_lag4,usage_lag96,usage_lag672,hour_sin,hour_cos,
        is_weekend,usage_lag1_sq) then do;
   _LMR_BAD=1;
   goto _SKIP_000;
end;

*** Compute Linear Predictors;
drop _LP0;
_LP0 = 0;

*** Effect: usage_lag1;
_LP0 = _LP0 + (1.04055516102348) * usage_lag1;
*** Effect: usage_lag4;
_LP0 = _LP0 + (-0.02734953591693) * usage_lag4;
*** Effect: usage_lag96;
_LP0 = _LP0 + (0.07731772957606) * usage_lag96;
*** Effect: usage_lag672;
_LP0 = _LP0 + (0.09437119994865) * usage_lag672;
*** Effect: hour_sin;
_LP0 = _LP0 + (1.12543767722858) * hour_sin;
*** Effect: hour_cos;
_LP0 = _LP0 + (-0.93900333488321) * hour_cos;
*** Effect: is_weekend;
_LP0 = _LP0 + (-1.97408248403902) * is_weekend;
*** Effect: usage_lag1_sq;
_LP0 = _LP0 + (-0.00237915617447) * usage_lag1_sq;

*** Predicted values;
_LP0 = _LP0 +    -0.01697242092658;
_SKIP_000:
if _LMR_BAD=1 then do;
   P_Usage_kWh = .;
end;
else do;
   P_Usage_kWh = _LP0;
end;
