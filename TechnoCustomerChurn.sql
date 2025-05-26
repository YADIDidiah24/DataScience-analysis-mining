CREATE OR REPLACE MODEL `customer_churn_dataset.customer_churn_model_v2`
OPTIONS(
  model_type="LOGISTIC_REG",
  input_label_cols=['Churn'],
  -- Address class imbalance with auto class weights
  auto_class_weights=TRUE,
  -- Use L1 regularization to prevent overfitting
  l1_reg=0.1,
  -- Enable early stopping
  early_stop=TRUE,
  -- Increase max iterations
  max_iterations=50
) AS
SELECT
  gender,
  CAST(SeniorCitizen AS STRING) AS SeniorCitizen,
  Partner,
  Contract,
  PaymentMethod,
  tenure,
  MonthlyCharges,
  TotalCharges,
  tenure_category,
  monthly_charges_category,
  avg_monthly_revenue,
  contract_risk_score,
  payment_risk_score,
  senior_single_flag,
  Churn
FROM `customer_churn_dataset.customer_data_engineered`
WHERE TotalCharges IS NOT NULL; -- Remove null values



SELECT
  'Logistic_Regression_v2' as model_name,
  *
FROM
  ML.CONFUSION_MATRIX(MODEL `customer_churn_dataset.customer_churn_model_v2`);



CREATE OR REPLACE TABLE `customer_churn_dataset.churn_predictions_optimized` AS
SELECT
  customerID,
  gender,
  Contract,
  PaymentMethod,
  predicted_Churn,
  predicted_Churn_probs,
  -- Custom threshold for better recall (lower threshold = higher recall)
  CASE 
    WHEN predicted_Churn_probs[OFFSET(1)].prob >= 0.3 THEN 'Yes'
    ELSE 'No'
  END AS churn_prediction_30_threshold,
  CASE 
    WHEN predicted_Churn_probs[OFFSET(1)].prob >= 0.25 THEN 'Yes'
    ELSE 'No'
  END AS churn_prediction_25_threshold,
  predicted_Churn_probs[OFFSET(1)].prob as churn_probability
FROM
  ML.PREDICT(
    MODEL `customer_churn_dataset.customer_churn_model_v2`, -- Use your best model here
    (
      SELECT
        customerID,
        gender,
        CAST(SeniorCitizen AS STRING) AS SeniorCitizen,
        Partner,
        Contract,
        PaymentMethod,
        tenure,
        MonthlyCharges,
        TotalCharges,
        tenure_category,
        monthly_charges_category,
        avg_monthly_revenue,
        contract_risk_score,
        payment_risk_score,
        senior_single_flag
      FROM
        `customer_churn_dataset.customer_data_engineered`
    )
  )
ORDER BY churn_probability DESC;


