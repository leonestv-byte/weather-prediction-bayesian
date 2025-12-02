import pandas as pd
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
import scipy.stats as st

# Load the data
df = pd.read_csv('NYC_Weather_2016_2022.csv')

# Convert 'time' column to datetime
df['time'] = pd.to_datetime(df['time'])

# Set the time as the index
df.set_index('time', inplace=True)

# Extract
target_dates = pd.date_range('2021-12-03', '2021-12-07')
target_mask = df.index.isin(target_dates)

df_train = df[~target_mask]
df_test = df[target_mask]



# Keep only temperature column
ts_train = df_train['temperature_2m (°C)']
ts_test = df_test['temperature_2m (°C)']

ts_train = ts_train.dropna()
ts_test = ts_test.dropna()

# Add time-based features as covariates
ts_df_train = ts_train.to_frame()
ts_df_train['day_of_year'] = ts_df_train.index.dayofyear
ts_df_train['year'] = ts_df_train.index.year
ts_df_train['days_from_start'] = (ts_df_train.index - ts_df_train.index[0]).days
ts_df_train['month'] = ts_df_train.index.month
ts_df_train['quarter'] = ts_df_train.index.quarter

# The temperature values themselves are our y values
y_train = ts_df_train['temperature_2m (°C)'].values
# Use all of our time based features, except for year, as covariates
x_train= ts_df_train[['month', 'quarter', 'day_of_year']].values

# Do the same for test data

# Add time-based features as covariates
ts_df_test = ts_test.to_frame()
ts_df_test['day_of_year'] = ts_df_test.index.dayofyear
ts_df_test['year'] = ts_df_test.index.year
ts_df_test['days_from_start'] = (ts_df_test.index - ts_df_test.index[0]).days
ts_df_test['month'] = ts_df_test.index.month
ts_df_test['quarter'] = ts_df_test.index.quarter

# The temperature values themselves are our y values
y_test = ts_df_test['temperature_2m (°C)'].values
# Use all of our time based features, except for year, as covariates
x_test = ts_df_test[['month', 'quarter', 'day_of_year']].values



#Train and predict from bayesian linear regression
with pm.Model() as basic_model:
    # Priors for regression coefficients
    beta = pm.Normal("beta", mu=5, sigma=5, shape=x_train.shape[1])
    intercept = pm.Normal("intercept", mu=30, sigma=20)
    
    # Noise (sigma)
    sigma = pm.HalfNormal("sigma", sigma=2)

    # Expected value of outcome
    mu = intercept + pm.math.dot(x_train, beta)

    # Likelihood
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_train)

    # Sample from posterior
    trace = pm.sample(
        2000,
        tune=2000,
        target_accept=0.9,
        chains=4,
        random_seed=321
    )

print("betas shape:", trace.posterior["beta"].shape)

post = trace.posterior

betas = post["beta"].stack(sample=("chain", "draw")).transpose("sample", "beta_dim_0").values
intercepts = post["intercept"].stack(sample=("chain", "draw")).values
sigmas = post["sigma"].stack(sample=("chain", "draw")).values


mu_test = x_test @ betas.T   

# Add intercept (broadcasts across draws)
mu_test = mu_test + intercepts

# Add predictive noise
y_pred_samples = mu_test + np.random.randn(*mu_test.shape) * sigmas

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    if n <= 1:
        return (np.nan, np.nan) # Cannot compute CI with less than 2 samples
    m = np.mean(a)
    se = st.sem(a) # Standard error of the mean
    lower_bound, upper_bound = st.t.interval(confidence, n - 1, loc=m, scale=se)
    return lower_bound, upper_bound


print((np.mean(y_pred_samples[0])) * (9/5) + 32)
print((np.mean(y_pred_samples[1]))  * (9/5) + 32)
print((np.mean(y_pred_samples[2]))  * (9/5) + 32)
print((np.mean(y_pred_samples[3])) * (9/5) + 32)
print((np.mean(y_pred_samples[4])) * (9/5) + 32)

results = []

conf_level = 0.95
Ld, Ud = mean_confidence_interval(y_pred_samples[0], confidence=conf_level)
print(f"Sample Mean: {np.mean(y_pred_samples[0] * (9/5) + 32):.2f} °F")
print(f"{conf_level*100}% Confidence Interval [Ld, Ud] for December 3rd: [{Ld* (9/5) + 32:.2f} °F, {Ud* (9/5) + 32:.2f} °F]")

results.append([np.mean(y_pred_samples[0]) * (9/5) + 32, Ld* (9/5) + 32 ,Ud* (9/5) + 32 ])

conf_level = 0.95
Ld, Ud = mean_confidence_interval(y_pred_samples[1], confidence=conf_level)
print(f"Sample Mean: {np.mean(y_pred_samples[1]* (9/5) + 32):.2f} °F")
print(f"{conf_level*100}% Confidence Interval [Ld, Ud] for December 4th: [{Ld* (9/5) + 32:.2f} °F, {Ud* (9/5) + 32:.2f} °F]")

results.append([np.mean(y_pred_samples[1])  * (9/5) + 32, Ld* (9/5) + 32,Ud* (9/5) + 32 ])

conf_level = 0.95
Ld, Ud = mean_confidence_interval(y_pred_samples[2], confidence=conf_level)
print(f"Sample Mean: {np.mean(y_pred_samples[2]* (9/5) + 32):.2f} °F")
print(f"{conf_level*100}% Confidence Interval [Ld, Ud] for December 5th: [{Ld* (9/5) + 32:.2f} °F, {Ud* (9/5) + 32:.2f} °F]")

results.append([np.mean(y_pred_samples[2])  * (9/5) + 32 , Ld* (9/5) + 32,Ud* (9/5) + 32 ])

conf_level = 0.95
Ld, Ud = mean_confidence_interval(y_pred_samples[3], confidence=conf_level)
print(f"Sample Mean: {np.mean(y_pred_samples[3]* (9/5) + 32):.2f} °F")
print(f"{conf_level*100}% Confidence Interval [Ld, Ud] for December 6th: [{Ld* (9/5) + 32:.2f} °F, {Ud* (9/5) + 32:.2f} °F]")

results.append([np.mean(y_pred_samples[3])  * (9/5) + 32, Ld* (9/5) + 32 ,Ud* (9/5) + 32 ])

conf_level = 0.95
Ld, Ud = mean_confidence_interval(y_pred_samples[4], confidence=conf_level)
print(f"Sample Mean: {np.mean(y_pred_samples[4]* (9/5) + 32):.2f} °F")
print(f"{conf_level*100}% Confidence Interval [Ld, Ud] for December 7th: [{Ld* (9/5) + 32:.2f} °F, {Ud* (9/5) + 32:.2f} °F]")

results.append([np.mean(y_pred_samples[4])  * (9/5) + 32, Ld* (9/5) + 32 , Ud* (9/5) + 32 ])


from tabulate import tabulate

headers = ["Dec 3", "Dec 4", "Dec 5", "Dec 6", "Dec 7"]


print(tabulate(results, headers=headers, tablefmt="grid"))










# import pmdarima as pm

# # Bayesian ARIMA
# model = pm.auto_arima(
#     ts,
#     start_p=1, start_q=1,
#     max_p=5, max_q=5,
#     seasonal=False,
#     stepwise=True,
#     suppress_warnings=True,
#     information_criterion='aic',
#     trace=True,
#     error_action='ignore'
# )

# # Define forecast horizon
# forecast_dates = pd.date_range('2021-12-03', '2021-12-07')
# print("forecast_dates")
# print(type(forecast_dates))
# print(forecast_dates)

# # Forecast
# n_periods = len(forecast_dates)
# forecast, conf_int = model.predict(n_periods=n_periods, return_conf_int=True)

# # Create a DataFrame with results
# forecast_df = pd.DataFrame({
#     'forecast': forecast,
#     'lower_ci': conf_int[:, 0],
#     'upper_ci': conf_int[:, 1]
# }, index=forecast_dates)

# print(forecast_df)

# def get_three_predictions(df):
#     predictions = {}
#     for date, row in df.iterrows():
#         lower = (row['lower_ci']* 9/5) + 32
#         upper = (row['upper_ci']* 9/5) + 32
#         midpoint = (((lower + upper) / 2)* 9/5) + 32
#         predictions[date] = [lower, midpoint, upper]
#     return pd.DataFrame(predictions).T 

# # Get predictions
# predictions_df = get_three_predictions(forecast_df)
# predictions_df.columns = ['pred_1', 'pred_2', 'pred_3']

# print(predictions_df)




