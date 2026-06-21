import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
events = pd.read_csv('ecommerce_recommendation_events.csv')
print(events.head(10))
print(events.shape)
print(events.columns.tolist())
print(events.dtypes)
events['converted'] = events['event_type'].eq('purchase')
print(events['converted'].describe())
total_revenue = events['revenue'].sum()
total_costs = events['total_cost'].sum()
total_profit = events['profit'].sum()
avg_profit_per_conversion = events.loc[events['converted'], 'profit'].mean()

print("Total revenue:", total_revenue)
print("Total costs:", total_costs)
print("Total profit:", total_profit)
print("Average profit per conversion:", avg_profit_per_conversion)
rec_vs_nonrec = events.groupby('recommended').agg(
    conversion_rate=('converted', 'mean'),
    avg_profit_per_event=('profit', 'mean')
).rename(index={0: 'not_recommended', 1: 'recommended'})

print(rec_vs_nonrec)
model_metrics = events.groupby('model_version').agg(
    event_count=('converted', 'size'),
    conversion_rate=('converted', 'mean'),
    revenue_sum=('revenue', 'sum'),
    inference_cost_sum=('inference_cost', 'sum'),
    product_cost_sum=('product_cost', 'sum'),
    profit_sum=('profit', 'sum')
).reset_index()

model_metrics['total_costs'] = model_metrics['inference_cost_sum'] + model_metrics['product_cost_sum']
model_metrics['ai_roi_pct'] = np.where(
    model_metrics['inference_cost_sum'] > 0,
    model_metrics['profit_sum'] / model_metrics['inference_cost_sum'] * 100,
    np.nan
)
model_metrics['overall_roi_pct'] = np.where(
    model_metrics['total_costs'] > 0,
    model_metrics['profit_sum'] / model_metrics['total_costs'] * 100,
    np.nan
)

print(model_metrics)
recommended_only = events[events['recommended'].eq(1)].copy()

recommended_version_metrics = recommended_only.groupby('model_version').agg(
    conversion_rate=('converted', 'mean'),
    avg_profit_per_conversion=('profit', lambda s: recommended_only.loc[s.index & recommended_only.index, 'profit'].mean() if len(s) else np.nan)
).reset_index()

print(recommended_version_metrics)
overall_rec_rate = events['recommended'].mean()

region_rates = events.groupby('region')['recommended'].mean().reset_index(name='recommendation_rate')
region_rates['diff_from_overall_pct'] = (
    (region_rates['recommendation_rate'] - overall_rec_rate).abs() / overall_rec_rate
) * 100
region_rates['flag_over_10pct'] = region_rates['diff_from_overall_pct'] > 10

print(region_rates)

region_table = pd.crosstab(events['region'], events['recommended'])
chi2_region = chi2_contingency(region_table)

print("Chi-square statistic:", chi2_region[0])
print("p-value:", chi2_region[1])
region_perf = events.groupby(['region', 'recommended']).agg(
    conversion_rate=('converted', 'mean'),
    avg_profit_per_event=('profit', 'mean')
).reset_index()

print(region_perf)
age_rates = events.groupby('age')['recommended'].mean().reset_index(name='recommendation_rate')
age_rates['diff_from_overall_pct'] = (
    (age_rates['recommendation_rate'] - overall_rec_rate).abs() / overall_rec_rate
) * 100
age_rates['flag_over_10pct'] = age_rates['diff_from_overall_pct'] > 10

print(age_rates)

age_table = pd.crosstab(events['age'], events['recommended'])
chi2_age = chi2_contingency(age_table)

print("Chi-square statistic:", chi2_age[0])
print("p-value:", chi2_age[1])
segment_metrics = events.groupby('customer_segment').agg(
    conversion_rate=('converted', 'mean'),
    avg_revenue_per_conversion=('revenue', lambda s: events.loc[s.index[events.loc[s.index, 'converted']], 'revenue'].mean()),
    avg_profit_per_conversion=('profit', lambda s: events.loc[s.index[events.loc[s.index, 'converted']], 'profit'].mean()),
    total_profit=('profit', 'sum')
).reset_index()

segment_metrics['profit_pct_of_total'] = segment_metrics['total_profit'] / events['profit'].sum() * 100

print(segment_metrics)
segment_total_profit = events.groupby('customer_segment')['profit'].sum().reset_index(name='total_profit')
segment_total_profit['profit_pct_of_total'] = segment_total_profit['total_profit'] / events['profit'].sum() * 100

print(segment_total_profit)
model_segment_metrics = events.groupby(['model_version', 'customer_segment']).agg(
    total_profit=('profit', 'sum'),
    inference_cost_sum=('inference_cost', 'sum')
).reset_index()

model_segment_metrics['roi_pct'] = np.where(
    model_segment_metrics['inference_cost_sum'] > 0,
    model_segment_metrics['total_profit'] / model_segment_metrics['inference_cost_sum'] * 100,
    np.nan
)

print(model_segment_metrics)
roi_pivot = model_segment_metrics.pivot(
    index='model_version',
    columns='customer_segment',
    values='roi_pct'
)

print(roi_pivot)
age_v1 = pd.crosstab(v1['age'], v1['recommended'])
age_v2 = pd.crosstab(v2['age'], v2['recommended'])

print("v1.0 age table:")
print(age_v1)
print("v2.0 age table:")
print(age_v2)

print("v1.0 chi-square:")
print(chi2_contingency(age_v1))
print("v2.0 chi-square:")
print(chi2_contingency(age_v2))
summary_report = model_metrics[['model_version', 'conversion_rate', 'ai_roi_pct', 'overall_roi_pct', 'profit_sum']].copy()
summary_report = summary_report.rename(columns={'profit_sum': 'total_profit'})

print(summary_report)
