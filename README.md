# the-beginning-of-the-way
#we answer questions:
#1 How many users do we have who made a purchase only once?
#2 How many orders per month, on average, are not delivered for various reasons?
#3 For each product, determine on which day of the week the product is most often bought.
#4 How many purchases does each user make on average per week (by months)?
#5 Conduct a cohort analysis of users. Between January and December, identify the cohort with the highest retention for the 3rd month.
#6 Build RFM-segmentation of users in order to qualitatively evaluate the audience.

# importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime

#uploading and merging data
customers = pd.read_csv('/mnt/_dataset.csv')
orders = pd.read_csv('/mnt/_dataset.csv')
items = pd.read_csv('/mnt/_dataset.csv')
df = orders.merge(customers, on='customer_id', how='left')
df = df.merge(items, on='order_id', how='left')

#change data type to datetime
df['order_purchase_timestamp'] = pd.to_datetime(df.order_purchase_timestamp)
df['order_approved_at'] = pd.to_datetime(df.order_approved_at)
df['order_delivered_carrier_date'] = pd.to_datetime(df.order_delivered_carrier_date)
df['order_delivered_customer_date'] = pd.to_datetime(df.order_delivered_customer_date)
df['order_estimated_delivery_date'] = pd.to_datetime(df.order_estimated_delivery_date)
df['shipping_limit_date'] = pd.to_datetime(df.shipping_limit_date)

#we count the number of users who made a purchase once
order_count_by_customers = df.query("order_status=='delivered'") \
  .groupby('customer_unique_id', as_index=False) \
  .agg({'order_id':'nunique'}) \
  .sort_values('order_id', ascending=False) \
  .rename(columns={'order_id':'order_count'})
customers_count_with_one_order = order_count_by_customers.query('order_count == 1').count().order_count
order_count_sum = order_count_by_customers.order_count.sum()
customers_count_with_one_order
customers_with_1_order_percent = 100 * customers_count_with_one_order / order_count_sum
customers_with_1_order_percent.round(2)

#see how many orders are not delivered
orders.head(1)
orders['order_purchase_yearmonth'] = pd.to_datetime(orders.order_purchase_timestamp).dt.strftime('%Y-%m')
orders_not_delivered = orders.query('order_status != "delivered"') \
    .groupby(['order_purchase_yearmonth', 'order_status'], as_index=False) \
    .agg({'order_id':'count'}) \
    .rename(columns={'order_id':'order_count'})
    
#There is no data in the dataframe for two months - November and December 2016 => the average is considered incorrect, in the lines with these months there should be 0 in the order_count column
orders_not_delivered.groupby('order_purchase_yearmonth').agg({'order_count':'sum'})
orders_not_delivered_mean_by_months = orders_not_delivered.groupby('order_purchase_yearmonth', as_index=False).agg({'order_count':'sum'}).order_count.mean()
orders_not_delivered_mean_by_months
orders.query('order_purchase_timestamp > "2016-11-01 00:00:00" & order_purchase_timestamp < "2017-01-01 00:00:00"')

#We are trying to fix this shortcoming
orders.order_purchase_yearmonth.max()
orders.order_purchase_yearmonth.min()
number_of_month_observed = (pd.to_datetime(orders.order_purchase_yearmonth.max())-pd.to_datetime(orders.order_purchase_yearmonth.min()))/ np.timedelta64(1,'M')
orders_not_delivered_mean_by_status = orders_not_delivered.groupby('order_status', as_index=False).agg({'order_count':'sum'})
orders_not_delivered_mean_by_month = orders_not_delivered_mean_by_status.order_count.sum() / number_of_month_observed
orders_not_delivered_mean_by_month
orders_not_delivered_mean_by_status['order_status_mean_by_month'] = orders_not_delivered_mean_by_status.order_count / number_of_month_observed
orders_not_delivered_mean_by_status

#we check that the average "general" and the average by status converge
orders_not_delivered_mean_by_status.order_status_mean_by_month.sum()

#visualize
sns.set(rc={'figure.figsize':(20,10)}, font_scale=2)
graph = sns.barplot(data=orders_not_delivered_mean_by_status, x='order_status', y='order_status_mean_by_month')
graph.set_ylabel('Average orders cancelled per month')
graph.set_xlabel('Order status')

#add a column with the day of the week on which the order is made to the general dataframe
df['purchase_weekday'] = df.order_purchase_timestamp.dt.day_name()

#grouping by products and days of the week
weekday_of_success = df.query('order_status == "delivered"') \
    .groupby(['product_id', 'purchase_weekday'], as_index=False) \
    .agg({'order_id':'count'}) \
    .rename(columns={'order_id':'purchase_count'})
    
#we select from the dataframe only rows with the maximum purchase_count value
weekday_of_success = weekday_of_success.loc[weekday_of_success.groupby('product_id')['purchase_count'].idxmax()]
weekday_of_success = weekday_of_success.sort_values('purchase_count', ascending=False).reset_index(drop=True)
weekday_of_success.head()

#we calculate the average number of purchases per week for each user and the number of weeks in a month
df['order_purchase_yearmonth'] = pd.to_datetime(df.order_purchase_timestamp).dt.strftime('%Y-%m')
df['weeks_in_month'] = df.order_purchase_timestamp.dt.days_in_month / 7

#we count the number of purchases per month for the user
week_mean =  df.query('order_status == "delivered"') \
    .groupby(['customer_unique_id','order_purchase_yearmonth','weeks_in_month', 'order_id'], as_index=False) \
    .agg({'order_purchase_timestamp':'count'}) \
    .rename(columns={'order_purchase_timestamp':'order_count'}) \
    .groupby(['customer_unique_id', 'order_purchase_yearmonth', 'weeks_in_month'], as_index=False) \
    .agg({'order_count':'sum'})
week_mean.sort_values('order_count')

#divide by the number of weeks in a month to get the number of purchases per week
week_mean['average_weekly_order_count'] = (week_mean.order_count / week_mean.weeks_in_month).round(2)

#expand the date frame to see which months users are active
pivot = week_mean.pivot(index='customer_unique_id', columns='order_purchase_yearmonth', values='average_weekly_order_count')
pivot = pivot.fillna(0)
pivot.head()

#calculation of the average number of purchases per week during "active" months for each user
week_mean = week_mean.groupby('customer_unique_id', as_index=False).agg({'average_weekly_order_count':'mean'})

#we do a cohort analysis. Consider both month and year
cohorting = df.query('order_status == "delivered"') \
    .groupby('customer_unique_id', as_index=False) \
    .agg({'order_purchase_timestamp':'min'})
cohorting['first_order'] = pd.to_datetime(cohorting.order_purchase_timestamp).dt.strftime('%Y-%m')
cohorting = cohorting.sort_values('first_order')
cohorting.head()

#let's add the indicator cohort_rank - in what month since the launch of the project the user made the first purchase. And eliminate the misunderstanding with the missing months
cohorting_rank = cohorting.groupby('first_order', as_index=False).agg({'customer_unique_id':'count'})
cohorting_rank.iloc[2] = ['2016-11', '0']
cohorting_rank['cohort_rank'] = cohorting_rank.first_order.rank()
cohorting = cohorting.merge(cohorting_rank[['first_order','cohort_rank']], on='first_order', how='left')
cohorting.head()

#add cohorts to a common dataframe
df = df.merge(cohorting[['customer_unique_id', 'first_order', 'cohort_rank']], on='customer_unique_id', how='left')
retention_3d_month = df.query('order_status == "delivered"') \
        .groupby(['cohort_rank', 'first_order','order_purchase_yearmonth'], as_index=False) \
        .agg({'customer_unique_id':'nunique'})

#we count the difference from the order date to the first order
retention_3d_month['first_order'] =retention_3d_month.first_order.to_numpy().astype('datetime64[M]')
retention_3d_month['order_purchase_yearmonth'] =retention_3d_month.order_purchase_yearmonth.to_numpy().astype('datetime64[M]')
retention_3d_month['date_dif'] = ((retention_3d_month.order_purchase_yearmonth - retention_3d_month.first_order) / np.timedelta64(1,'M')).round()
retention_3d_month.head()

#calculate retention_rate
retention_base = retention_3d_month.query('first_order == order_purchase_yearmonth').rename(columns={'customer_unique_id':'retention_base'})
retention_3d_month = retention_3d_month.merge(retention_base[['cohort_rank', 'retention_base']], on='cohort_rank', how='left')
retention_3d_month['retention_rate'] = 100 * retention_3d_month.customer_unique_id / retention_3d_month.retention_base

#calculate the highest retention rate for the third month
retention_3d_month = retention_3d_month.set_index('first_order')
retention_3d_month.query('date_dif == 3').retention_rate.idxmax()
retention_3d_month.query('date_dif == 3').retention_rate.max()

#visualize the received data
retention_3d_month.loc[retention_3d_month['retention_rate'] == 100, 'retention_rate'] = 0
retention_3d_month.head()
pivot = retention_3d_month.pivot(index = 'cohort_rank', columns='date_dif', values='retention_rate').fillna(0)
pivot

plt.rcParams['font.size'] = '11'
plt.figure(figsize=(18,14))
plt.title('Users Active')
ax = sns.heatmap(data=pivot, annot=True, vmin=0.0,vmax=0.5 ,cmap='Reds')
ax.set_yticklabels(pivot.index)
ax.set_ylabel('Month from start')
ax.set_xlabel('Months group active')
ax

#foreach user, we count the number of days from the moment of the last order to today (for today, I took the date of the last order in the dataframe)
recency = df.query('order_status == "delivered"').groupby(['customer_unique_id'], as_index=False).agg({'order_purchase_timestamp':'max'})
recency['recency'] = (recency.order_purchase_timestamp.max() - recency.order_purchase_timestamp).dt.days

#for us, users who have made a purchase recently are of more interest, since there is a greater chance of retaining them. Based on this, we divide into groups.
recency['R'] = pd.cut(recency.recency, [0, 30, 90, 180, 300, np.inf], labels = [1,2,3,4,5])
recency['R_description'] = pd.cut(recency.recency, [0, 30, 90, 180, 300, np.in
recency.groupby('R').agg({'customer_unique_id':'count'})
sns.distplot(recency.recency)

#For each user, we count the number of purchases made over the entire time. For good, you need to count the number of orders for a certain period of time (week, month, etc.). But in our case, 93% of users have 1 order, so it will not be representative
frequency = df.query('order_status == "delivered"').groupby('customer_unique_id', as_index=False).agg({'order_id':'nunique'}).rename(columns={'order_id':'order_count'})
frequency['F'] = pd.cut(frequency.order_count, [0.0, 1.0, 2.0, 3.0, 5.0, 100.0], labels=[5,4,3,2,1])
frequency['F_description'] = pd.cut(frequency.order_count, [0.0, 1.0, 2.0, 3.0, 5.0, 100.0])
frequency.groupby('F', as_index=False).agg({'customer_unique_id':'count'})

#let's calculate the average check for each user
monetary = df.query('order_status=="delivered"').groupby('customer_unique_id', as_index=False).agg({'price':'sum', 'order_id':'nunique'})
monetary['monetary'] = monetary.price / monetary.order_id
monetary.head()
monetary['M'] = pd.qcut(monetary.price, [.0, 0.5, 0.7, 0.85, 0.95, 1.0], labels=[5,4,3,2,1])
monetary['M_description'] = pd.qcut(monetary.price, [.0, 0.5, 0.7, 0.85, 0.95, 1.0])
monetary.groupby('M').agg({'customer_unique_id':'count'})
RFM = recency[['customer_unique_id', 'R', 'R_description']].merge(frequency[['customer_unique_id', 'F', 'F_description']], on='customer_unique_id', how='left')
RFM = RFM.merge(monetary[['customer_unique_id' , 'M', 'M_description']], on='customer_unique_id', how='left')
RFM



