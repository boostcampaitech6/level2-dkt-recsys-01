from auth.client import client

print("##### 3. Execute query")
QUERY = (
    'SELECT name FROM `bigquery-public-data.usa_names.usa_1910_2013` '
    'WHERE state = "TX" '
    'LIMIT 100')
query_job = client.query(QUERY)  # API request
result = query_job.to_dataframe()  # Waits for query to finish
print(result.head())
