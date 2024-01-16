from google.cloud import bigquery
from client import client
import pandas as pd

table_id = 'dkt-recsys01.dataset.dkt_train'
df = pd.read_csv('/opt/ml/input/data/train_data.csv')

print(df.dtypes)

schema = [
    bigquery.SchemaField('userID', 'INTEGER'),
    bigquery.SchemaField('assessmentItemID', 'STRING'),
    bigquery.SchemaField('testId', 'STRING'),
    bigquery.SchemaField('answerCode', 'INTEGER'),
    bigquery.SchemaField('Timestamp', 'STRING'),
    bigquery.SchemaField('KnowledgeTag', 'INTEGER'),
]

job_config = bigquery.LoadJobConfig(schema=schema, write_disposition='WRITE_TRUNCATE')

job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
job.result()

table = client.get_table(table_id)
print(f"Loaded {table.num_rows} rows and {len(table.schema)} columns to {table_id}")

