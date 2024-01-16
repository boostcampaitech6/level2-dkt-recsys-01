import glob
from google.cloud import bigquery
from google.oauth2 import service_account

key_files = glob.glob("./config/dkt-recsys01-gbq-private-key.json")

if not key_files:
    raise ValueError(f"Cannot find BigQuery credential key file.")

key_files = key_files[0]

# Credentials 객체 생성
print("##### 1. create credentials")
credentials = service_account.Credentials.from_service_account_file(key_files)

# GCP 클라이언트 객체 생성
print("##### 2. create GCP Client")
client = bigquery.Client(credentials=credentials, project=credentials.project_id)