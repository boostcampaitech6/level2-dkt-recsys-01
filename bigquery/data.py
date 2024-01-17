from auth.client import client
from google.cloud import bigquery
import time
import pandas as pd

def read_dataframe_with_features(
        *features: tuple[str], 
        table_id: str = 'dkt-recsys01.dataset.dkt_train'
        ) -> pd.DataFrame:
    '''
    입력한 features 값들을 BigQuery 테이블에서 조회해 데이터프레임으로 반환합니다.

    Parameters:
    - features (tuple[str]): 조회하길 원하는 테이블 열의 튜플
    - table_id (str): 조회 대상 BigQuery 테이블 ID

    Returns:
    - pd.DataFrame: 조회 결과 데이터의 데이터프레임

    Notes:
    - features가 테이블에 존재하지 않는 column일 경우 ValueError
    '''
    QUERY = f"SELECT {__as_fields(features, table_id)} FROM {table_id}"
    
    start_time = time.time()
    print(f">> {start_time}: {QUERY}")
    result = client.query(QUERY).to_dataframe()  #  API request and Waits for query to finish
    print(f">> Query Process Time: {time.time() - start_time} secs for {result.shape[0]} rows")

    return result

def __select_current_table(table_id: str = 'dkt-recsys01.dataset.dkt_train'):
    QUERY = f"SELECT * FROM {table_id}"

    start_time = time.time()
    print(f">> {start_time}: {QUERY}")
    result = client.query(QUERY).to_dataframe()  #  API request and Waits for query to finish
    print(f">> Query Process Time: {time.time() - start_time} secs for {result.shape[0]} rows")

    return result

def add_new_feature_data(
        data_frame: pd.DataFrame,
        feature_name: str,
        feature_type: str,
        table_id: str = 'dkt-recsys01.dataset.dkt_train',
        write_disposition: str = 'WRITE_TRUNCATE'
        ) -> None:
    '''
    data_frame의 feature_name 데이터를 BigQuery에 feature_type 타입으로 추가합니다.

    Parameters:
    - data_frame (pd.DataFrame): 테이블에 로드하길 바라는 데이터가 포함된 데이터프레임
    - feature_name (str): 테이블에 로드하길 바라는 열의 이름
    - feature_type (str): 테이블에 로드할 열의 데이터 타입
    - table_id (str): BigQuery에 로드할 대상 테이블 ID
    - write_disposition (str): 타겟 테이블이 이미 존재할 경우 처리 방식 지정
        - https://cloud.google.com/bigquery/docs/reference/auditlogs/rest/Shared.Types/BigQueryAuditMetadata.WriteDisposition

    Returns:
    None: 없음

    Notes:
    - 기존 테이블 데이터를 조회한 뒤 새로운 열 데이터를 concat해 기존 테이블 truncate 후 재생성하는 방식
        - 열 방향 append가 불가능 함
    - data_frame이 없을 경우 ValueError
    - feature_name 이 data_frame 의 columns에 없을 경우 ValueError
    - feature_type 이 BigQuery에서 지원하지 않는 타입일 경우 ValueError 
        - https://cloud.google.com/bigquery/docs/reference/standard-sql/data-types#data_type_sizes
    '''

    if data_frame is None:
        raise ValueError(f"Input DataFrame should not be empty!")
    
    if feature_name not in data_frame.columns:
        raise ValueError(f"Input DataFrame should include '{feature_name}' column!")
    
    prev_data = __select_current_table(table_id)

    new_data = pd.concat([prev_data, data_frame[feature_name]], axis=1)
    
    job_config = bigquery.LoadJobConfig(
        schema = __new_schema(feature_name, __mapped_type(data_frame, feature_name, feature_type)), 
        write_disposition = write_disposition
        )

    job = client.load_table_from_dataframe(new_data, table_id, job_config = job_config)
    job.result()

    table = client.get_table(table_id)
    print(f">> Loaded {table.num_rows} rows and {len(table.schema)} columns to {table_id}")

def __new_schema(feature_name: str, feature_type: str) -> list[bigquery.SchemaField]:
    return [bigquery.SchemaField(feature_name, __valid_type(feature_type))]

def __mapped_type(data_frame: pd.DataFrame, feature_name: str, feature_type: str) -> str:
    dtype = data_frame.dtypes[feature_name].name
    TYPE_MAP = {
        'int16': 'NUMERIC',
        'object': 'STRING',
        'int8': 'INT64',
        'datetime64[ns]': 'STRING'
    }
    return TYPE_MAP[dtype] if dtype in TYPE_MAP else feature_type

def __valid_type(feature_type: str) -> str:
    VALID_TYPES = {
        'ARRAY': "The sum of the size of its elements. For example, an array defined as (ARRAY<INT64>) that contains 4 entries is calculated as 32 logical bytes (4 entries x 8 logical bytes).",
        'BIGNUMERIC': "32 logical bytes",
        'BOOL': "1 logical byte",
        'BYTES': "2 logical bytes + the number of logical bytes in the value",
        'DATE': "8 logical bytes",
        'DATETIME': "8 logical bytes",
        'FLOAT64': "8 logical bytes",
        'GEOGRAPHY': "16 logical bytes + 24 logical bytes * the number of vertices in the geography type.",
        'INT64': "8 logical bytes",
        'INTERVAL': "16 logical bytes",
        'JSON': "The number of logical bytes in UTF-8 encoding of the JSON-formatted string equivalent after canonicalization.",
        'NUMERIC': "16 logical bytes",
        'STRING': "2 logical bytes + the UTF-8 encoded string size",
        'STRUCT': "0 logical bytes + the size of the contained fields",
        'TIME': "8 logical bytes",
        'TIMESTAMP': "8 logical bytes"
    }

    if feature_type not in VALID_TYPES:
        message = '\n'.join([f"{key} : {value}" for key, value in VALID_TYPES.items()])
        raise ValueError(f"Type should be in below list:\n{message}")
    
    return feature_type

def __as_fields(features: tuple[str], table_id: str) -> str:
    __validate_fields(features, table_id)
    return __joined(features)

def __joined(features: tuple[str], delimeter: str = ', ') -> str:
    return delimeter.join(features)

def __validate_fields(features: tuple[str], table_id: str) -> None:
    valid_fields = __valid_fields(table_id)
    __validate_features(valid_fields, features)

def __validate_features(valid_features: list[str], features: tuple[str]) -> None:
    for feature in features:
        __validate_feature(valid_features, feature)

def __validate_feature(valid_features: list[str], feature: str) -> None:
    if feature not in valid_features:
        raise ValueError(f"'{feature}' column is not in table schema {valid_features}")

def __valid_fields(table_id: str) -> list[str]:
    columns = [field.name for field in __schema_of_table(table_id)]
    return columns

def __schema_of_table(table_id: str) -> list[bigquery.SchemaField]:
    table = client.get_table(table_id)
    return table.schema
