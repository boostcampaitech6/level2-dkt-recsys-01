from auth.client import client
import time
import pandas as pd

def read_dataframe_with_features(*features: tuple[str], table_id: str = 'dkt-recsys01.dataset.dkt_train') -> pd.DataFrame:
    QUERY = f"SELECT {__as_fields(features, table_id)} FROM {table_id}"
    
    start_time = time.time()
    print(f">> {start_time}: {QUERY}")
    result = client.query(QUERY).to_dataframe()  #  API request and Waits for query to finish
    print(f">> Query Process Time: {time.time() - start_time} secs for {result.shape[0]} rows")

    return result

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
    table = client.get_table(table_id)
    schema = table.schema
    columns = [field.name for field in schema]
    return columns
