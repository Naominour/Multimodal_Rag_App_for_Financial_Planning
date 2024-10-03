# Defult arguments for DAG
defult_arguments = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 10, 3),
    'email_on_success': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
}