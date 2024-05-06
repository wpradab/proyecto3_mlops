from datetime import datetime, timedelta
import mlflow
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy_operator import DummyOperator
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from ucimlrepo import fetch_ucirepo
import pandas as pd

# Establece el URI de seguimiento de MLflow y el nombre del experimento
mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("My_Model_Experiment")

def fetch_dataset(**kwargs):
    data = fetch_ucirepo(id=296)
    X = data.data.features
    y = data.data.targets
    df = pd.DataFrame(X)
    df['readmitted'] = y
    kwargs['ti'].xcom_push(key='dataframe', value=df)

def eliminar_columnas_faltantes(df, umbral=0.1):
    max_valores_faltantes = len(df) * umbral
    columnas_validas = df.columns[df.isnull().sum() < max_valores_faltantes]
    return df[columnas_validas].dropna()

def preprocess_data(**kwargs):
    ti = kwargs['ti']
    dataset = ti.xcom_pull(task_ids='fetch_dataset', key='dataframe')
    df2 = eliminar_columnas_faltantes(dataset)
    df2['readmitted'] = df2['readmitted'].replace(['>30', 'NO', '<30'], [0, 1, 2])
    columnas_a_eliminar = ["acetohexamide", "chlorpropamide", "tolbutamide", "miglitol", "troglitazone", "tolazamide", "examide", "citoglipton", "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone"]
    df2 = df2.drop(columns=columnas_a_eliminar, errors='ignore')
    df3 = df2.select_dtypes(include=['number'])
    df3.to_csv("/opt/airflow/data/data_diabetes.csv", index=False)
    ti.xcom_push(key='preprocessed_data', value=df3)

def train_model(**kwargs):
    ti = kwargs['ti']
    df3 = pd.read_csv("/opt/airflow/data/data_diabetes.csv")
    y3 = df3.pop('readmitted')
    X_train, X_test, y_train, y_test = train_test_split(df3, y3, test_size=0.2, random_state=42)
    mlflow.sklearn.autolog()
    with mlflow.start_run(run_name="LogisticRegression"):
        logistic_classifier = LogisticRegression(multi_class='multinomial', solver='lbfgs')
        logistic_classifier.fit(X_train, y_train)
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model_cancer"
        mlflow.sklearn.log_model(logistic_classifier, "DiabetesLogisticRegressionModel")

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 5, 1),
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'diabetes_prediction',
    default_args=default_args,
    description='Predict diabetes readmission',
    schedule_interval=timedelta(days=1),
)

start_task = DummyOperator(task_id='start_task', dag=dag)
fetch_data = PythonOperator(task_id='fetch_dataset', python_callable=fetch_dataset, provide_context=True, dag=dag)
preprocess_data = PythonOperator(task_id='preprocess_data', python_callable=preprocess_data, provide_context=True, dag=dag)
train_model = PythonOperator(task_id='train_model', python_callable=train_model, provide_context=True, dag=dag)
end_task = DummyOperator(task_id='end_task', dag=dag)

start_task >> fetch_data >> preprocess_data >> train_model >> end_task
