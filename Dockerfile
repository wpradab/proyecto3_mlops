# Usar la imagen base de Python
FROM python:3.8

# Comando por defecto para ejecutar al iniciar el contenedor
CMD  mlflow server --host 0.0.0.0 --port 1880 --backend-store-uri sqlite:///mlflow.db --artifacts-destination /mlflow/artifacts
