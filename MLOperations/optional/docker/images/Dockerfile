FROM b4ebf977fa1b4412adb84bc5f3bef665.azurecr.io/azureml/azureml_ff2988e7393ebc1b241bc1c9efecbaba
COPY azureml-app /var/azureml-app
RUN mkdir -p '/var/azureml-app' && echo eyJhY2NvdW50Q29udGV4dCI6eyJzdWJzY3JpcHRpb25JZCI6ImRhZmQ2MmZlLTUxZDMtNDY3MS1hZjUwLTZiYWMyZjAwMWI1ZCIsInJlc291cmNlR3JvdXBOYW1lIjoiYW1sLXF1aWNrc3RhcnRzLTEyODEwOCIsImFjY291bnROYW1lIjoicXVpY2stc3RhcnRzLXdzLTEyODEwOCIsIndvcmtzcGFjZUlkIjoiYjRlYmY5NzctZmExYi00NDEyLWFkYjgtNGJjNWYzYmVmNjY1In0sIm1vZGVscyI6e30sIm1vZGVsc0luZm8iOnt9fQ== | base64 --decode > /var/azureml-app/model_config_map.json
RUN mv '/var/azureml-app/tmpi8mix55y.py' /var/azureml-app/main.py
CMD ["runsvdir","/var/runit"]
