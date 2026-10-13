# Azure ML / MLflow implementation notes

## Workspace setup
Create a resource group and Azure Machine Learning workspace in Azure Portal. A workspace provisions or links the required supporting resources. Confirm region and quota before creating compute.

## Authentication and tracking
Install Azure CLI, run `az login`, and set `MLFLOW_TRACKING_URI` to the tracking URI shown for your Azure ML workspace. Confirm that the logged-in identity has access to the workspace. Run the same `src.train` command; MLflow should send parameters, metrics, model artifacts, and plots to the workspace.

## Registry and endpoint
After comparing runs, register the selected model from the Azure ML Studio model/MLflow interface or with the Azure ML SDK. Deploy a managed online endpoint only after verifying the registered artifact, input schema, inference environment, permissions, and quota. Keep endpoint authentication enabled. Test the endpoint with a small request and record latency/errors. Delete the endpoint and unused compute when finished to avoid ongoing charges.

The exact SDK commands depend on workspace details and current Azure ML SDK version, so record your workspace name, resource group, region, registered model name/version, endpoint name, and deployment instance type before writing the deployment script.
