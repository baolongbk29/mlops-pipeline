from pathlib import Path

import pendulum
from airflow.models import Variable
from docker.types import Mount



class AppConst:
    DOCKER_USER = Variable.get("DOCKER_USER", "longlam071")


class AppPath:
    MLOPS_PIPELINE_DIR = Path(Variable.get("MLOPS_CODE_DIR"))
    ARTIFACTS = MLOPS_PIPELINE_DIR / "artifacts"


class DefaultConfig:
    DEFAULT_DAG_ARGS = {
        "owner": "longlam071",
        "retries": 0,
        "retry_delay": pendulum.duration(seconds=20),
    }

    DEFAULT_DOCKER_OPERATOR_ARGS = {
        "image": "longlam071/mlops_pipeline:latest",
        "api_version": "auto",
        "auto_remove": True,
        "network_mode": "bridge",
        "docker_url": "tcp://docker-proxy:2375",
        # "mounts": [
        #     # artifacts
        #     # Mount(
        #     #     source=AppPath.ARTIFACTS.absolute().as_posix(),
        #     #     target="/mlops-pipeline/artifacts",
        #     #     type="bind",
        #     # ),
        # ],
    }