#!/usr/bin/env python3
# pylint: disable=too-many-lines

"""
Standalone Distributed training script

This script provides a standalone version of the pipeline.py script, designed to be used when
Kubeflow pipelines are not available.

Usage:
    This script can be executed directly from the command line. Ensure that the Kubernetes client is
    properly configured before running the script.

Dependencies:
    kubernetes: The Kubernetes Python client library.
    click: A package for creating command-line interfaces.

TODO:
    - Make sure resources get cleaned up after the job is done. (configmap, secret etc) using a
      finalizer.
    - See if we can use KServe to deploy the model and serve it for SDG Data Generation.
      kubernetes_yaml/mixtral_serve/mixtral_serve.yaml
"""

import base64
import json
import logging
import os
import time
import typing
from ast import literal_eval
from os import path
from urllib.parse import urlparse

import click
import kubernetes
import kubernetes.client
import kubernetes.client.exceptions
import kubernetes.client.rest
import kubernetes.config
import kubernetes.utils
import kubernetes.watch
import urllib3.exceptions
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(asctime)s %(name)s:%(lineno)d: %(message)s",
)
logger = logging.getLogger(__name__)

# IMAGES
DS_IMAGE = "quay.io/opendatahub/workbench-images:jupyter-datascience-ubi9-python-3.11-20241004-609ffb8"  # pylint: disable=line-too-long
RHELAI_IMAGE = "registry.redhat.io/rhelai1/instructlab-nvidia-rhel9:1.2"

# SDG
DEFAULT_REPO_URL = "https://github.com/instructlab/taxonomy.git"
SDG_OBJECT_STORE_SECRET_NAME = "sdg-object-store-credentials"
REPO_GRANITE_7B_IMAGE = "ibm-granite/granite-7b-base"  # used by HF downloader

# SDG DATA PREPROCESSING (before doing training, data has to be converted)
MAX_SEQ_LEN = 4096
MAX_BATCH_LEN = 20000

# DATA
DATA_PVC_NAME = "data"
DATA_PVC_MOUNT_PATH = "/data"
DATA_PVC_SDG_PATH = path.join(DATA_PVC_MOUNT_PATH, "data")
DATA_PVC_MODEL_PATH = path.join(DATA_PVC_MOUNT_PATH, "model")
DATA_VOLUME_NAME = "data"
TAXONOMY_PATH = path.join(DATA_PVC_MOUNT_PATH, "taxonomy")
DATA_PVC_OUTPUT_PATH = path.join(DATA_PVC_MOUNT_PATH, "output")
DATA_PVC_OUTPUT_DATA_PATH = path.join(DATA_PVC_OUTPUT_PATH, "data")
PREPROCESSED_DATA_PATH = path.join(DATA_PVC_SDG_PATH, "processed_data")
PREPROCESSED_DATA_SKILLS_PATH = path.join(PREPROCESSED_DATA_PATH, "skills")
PREPROCESSED_DATA_KNOWLEDGE_PATH = path.join(PREPROCESSED_DATA_PATH, "knowledge")
MT_BENCH_OUTPUT_PATH = path.join(DATA_PVC_MOUNT_PATH, "mt-bench-results.txt")
MT_BENCH_SCORES_PATH = path.join(DATA_PVC_MOUNT_PATH, "mt-bench-best.txt")
MT_BENCH_BRANCH_SCORES_PATH = path.join(DATA_PVC_MOUNT_PATH, "mt-bench-branch-best.txt")
MMLU_BRANCH_SCORES_PATH = path.join(DATA_PVC_MOUNT_PATH, "mmlu-branch-best.txt")
CANDIDATE_MODEL_PATH_PREFIX = path.join(
    DATA_PVC_MOUNT_PATH, "model/output/phase_2/hf_format"
)
CANDIDATE_MODEL_PATH = path.join(CANDIDATE_MODEL_PATH_PREFIX, "candidate_model")
SDG_GENERATED_DATA_PATH = path.join(DATA_PVC_MOUNT_PATH, "generated")
TAXONOMY_DATA_PATH = path.join(DATA_PVC_MOUNT_PATH, "taxonomy")
# MMLU_SCORES_PATH = "/output/mmlu-results.txt" - after training phase 1 is done MMLU is not performed anymore

# TRAINING
PYTORCH_NNODES = 2

# EVALUATION
EVAL_TYPE_MT_BENCH = "mt-bench"
EVAL_TYPE_FINAL = "final"
JUDGE_SERVING_NAME = "judge-serving-details"
MODEL_DTYPE = "bfloat16"
MAX_WORKERS = "auto"
MERGE_SYSTEM_USER_MESSAGE = False
FEW_SHOTS = 5
BATCH_SIZE = 8
JUDGE_CA_CERT_ENV_VAR_NAME = "JUDGE_CA_CERT_PATH"
JUDGE_CA_CERT_PATH = "/tmp/cert"
JUDGE_CA_CERT_CM_KEY = "ca-bundle.crt"

# TEMPLATES
PYTORCH_TRAINING_JOB = """
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: {name}
spec:
  nprocPerNode: \"{nproc_per_node}\"
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      restartPolicy: OnFailure
      template:
        metadata:
          annotations:
            sidecar.istio.io/inject: 'false'
        spec:
          containers:
            - args:
                - |
                  phase_num={phase_num}
                  PATH_TO_DATA={preprocessed_data_knowledge_path}
                  if [ "$phase_num" -eq 2 ]; then PATH_TO_DATA="{preprocessed_data_skills_path}"; fi
                  echo "Running phase $phase_num"
                  PATH_TO_MODEL={path_to_model}
                  if [ "$phase_num" -eq 2 ]; then PATH_TO_MODEL="{path_to_model}/output/phase_1/hf_format/$(ls --sort=time {path_to_model}/output/phase_1/hf_format|head -n 1)"; fi
                  echo "Using $PATH_TO_MODEL model for training"
                  echo "Using $PATH_TO_DATA data for training"
                  mkdir -p {data_pvc_model_path};
                  mkdir -p {data_pvc_sdg_path};
                  mkdir -p {path_to_model}/output/phase_{phase_num}
                  torchrun --nnodes {nnodes} \
                    --nproc_per_node {nproc_per_node} \
                    --node_rank $(RANK) \
                    --rdzv_endpoint $(MASTER_ADDR):$(MASTER_PORT) \
                    -m instructlab.training.main_ds \
                    --model_name_or_path="$PATH_TO_MODEL" \
                    --data_path="$PATH_TO_DATA"/data.jsonl \
                    --output_dir={path_to_model}/output/phase_{phase_num} \
                    --num_epochs={epoch_num} \
                    --effective_batch_size=3840 \
                    --learning_rate=1e-4 \
                    --num_warmup_steps=800 \
                    --save_samples=0 \
                    --log_level=INFO \
                    --max_batch_len=20000 \
                    --seed=42 \
                    --cpu_offload_optimizer \
                    --distributed_training_framework fsdp \
                    --cpu_offload_params \
                    --is_granite \
                    --checkpoint_at_epoch
              command:
                - /bin/bash
                - '-c'
                - '--'
              image: {image}
              name: pytorch
              volumeMounts:
                - mountPath: /data
                  name: data
              env:
                - name: NNODES
                  value: \"{nnodes}\"
                - name: NPROC_PER_NODE
                  value: \"{nproc_per_node}\"
                - name: XDG_CACHE_HOME
                  value: /tmp
                - name: TRITON_CACHE_DIR
                  value: /tmp
                - name: HF_HOME
                  value: /tmp
                - name: TRANSFORMERS_CACHE
                  value: /tmp
              resources:
                requests:
                  cpu: 2
                  "nvidia.com/gpu": {nproc_per_node}
                limits:
                  cpu: 2
                  "nvidia.com/gpu": {nproc_per_node}
          volumes:
            - name: data
              persistentVolumeClaim:
                claimName: {data_pvc_name}
    Worker:
      replicas: {worker_replicas}
      restartPolicy: OnFailure
      template:
        metadata:
          annotations:
            sidecar.istio.io/inject: 'false'
        spec:
          containers:
            - args:
                - |
                  phase_num={phase_num}
                  PATH_TO_DATA={preprocessed_data_knowledge_path}
                  if [ "$phase_num" -eq 2 ]; then PATH_TO_DATA="{preprocessed_data_skills_path}"; fi
                  echo "Running phase $phase_num"
                  PATH_TO_MODEL={path_to_model}
                  if [ "$phase_num" -eq 2 ]; then PATH_TO_MODEL="{path_to_model}/output/phase_1/hf_format/$(ls --sort=time {path_to_model}/output/phase_1/hf_format|head -n 1)"; fi
                  echo "Using $PATH_TO_MODEL model for training"
                  echo "Using $PATH_TO_DATA data for training"
                  tmp_model=$(mktemp -d)
                  mkdir -p "$tmp_model";
                  torchrun --nnodes {nnodes} \
                    --nproc_per_node {nproc_per_node} \
                    --node_rank $(RANK) \
                    --rdzv_endpoint $(MASTER_ADDR):$(MASTER_PORT) \
                    -m instructlab.training.main_ds \
                    --model_name_or_path="$PATH_TO_MODEL" \
                    --data_path="$PATH_TO_DATA"/data.jsonl \
                    --output_dir="$tmp_model" \
                    --num_epochs={epoch_num} \
                    --effective_batch_size=3840 \
                    --learning_rate=1e-4 \
                    --num_warmup_steps=800 \
                    --save_samples=0 \
                    --log_level=INFO \
                    --max_batch_len=20000 \
                    --seed=42 \
                    --cpu_offload_optimizer \
                    --distributed_training_framework fsdp \
                    --cpu_offload_params \
                    --is_granite \
                    --checkpoint_at_epoch
              command:
                - /bin/bash
                - '-c'
                - '--'
              image: {image}
              name: pytorch
              volumeMounts:
                - mountPath: /data
                  name: data
              env:
                - name: NNODES
                  value: \"{nnodes}\"
                - name: NPROC_PER_NODE
                  value: \"{nproc_per_node}\"
                - name: XDG_CACHE_HOME
                  value: /tmp
                - name: TRITON_CACHE_DIR
                  value: /tmp
                - name: HF_HOME
                  value: /tmp
                - name: TRANSFORMERS_CACHE
                  value: /tmp
              resources:
                requests:
                  cpu: 2
                  "nvidia.com/gpu": {nproc_per_node}
                limits:
                  cpu: 2
                  "nvidia.com/gpu": {nproc_per_node}
          volumes:
            - name: data
              persistentVolumeClaim:
                claimName: {data_pvc_name}
"""

DATA_SCRIPT = """
set -e

export STRATEGY={strategy}

if [ -z "$STRATEGY" ] || [ "$STRATEGY" == "None" ]; then
    echo "STRATEGY is not set - must be 'download' or 'upload'"
    exit 1
fi

if [ "$STRATEGY" == "download" ]; then
    FORCE_PULL={force_pull}
    if [ -s {data_pvc_mount_path}/data.tar.gz ] && [ -d {data_pvc_mount_path}/data ] && [ -d {data_pvc_mount_path}/model ] ; then
        echo "Data tarball and sdg/model directories already exist in the PVC. Skipping download."
        if [ "$FORCE_PULL" == "None" ] || [ "$FORCE_PULL" == "False" ]; then
            echo "'--force-pull' is not set - will not force pull the data from the object store"
            ls -laR {data_pvc_mount_path}
            exit 0
        else
            echo "'--force-pull' is set to true - will force pull the data from the object store"
        fi
    fi

    if python3 -c 'import boto3'; then
        echo 'boto3 is already installed'
    else
        if ! [ -x "$(command -v pip)" ]; then
            python3 -m ensurepip || python3 -m ensurepip --user || dnf install python3-pip -y
        fi
        python3 -m pip install boto3
    fi
fi

if [ "$STRATEGY" == "upload" ]; then
    export FINAL_DATA_TAR_FILE="$(date +"%Y-%m-%d_%H-%M-%S").$SDG_OBJECT_STORE_DATA_KEY"
    export FINAL_DATA_TAR_PATH="{data_pvc_mount_path}/$FINAL_DATA_TAR_FILE"
    echo "Final data tarball path: $FINAL_DATA_TAR_PATH"
    echo "Final data tarball file: $FINAL_DATA_TAR_FILE"
    echo "Archiving data before pushing to the object store"
    # Use '--ignore-failed-read' to ignore missing files, needed when no MMLU tasks directories are found MMLU_branch is skipped
    # So '{mmlu_branch_scores_path}' will not exist
    tar --create \
      --gzip \
      --verbose \
      --ignore-failed-read \
      --file "$FINAL_DATA_TAR_PATH" {mt_bench_output_path} {mt_bench_scores_path} {mt_bench_branch_scores_path} {mmlu_branch_scores_path} {candidate_model_path}
fi

tmp=$(mktemp -d)
cat <<EOF > "$tmp"/s3.py
import os
import boto3
import sys
import threading

# Credit: https://gist.github.com/egeulgen/538aadc90275d79d514a5bacc4d5694e
class ProgressPercentage(object):
    ''' Progress Class
    Class for calculating and displaying download progress
    '''
    def __init__(self, client, bucket, filename):
        ''' Initialize
        initialize with: file name, file size and lock.
        Set seen_so_far to 0. Set progress bar length
        '''
        self._filename = filename
        self._size = float(os.path.getsize(filename)) if os.getenv('STRATEGY') == 'upload' else client.head_object(Bucket=bucket, Key=filename)['ContentLength']
        self._seen_so_far = 0
        self._lock = threading.Lock()
        self.prog_bar_len = 80

    def __call__(self, bytes_amount):
        ''' Call
        When called, increments seen_so_far by bytes_amount,
        calculates percentage of seen_so_far/total file size
        and prints progress bar.
        '''
        # To simplify we'll assume this is hooked up to a single filename.
        with self._lock:
            self._seen_so_far += bytes_amount
            ratio = round((float(self._seen_so_far) / float(self._size)) * (self.prog_bar_len - 6), 1)
            current_length = int(round(ratio))

            percentage = round(100 * ratio / (self.prog_bar_len - 6), 1)

            bars = '+' * current_length
            output = bars + ' ' * (self.prog_bar_len - current_length - len(str(percentage)) - 1) + str(percentage) + '%'

            if self._seen_so_far != self._size:
                sys.stdout.write(output + '\\r')
            else:
                sys.stdout.write(output + '\\n')
            sys.stdout.flush()

def str_to_bool(s):
    if s is None:
      return False
    return s.lower() in ['true', '1', 't', 'y', 'yes']

# TODO: support signature version?
def build_boto3_client():
  return boto3.client(
    's3',
    aws_access_key_id=os.getenv('SDG_OBJECT_STORE_ACCESS_KEY'),
    aws_secret_access_key=os.getenv('SDG_OBJECT_STORE_SECRET_KEY'),
    endpoint_url=os.getenv('SDG_OBJECT_STORE_ENDPOINT', None),
    region_name=os.getenv('SDG_OBJECT_STORE_REGION', None),
    verify=str_to_bool(os.getenv('SDG_OBJECT_STORE_VERIFY_TLS', 'true'))
)

def download_s3_file():
    s3 = build_boto3_client()

    bucket_name = os.getenv('SDG_OBJECT_STORE_BUCKET')
    s3_key = os.getenv('SDG_OBJECT_STORE_DATA_KEY')
    output_file = '{data_pvc_mount_path}/data.tar.gz'

    progress = ProgressPercentage(s3, bucket_name, s3_key)
    s3.download_file(bucket_name, s3_key, output_file, Callback=progress)

def upload_s3_file():
    s3 = build_boto3_client()

    bucket_name = os.getenv('SDG_OBJECT_STORE_BUCKET')
    s3_key = os.getenv('FINAL_DATA_TAR_FILE')
    input_file = os.getenv('FINAL_DATA_TAR_PATH')

    progress = ProgressPercentage(s3, bucket_name, input_file)
    s3.upload_file(input_file, bucket_name, s3_key, Callback=progress)

if __name__ == "__main__":
    if os.getenv('STRATEGY') == 'download':
      print('Downloading file from object store')
      download_s3_file()
    elif os.getenv('STRATEGY') == 'upload':
      print('Uploading file to object store')
      upload_s3_file()
    else:
      raise ValueError('Unknown STRATEGY')
EOF

python "$tmp"/s3.py

if [ "$STRATEGY" == "download" ]; then
    for dir in data model taxonomy; do
        dir_path="{data_pvc_mount_path}/$dir"
        if [ -d "$dir_path" ]; then
            echo "Directory $dir_path exists, it will be overwritten by the content of the archive"
        fi
    done

    echo "Extracting data from the archive"

    tar \
      --touch \
      --no-same-owner \
      --no-same-permissions \
      --directory {data_pvc_mount_path} \
      --extract \
      --verbose \
      --file {data_pvc_mount_path}/data.tar.gz

    # Enable globstar for recursive globbing
    shopt -s globstar

    # Patterns to match
    patterns=(
        "{data_pvc_mount_path}/model/config.json"
        "{data_pvc_mount_path}/model/tokenizer.json"
        "{data_pvc_mount_path}/model/tokenizer_config.json"
        "{data_pvc_mount_path}/model/*.safetensors"
        "{data_pvc_mount_path}/data/skills_recipe_*.yaml"
        "{data_pvc_mount_path}/data/knowledge_recipe_*.yaml"
        "{data_pvc_mount_path}/data/skills_train_*.jsonl"
        "{data_pvc_mount_path}/data/knowledge_train_*.jsonl"
        "{data_pvc_mount_path}/taxonomy/knowledge"
        "{data_pvc_mount_path}/taxonomy/foundational_skills"
    )

    match_count=0

    for pattern in "${{patterns[@]}}"; do
        matching_files=($pattern)
        if [ ! -s "${{matching_files[0]}}" ]; then
            echo "No files found matching pattern: $pattern: ${{matching_files[0]}}"
        else
            echo "Files found matching pattern: $pattern: ${{matching_files[0]}}"
            match_count=$((match_count+1))
        fi
    done

    if [ $match_count -ne ${{#patterns[@]}} ]; then
        echo "Error: Not all files were found, only $match_count files were found"
        ls -laR {data_pvc_mount_path}
        exit 1
    fi


    ls -laR {data_pvc_mount_path}
fi
"""

JOB_SCRIPT_EXAMPLE = """
kind: Job
apiVersion: batch/v1
metadata:
  name: {name}
  namespace: {namespace}
spec:
  template:
    spec:
      containers:
      - name: {name}
        image: {image}
        command:
          - "python3"
          - "/config/{script_name}"
          - "run"
        args: {args}
        volumeMounts:
        - name: script-config
          mountPath: /config
      restartPolicy: Never
      volumes:
      - name: script-config
        configMap:
          name: {script_configmap}
"""

PYTHON_EXECUTOR = """
set -e
export XDG_CACHE_HOME=/tmp
export OUTLINES_CACHE_DIR=/tmp
export NUMBA_CACHE_DIR=/tmp
export TRANSFORMERS_CACHE=/tmp
export HF_HOME=/tmp
export HOME=/tmp
export TRITON_CACHE_DIR=/tmp

tmp=$(mktemp -d)
cat <<EOF > "$tmp"/exec.py

{python_code}

if __name__ == "__main__":
    {python_main}

EOF

python3 "$tmp"/exec.py
"""


@click.group()
def cli():
    """
    Command Line Interface (CLI) entry point.

    This function serves as the main entry point for the command line interface.
    It currently does not perform any operations.
    """


@cli.group(invoke_without_command=True)
@click.option(
    "--namespace",
    type=str,
    default="default",
    help="Kubernetes namespace to run the job",
)
@click.option(
    "--name",
    type=str,
    default="distributed-ilab",
    help="Name of the Job to that can run the script",
)
@click.option(
    "--image",
    type=str,
    help="Image to use to run the script in a Job",
    required=True,
)
@click.option(
    "--service-account",
    type=str,
    help="Service account to use for the Job",
)
@click.option(
    "--script-configmap",
    type=str,
    help="Name of the ConfigMap containing the standalone.py script",
    required=True,
)
@click.option(
    "--script-name",
    type=str,
    help="Name of the standalone script in the ConfigMap (key)",
    default="standalone",
)
@click.option(
    "--args",
    type=str,
    help="Extra arguments to pass to the script",
    multiple=True,
    required=True,
)
def show(
    namespace: str,
    name: str,
    image: str,
    script_configmap: str,
    script_name: str,
    service_account: str,
    args: typing.List[str],
):
    """
    Print an example Job YAML to stdout to run the script in a Kubernetes cluster.
    The job excepts the standalone.py script to be available in a ConfigMap.
    """
    script = yaml.safe_load(
        JOB_SCRIPT_EXAMPLE.format(
            name=name,
            namespace=namespace,
            image=image,
            script_configmap=script_configmap,
            script_name=script_name,
            args=list(args),
        )
    )

    if service_account:
        script["spec"]["template"]["spec"]["serviceAccountName"] = service_account

    print(yaml.dump(script))


@cli.group(invoke_without_command=True)
@click.option("--namespace", type=str, help="Kubernetes namespace to use")
@click.option(
    "--taxonomy-repo-url",
    type=str,
    default=DEFAULT_REPO_URL,
    help="URL of the taxonomy repository - for SDG only",
    hidden=True,
)
@click.option(
    "--taxonomy-repo-branch",
    type=str,
    help="Branch of the taxonomy repository - for SDG only",
    hidden=True,
)
@click.option(
    "--taxonomy-repo-pr",
    type=str,
    help="Pull request number of the taxonomy repository - for SDG only",
    hidden=True,
)
@click.option(
    "--storage-class",
    type=str,
    help="Storage class to use for the PersistentVolumeClaim - for SDG only",
)
@click.option(
    "--serving-endpoint",
    type=str,
    help="Serving endpoint for SDG - for SDG only",
    hidden=True,
)
@click.option(
    "--serving-model",
    type=str,
    help="Serving model for SDG - for SDG only",
    hidden=True,
)
@click.option(
    "--judge-serving-model-endpoint",
    type=str,
    help=(
        "Judge model serving endpoint for evaluation."
        "e.g. http://serving.kubeflow.svc.cluster.local:8080/v1"
    ),
)
@click.option(
    "--judge-serving-model-name",
    type=str,
    help="The name of the model to use for evaluation.",
)
@click.option(
    "--judge-serving-model-api-key",
    type=str,
    help=(
        "Serving model API key for evaluation. " "(JUDGE_SERVING_MODEL_API_KEY env var)"
    ),
    envvar="JUDGE_SERVING_MODEL_API_KEY",
)
@click.option(
    "--judge-serving-model-ca-cert",
    type=str,
    help=(
        "Name of the Kubernetes ConfigMap containing the serving model CA cert."
        "The expected key name is 'ca-bundle.crt'."
    ),
)
@click.option(
    "--judge-serving-model-ca-cert-cm-key",
    type=str,
    help="Name of the Key in the Kubernetes ConfigMap containing the serving model CA cert.",
    default=JUDGE_CA_CERT_CM_KEY,
)
@click.option(
    "--judge-serving-model-secret",
    type=str,
    envvar="JUDGE_SERVING_MODEL_SECRET",
    help=(
        "Name of the Kubernetes Secret containing the judge serving model endpoint. "
        "For evaluation only. "
        "The namespace is inferred from the namespace option. "
        "The following keys are expected: JUDGE_API_KEY, JUDGE_ENDPOINT, JUDGE_NAME"
        "Optional keys are: JUDGE_CA_CERT, JUDGE_CA_CERT_CM_KEY"
        " (JUDGE_SERVING_MODEL_SECRET env var)"
        "If used, --judge-serving-model-{api-key,endpoint,name,ca-cert} will be ignored."
    ),
)
@click.option(
    "--nproc-per-node",
    type=int,
    help="Number of GPU to use per node - for training only",
    default=1,
)
@click.option(
    "--eval-type",
    help="Type of evaluation to run",
    type=click.Choice([EVAL_TYPE_MT_BENCH, EVAL_TYPE_FINAL]),
    hidden=True,
)
@click.option(
    "--training-phase",
    help="Type of training phase to run",
    type=click.Choice(["1", "2"]),
)
@click.option(
    "--model-to-train",
    help=(
        "Path to model to train (PVC filesystem path). "
        "Useful when calling training phases independently "
        "and users wants to point to the epoch directory. "
        "Very advanced usage, not recommended for general use."
    ),
    type=str,
)
@click.option(
    "--sdg-object-store-endpoint",
    envvar="SDG_OBJECT_STORE_ENDPOINT",
    help=(
        "Object store endpoint if different than the official AWS S3 endpoint. "
        "Expects an URL. TLS with self-signed certificates is not supported. "
        "(SDG_OBJECT_STORE_ENDPOINT env var)"
        "e.g. https://s3.openshift-storage.svc:443"
        "Don't forget the URL scheme (http/https) and the port"
    ),
    type=str,
)
@click.option(
    "--sdg-object-store-bucket",
    envvar="SDG_OBJECT_STORE_BUCKET",
    help="Object store bucket containing SDG data. (SDG_OBJECT_STORE_BUCKET env var)",
    type=str,
)
@click.option(
    "--sdg-object-store-access-key",
    envvar="SDG_OBJECT_STORE_ACCESS_KEY",
    help="Object store access key for SDG. (SDG_OBJECT_STORE_ACCESS_KEY env var)",
    type=str,
)
@click.option(
    "--sdg-object-store-secret-key",
    envvar="SDG_OBJECT_STORE_SECRET_KEY",
    help="Object store secret key for SDG. (SDG_OBJECT_STORE_SECRET_KEY env var)",
    type=str,
)
@click.option(
    "--sdg-object-store-region",
    envvar="SDG_OBJECT_STORE_REGION",
    help="Region for the object store. (SDG_OBJECT_STORE_REGION env var)",
    type=str,
)
@click.option(
    "--sdg-object-store-data-key",
    envvar="SDG_OBJECT_STORE_DATA_KEY",
    help=(
        "Name of tarball that contains SDG data AND model files."
        "(SDG_OBJECT_STORE_DATA_KEY env var)."
        "The tarball MUST contain two directories: data and model."
        "The data directory contains the SDG data."
        "The model directory contains the model to train."
        "To archive use the following command: "
        "tar -czvf data.tar.gz /path/to/data /path/to/model /path/to/taxonomy."
    ),
    type=str,
)
@click.option(
    "--sdg-object-store-verify-tls",
    envvar="SDG_OBJECT_STORE_VERIFY_TLS",
    help="Verify TLS for the object store. (SDG_OBJECT_STORE_VERIFY_TLS env var).",
    default=True,
    type=bool,
)
@click.option(
    "--sdg-object-store-secret",
    envvar="SDG_OBJECT_STORE_SECRET",
    help=(
        "Name of the Kubernetes Secret containing the SDG object store credentials. "
        "The namespace is inferred from the namespace option. "
        "The following keys are expected: bucket, access_key, secret_key, data_key. "
        " (SDG_OBJECT_STORE_SECRET env var)"
        "If used "
        "endpoint, bucket, access_key, secret_key, region, data_key, verify_tls will be ignored."
        "All supported options are: "
        "endpoint, bucket, access_key, secret_key, region, data_key, verify_tls"
    ),
    type=str,
)
@click.option(
    "--force-pull",
    help=(
        "Force pull the data (sdg data and model) from the object store "
        "even if it already exists in the PVC."
    ),
    is_flag=True,
    default=False,
)
@click.option(
    "--training-1-epoch-num", help="Number of epochs to train the model for.", default=7
)
@click.option(
    "--training-2-epoch-num",
    help="Number of epochs to train the model for.",
    default=10,
)
@click.option(
    "--num-instructions-to-generate",
    help="Number of instructions to generate.",
    default=30,
    hidden=True,
)
@click.option(
    "--dry-run",
    help=(
        "Print the generated YAML to stdout instead of creating the resources."
        "**WARNING**: secrets will be printed too!"
    ),
    is_flag=True,
)
@click.pass_context
def run(
    ctx: click.Context,
    namespace: typing.Optional[str] = None,
    taxonomy_repo_url: str = "",
    taxonomy_repo_branch: typing.Optional[str] = "",
    taxonomy_repo_pr: typing.Optional[str] = "",
    storage_class: typing.Optional[str] = None,
    serving_endpoint: typing.Optional[str] = None,
    serving_model: typing.Optional[str] = None,
    judge_serving_model_endpoint: typing.Optional[str] = None,
    judge_serving_model_name: typing.Optional[str] = None,
    judge_serving_model_api_key: typing.Optional[str] = None,
    judge_serving_model_ca_cert: typing.Optional[str] = None,
    judge_serving_model_ca_cert_cm_key: typing.Optional[str] = None,
    judge_serving_model_secret: typing.Optional[str] = None,
    nproc_per_node: typing.Optional[int] = 1,
    eval_type: typing.Optional[str] = None,
    training_phase: typing.Optional[str] = None,
    model_to_train: typing.Optional[str] = None,
    sdg_object_store_endpoint: typing.Optional[str] = None,
    sdg_object_store_bucket: typing.Optional[str] = None,
    sdg_object_store_access_key: typing.Optional[str] = None,
    sdg_object_store_secret_key: typing.Optional[str] = None,
    sdg_object_store_region: typing.Optional[str] = None,
    sdg_object_store_data_key: typing.Optional[str] = None,
    sdg_object_store_verify_tls: typing.Optional[bool] = None,
    sdg_object_store_secret: typing.Optional[str] = None,
    force_pull: typing.Optional[bool] = False,
    training_1_epoch_num: int = 7,
    training_2_epoch_num: int = 10,
    num_instructions_to_generate: typing.Optional[int] = 30,
    dry_run: bool = False,
):
    """
    Execute the distributed training on Kubernetes.

    Args:
        namespace (str): The namespace to use for the setup process.
        taxonomy_repo_url (str): The URL of the taxonomy repository. For SDG only.
        taxonomy_repo_branch (str): The branch of the taxonomy repository. For SDG only.
        taxonomy_repo_pr (int): The pull request number of the taxonomy repository. For SDG only.
        storage_class (str): The storage class to use for the PersistentVolumeClaim. For SDG only.
        serving_endpoint (str): The serving endpoint for SDG. For SDG only.
        serving_model (str): The serving model for SDG. For SDG only.
        judge_serving_model_endpoint (str): The serving endpoint for evaluation. For Evaluation
        only.
        judge_serving_model_name (str): The serving model name for evaluation. For Evaluation only.
        judge_serving_model_api_key (str): The serving model API key for evaluation. For Evaluation
        only.
        judge_serving_model_ca_cert (str): The serving model CA cert for evaluation.
        judge_serving_model_ca_cert_cm_key (str): The name of the Key in the Kubernetes ConfigMap
        judge_serving_model_secret (str): The name of the Kubernetes Secret containing the serving
        model credentials. For Evaluation only.
        nproc_per_node (int): The number of processes per node. For training only.
        eval_type (str): The type of evaluation to run.
        training_phase (str): The type of training phase to run.
        model_to_train (str): The path to model to train (PVC filesystem path).
        sdg_object_store_endpoint (str): The object store endpoint for SDG.
        sdg_object_store_bucket (str): The object store bucket containing SDG data.
        sdg_object_store_access_key (str): The object store access key for SDG.
        sdg_object_store_secret_key (str): The object store secret key for SDG.
        sdg_object_store_region (str): The region for the object store.
        sdg_object_store_data_key (str): The name of the tarball that contains SDG data.
        sdg_object_store_verify_tls (bool): Verify TLS for the object store.
        sdg_object_store_secret (str): The name of the Kubernetes Secret containing the SDG object
        store credentials. The namespace is inferred from the namespace option.
        force_pull (bool): Force pull the data (sdg data and model) from the object store even if it
        already exists in the PVC.
        training_1_epoch_num (int): Number of epochs to train the model for during phase 1.
        training_2_epoch_num (int): Number of epochs to train the model for during phase 2.
        num_instructions_to_generate (int): Number of instructions to generate during SDG.
        dry_run (bool): Print the generated YAML to stdout instead of creating the resources.

    Returns:
        None
    """
    ctx.ensure_object(dict)
    ctx.obj["namespace"] = namespace
    ctx.obj["taxonomy_repo_url"] = taxonomy_repo_url
    ctx.obj["taxonomy_repo_branch"] = taxonomy_repo_branch
    ctx.obj["taxonomy_repo_pr"] = taxonomy_repo_pr
    ctx.obj["storage_class"] = storage_class
    ctx.obj["serving_endpoint"] = serving_endpoint
    ctx.obj["serving_model"] = serving_model
    ctx.obj["judge_serving_model_endpoint"] = judge_serving_model_endpoint
    ctx.obj["judge_serving_model_name"] = judge_serving_model_name
    ctx.obj["judge_serving_model_api_key"] = judge_serving_model_api_key
    ctx.obj["judge_serving_model_ca_cert"] = judge_serving_model_ca_cert
    ctx.obj["judge_serving_model_secret"] = judge_serving_model_secret
    ctx.obj["judge_serving_model_ca_cert_cm_key"] = judge_serving_model_ca_cert_cm_key
    ctx.obj["nproc_per_node"] = nproc_per_node
    ctx.obj["eval_type"] = eval_type
    ctx.obj["training_phase"] = training_phase
    ctx.obj["model_to_train"] = model_to_train
    ctx.obj["sdg_object_store_endpoint"] = sdg_object_store_endpoint
    ctx.obj["sdg_object_store_bucket"] = sdg_object_store_bucket
    ctx.obj["sdg_object_store_access_key"] = sdg_object_store_access_key
    ctx.obj["sdg_object_store_secret_key"] = sdg_object_store_secret_key
    ctx.obj["sdg_object_store_region"] = sdg_object_store_region
    ctx.obj["sdg_object_store_data_key"] = sdg_object_store_data_key
    ctx.obj["sdg_object_store_verify_tls"] = sdg_object_store_verify_tls
    ctx.obj["sdg_object_store_secret"] = sdg_object_store_secret
    ctx.obj["force_pull"] = force_pull
    ctx.obj["training_1_epoch_num"] = training_1_epoch_num
    ctx.obj["training_2_epoch_num"] = training_2_epoch_num
    ctx.obj["num_instructions_to_generate"] = num_instructions_to_generate
    ctx.obj["dry_run"] = dry_run

    ##########################
    # MAIN WORKFLOW SEQUENCE #
    ##########################
    # When the script is simply called like: 'python standalone.py run'
    # We will run the entire workflow
    if ctx.invoked_subcommand is None:
        # SDG Full
        # ctx.invoke(sdg)

        # SDG Data Fetch
        ctx.invoke(sdg_data_fetch)

        # Begin multi-phased distributed training
        logger.info("Running multi-phased distributed training.")

        # Training Phase 1
        ctx.obj["training_phase"] = "1"
        ctx.invoke(train)

        # Evaluation of phase 1 with MMLU
        # ctx.obj["eval_type"] = "mmlu"
        # scores = ctx.invoke(evaluation)
        # scores = json.loads(scores)
        # best_model = max(scores, key=lambda x: x["average_score"])
        # logger.info("Best model: %s", best_model.get("model"))
        # ctx.obj["model_to_train"] = best_model.get("model")

        # Training Phase 2
        ctx.obj["training_phase"] = "2"
        ctx.invoke(train)

        # Evaluation of phase 2 with MT-Bench
        ctx.obj["eval_type"] = EVAL_TYPE_MT_BENCH
        scores = ctx.invoke(evaluation)
        if not dry_run:
            scores = json.loads(scores)
            logger.info("Best model: %s", scores.get("best_model"))
            ctx.obj["candidate_model"] = scores.get("best_model")

        # Final evaluation
        ctx.obj["eval_type"] = EVAL_TYPE_FINAL
        ctx.invoke(evaluation)
        logger.info("InstructLab Training Finished!")

        # Push the best model to S3
        ctx.invoke(upload_trained_model)


def get_security_context() -> kubernetes.client.V1SecurityContext:
    """
    Get the security context.
    """
    return kubernetes.client.V1SecurityContext(
        capabilities=kubernetes.client.V1Capabilities(drop=["ALL"]),
        run_as_non_root=True,
    )


def get_vol_mount() -> list[kubernetes.client.V1VolumeMount]:
    """
    Get the volume mount for the SDG job.
    """
    return [
        kubernetes.client.V1VolumeMount(
            name=DATA_VOLUME_NAME, mount_path=DATA_PVC_MOUNT_PATH
        ),
    ]


def get_vol() -> list[kubernetes.client.V1Volume]:
    """
    Get the volume for the SDG job.
    """
    return [
        kubernetes.client.V1Volume(
            name=DATA_VOLUME_NAME,
            persistent_volume_claim=kubernetes.client.V1PersistentVolumeClaimVolumeSource(
                claim_name=DATA_PVC_NAME
            ),
        ),
    ]


def create_sdg_job(
    namespace: str,
    job_name: str,
    num_instructions_to_generate: int,
    exec_git_clone_op_repo_url: str = "",
    exec_git_clone_op_repo_branch: str = "",
    exec_git_clone_op_repo_pr: str = "",
) -> kubernetes.client.V1Job:
    """
    Create a Kubernetes Job object.

    This function generates a Kubernetes Job object configured to run SDG steps.

    Steps:
        1. InitContainer to fetch the taxonomy data. - EmptyDir volume to share data between
           containers.
        2. InitContainer to generate synthetic data. - Stored on EmptyDir volume. (Option to push to
           S3?)
        3. Main container to pre-process the data before training. From the EmptyDir volume and copy
           the result to the PVC.
    Args:
        namespace (str): The namespace in which the job will be created.
        job_name (str): The name of the job.
        num_instructions_to_generate (int): The number of instructions to generate.
        exec_git_clone_op_repo_url (str): The URL of the taxonomy repository.
        exec_git_clone_op_repo_branch (str, optional): The branch of the taxonomy repository.
        exec_git_clone_op_repo_pr (str, optional): The pull request number of the taxonomy
        repository.

    Returns:
        kubernetes.client.V1Job: A Kubernetes Job object configured with the specified parameters.
    """
    # Configureate Pod template container
    exec_sdg_op_command = """
from typing import *

def sdg_op(
    num_instructions_to_generate: int,
    taxonomy: str,
    sdg: str,
    repo_branch: Optional[str],
    repo_pr: Optional[int],
):
    from os import getenv

    import openai
    from instructlab.sdg import generate_data
    from instructlab.sdg.utils.taxonomy import read_taxonomy

    api_key = getenv("api_key")
    model = getenv("model")
    endpoint = getenv("endpoint")
    client = openai.OpenAI(base_url=endpoint, api_key=api_key)

    taxonomy_base = "main" if repo_branch or (repo_pr and int(repo_pr) > 0) else "empty"

    print("Generating syntetic dataset for:")
    print()
    print(read_taxonomy(taxonomy, taxonomy_base))

    # generate_data has a magic word for its taxonomy_base argument - 'empty'
    # it allows generating from the whole repo, see:
    # https://github.com/instructlab/sdg/blob/c6a9e74a1618b1077cd38e713b8aaed8b7c0c8ce/src/instructlab/sdg/utils/taxonomy.py#L230
    generate_data(
        client=client,
        num_instructions_to_generate=num_instructions_to_generate,
        output_dir=sdg,
        taxonomy=taxonomy,
        taxonomy_base=taxonomy_base,
        model_name=model,
        chunk_word_count=1000,
        server_ctx_size=4096,
    )
"""
    exec_sdg_op_args = f"""
sdg_op(num_instructions_to_generate={num_instructions_to_generate}, repo_branch="{exec_git_clone_op_repo_branch}", repo_pr={exec_git_clone_op_repo_pr}, taxonomy="{TAXONOMY_DATA_PATH}", sdg="{SDG_GENERATED_DATA_PATH}")
"""
    exec_huggingface_importer_op_command = """
from typing import *

def huggingface_importer_op(model: str, repo_name: str):
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=repo_name, cache_dir="/tmp", local_dir=model)
"""
    exec_huggingface_importer_op_args = f"""
huggingface_importer_op(repo_name="{REPO_GRANITE_7B_IMAGE}", model="{DATA_PVC_MODEL_PATH}")
"""
    exec_data_processing_op_command = """
from typing import *

def data_processing_op(
    sdg: str,
    skills_processed_data: str,
    knowledge_processed_data: str,
    model: str,
    max_seq_len: Optional[int] = 4096,
    max_batch_len: Optional[int] = 20000,
):
    import os

    import instructlab.training.data_process as dp
    from instructlab.training import (
        DataProcessArgs,
        TrainingArgs,
    )

    # define training-specific arguments
    skill_training_args = TrainingArgs(
        # define data-specific arguments
        model_path=model,
        data_path=f"{sdg}/skills_train_msgs*.jsonl",
        data_output_dir=skills_processed_data,
        # define model-trianing parameters
        max_seq_len=max_seq_len,
        max_batch_len=max_batch_len,
        # XXX(shanand): We don't need the following arguments
        # for data processing. Added them for now to avoid
        # Pydantic validation errors for TrainingArgs
        ckpt_output_dir="data/saved_checkpoints",
        num_epochs=2,
        effective_batch_size=3840,
        save_samples=0,
        learning_rate=2e-6,
        warmup_steps=800,
        is_padding_free=True,
    )

    knowledge_training_args = TrainingArgs(
        # define data-specific arguments
        model_path=model,
        data_path=f"{sdg}/knowledge_train_msgs*.jsonl",
        data_output_dir=knowledge_processed_data,
        # define model-trianing parameters
        max_seq_len=max_seq_len,
        max_batch_len=max_batch_len,
        # XXX(shanand): We don't need the following arguments
        # for data processing. Added them for now to avoid
        # Pydantic validation errors for TrainingArgs
        ckpt_output_dir="data/saved_checkpoints",
        num_epochs=2,
        effective_batch_size=3840,
        save_samples=0,
        learning_rate=2e-6,
        warmup_steps=800,
        is_padding_free=True,
    )

    def data_processing(train_args: TrainingArgs) -> None:
        # early validation logic here
        if train_args.max_batch_len < train_args.max_seq_len:
            raise ValueError(
                f"the 'max_batch_len' cannot be less than 'max_seq_len': {train_args.max_batch_len=} < {train_args.max_seq_len=}"
            )

            # process the training data
        if not os.path.exists(train_args.data_output_dir):
            os.makedirs(train_args.data_output_dir, exist_ok=True)
        dp.main(
            DataProcessArgs(
                # XXX(osilkin): make a decision here, either:
                #   1. the CLI is fully responsible for managing where the data is written
                #   2. we never cache it and simply write it to a tmp file every time.
                #
                # An important reason for why #1 would be preferable is in the case of OpenShift/SELinux
                # where the user has a defined place for new temporary data to be written.
                data_output_path=train_args.data_output_dir,
                model_path=train_args.model_path,
                data_path=train_args.data_path,
                max_seq_len=train_args.max_seq_len,
                chat_tmpl_path=train_args.chat_tmpl_path,
            )
        )

    data_processing(train_args=skill_training_args)
    data_processing(train_args=knowledge_training_args)
"""
    exec_data_processing_op_args = f"""
data_processing_op(max_seq_len={MAX_SEQ_LEN}, max_batch_len={MAX_BATCH_LEN}, sdg="{DATA_PVC_SDG_PATH}", model="{DATA_PVC_MODEL_PATH}", skills_processed_data="{PREPROCESSED_DATA_PATH_SKILLS}", knowledge_processed_data="{PREPROCESSED_DATA_PATH_KNOWLEDGE}")
"""
    exec_git_clone_op_args = literal_eval("""
['git clone {exec_git_clone_op_repo_url} {TAXONOMY_PATH} && cd {TAXONOMY_PATH} && if [ -n "{exec_git_clone_op_repo_branch}" ]; then git fetch origin {exec_git_clone_op_repo_branch} && git checkout {exec_git_clone_op_repo_branch}; elif [ -n "{exec_git_clone_op_repo_pr}" ] && [ {exec_git_clone_op_repo_pr} -gt 0 ]; then git fetch origin pull/{exec_git_clone_op_repo_pr}/head:{exec_git_clone_op_repo_pr} && git checkout {exec_git_clone_op_repo_pr}; fi ']
""")

    init_containers = [
        kubernetes.client.V1Container(
            name="sdg-op-fetch-taxonomy-data",
            image=DS_IMAGE,
            command=["/bin/sh", "-c"],
            args=exec_git_clone_op_args,
            volume_mounts=get_vol_mount(),
            security_context=get_security_context(),
        ),
        kubernetes.client.V1Container(
            name="sdg-op-generate-synthetic-data",
            image=RHELAI_IMAGE,
            command=["/bin/sh", "-ce"],
            args=[
                PYTHON_EXECUTOR.format(
                    python_code=exec_sdg_op_command,
                    python_main=exec_sdg_op_args.strip(),
                ),
            ],
            volume_mounts=get_vol_mount(),
            security_context=get_security_context(),
            # env_from=[
            #     kubernetes.client.V1EnvFromSource(
            #         config_map_ref=kubernetes.client.V1ConfigMapEnvSource(name=K8S_NAME)
            #     ),
            #     kubernetes.client.V1EnvFromSource(
            #         secret_ref=kubernetes.client.V1SecretEnvSource(name=K8S_NAME)
            #     ),
            # ],
        ),
        kubernetes.client.V1Container(
            name="huggingface-importer-op",
            image=RHELAI_IMAGE,
            command=["/bin/sh", "-ce"],
            args=[
                PYTHON_EXECUTOR.format(
                    python_code=exec_huggingface_importer_op_command,
                    python_main=exec_huggingface_importer_op_args.strip(),
                ),
            ],
            volume_mounts=get_vol_mount(),
            security_context=get_security_context(),
            # env_from=[
            #     kubernetes.client.V1EnvFromSource(
            #         config_map_ref=kubernetes.client.V1ConfigMapEnvSource(name=K8S_NAME)
            #     ),
            #     kubernetes.client.V1EnvFromSource(
            #         secret_ref=kubernetes.client.V1SecretEnvSource(name=K8S_NAME)
            #     ),
            # ],
        ),
        kubernetes.client.V1Container(
            name="sdg-preprocess",
            image=RHELAI_IMAGE,
            command=["/bin/sh", "-ce"],
            args=[
                PYTHON_EXECUTOR.format(
                    python_code=exec_data_processing_op_command,
                    python_main=exec_data_processing_op_args.strip(),
                ),
            ],
            volume_mounts=get_vol_mount(),
            security_context=get_security_context(),
        ),
    ]

    # Format each string in the args list of each init container
    for container in init_containers:
        if container.name == "sdg-op-fetch-taxonomy-data":
            container.args = [
                arg.format(
                    exec_git_clone_op_repo_url=exec_git_clone_op_repo_url or "",
                    exec_git_clone_op_repo_branch=exec_git_clone_op_repo_branch or "",
                    exec_git_clone_op_repo_pr=exec_git_clone_op_repo_pr or "",
                    TAXONOMY_PATH=TAXONOMY_PATH,
                )
                for arg in container.args
            ]

    container = kubernetes.client.V1Container(
        name="copy-model-to-pvc",
        image=DS_IMAGE,
        command=["/bin/sh", "-c"],
        args=[
            f"cp -r -v {DATA_PVC_MOUNT_PATH} {DATA_PVC_MOUNT_PATH}"
        ],  # TODO: fix me, dumb line to pass linter, this feat is unused anyway
        volume_mounts=get_vol_mount(),
    )

    volumes = get_vol()

    # Create and configure a spec section
    template = kubernetes.client.V1PodTemplateSpec(
        metadata=kubernetes.client.V1ObjectMeta(labels={"app": "sdg"}),
        spec=kubernetes.client.V1PodSpec(
            restart_policy="Never",
            init_containers=init_containers,
            containers=[container],
            volumes=volumes,
        ),
    )

    # Create the specification of deployment
    spec = kubernetes.client.V1JobSpec(
        template=template,
    )

    # Instantiate the job object
    job = kubernetes.client.V1Job(
        api_version="batch/v1",
        kind="Job",
        metadata=kubernetes.client.V1ObjectMeta(name=job_name, namespace=namespace),
        spec=spec,
    )

    return job


def create_data_job(
    namespace: str,
    job_name: str,
    sdg_object_store_secret: str,
    strategy: str,
    force_pull: bool = False,
) -> kubernetes.client.V1Job:
    """
    Create a Kubernetes Job object.

    This function generates a Kubernetes Job object configured to fetch SDG data from an object
    store.

    Args:
        namespace (str): The namespace in which the job will be created.
        job_name (str): The name of the job.
        sdg_object_store_secret (str): The name of the Kubernetes Secret containing the SDG object
        store credentials.
        strategy (str): The strategy to use to fetch the data. Either "download" or "upload".
        force_pull (bool): Force pull the data from the object store even if it already exists in
        the PVC.

    Returns:
        kubernetes.client.V1Job: A Kubernetes Job object configured with the specified parameters.
    """

    exec_data_processing_op_command = """
from typing import *

def data_processing_op(
    sdg: str,
    skills_processed_data: str,
    knowledge_processed_data: str,
    model: str,
    max_seq_len: Optional[int] = 4096,
    max_batch_len: Optional[int] = 20000,
):
    import os

    import instructlab.training.data_process as dp
    from instructlab.training import (
        DataProcessArgs,
        TrainingArgs,
    )

    # define training-specific arguments
    skill_training_args = TrainingArgs(
        # define data-specific arguments
        model_path=model,
        data_path=f"{sdg}/skills_train_msgs*.jsonl",
        data_output_dir=skills_processed_data,
        # define model-trianing parameters
        max_seq_len=max_seq_len,
        max_batch_len=max_batch_len,
        # XXX(shanand): We don't need the following arguments
        # for data processing. Added them for now to avoid
        # Pydantic validation errors for TrainingArgs
        ckpt_output_dir="data/saved_checkpoints",
        num_epochs=2,
        effective_batch_size=3840,
        save_samples=0,
        learning_rate=2e-6,
        warmup_steps=800,
        is_padding_free=True,
    )

    knowledge_training_args = TrainingArgs(
        # define data-specific arguments
        model_path=model,
        data_path=f"{sdg}/knowledge_train_msgs*.jsonl",
        data_output_dir=knowledge_processed_data,
        # define model-trianing parameters
        max_seq_len=max_seq_len,
        max_batch_len=max_batch_len,
        # XXX(shanand): We don't need the following arguments
        # for data processing. Added them for now to avoid
        # Pydantic validation errors for TrainingArgs
        ckpt_output_dir="data/saved_checkpoints",
        num_epochs=2,
        effective_batch_size=3840,
        save_samples=0,
        learning_rate=2e-6,
        warmup_steps=800,
        is_padding_free=True,
    )

    def data_processing(train_args: TrainingArgs) -> None:
        # early validation logic here
        if train_args.max_batch_len < train_args.max_seq_len:
            raise ValueError(
                f"the 'max_batch_len' cannot be less than 'max_seq_len': {train_args.max_batch_len=} < {train_args.max_seq_len=}"
            )

            # process the training data
        if not os.path.exists(train_args.data_output_dir):
            os.makedirs(train_args.data_output_dir, exist_ok=True)
        dp.main(
            DataProcessArgs(
                # XXX(osilkin): make a decision here, either:
                #   1. the CLI is fully responsible for managing where the data is written
                #   2. we never cache it and simply write it to a tmp file every time.
                #
                # An important reason for why #1 would be preferable is in the case of OpenShift/SELinux
                # where the user has a defined place for new temporary data to be written.
                data_output_path=train_args.data_output_dir,
                model_path=train_args.model_path,
                data_path=train_args.data_path,
                max_seq_len=train_args.max_seq_len,
                chat_tmpl_path=train_args.chat_tmpl_path,
            )
        )

    data_processing(train_args=skill_training_args)
    data_processing(train_args=knowledge_training_args)
"""
    exec_data_processing_op_args = f"""
data_processing_op(max_seq_len={MAX_SEQ_LEN}, max_batch_len={MAX_BATCH_LEN}, sdg="{DATA_PVC_SDG_PATH}", model="{DATA_PVC_MODEL_PATH}", skills_processed_data="{PREPROCESSED_DATA_SKILLS_PATH}", knowledge_processed_data="{PREPROCESSED_DATA_KNOWLEDGE_PATH}")
"""

    data_container = kubernetes.client.V1Container(
        name=f"{strategy}-data-object-store",
        image=DS_IMAGE,
        command=["/bin/sh", "-c"],
        args=[
            DATA_SCRIPT.format(
                strategy=strategy,
                force_pull=force_pull,
                data_pvc_mount_path=DATA_PVC_MOUNT_PATH,
                mt_bench_output_path=MT_BENCH_OUTPUT_PATH,
                mt_bench_scores_path=MT_BENCH_SCORES_PATH,
                mt_bench_branch_scores_path=MT_BENCH_BRANCH_SCORES_PATH,
                mmlu_branch_scores_path=MMLU_BRANCH_SCORES_PATH,
                candidate_model_path=CANDIDATE_MODEL_PATH,
            )
        ],
        volume_mounts=get_vol_mount(),
        env=[
            kubernetes.client.V1EnvVar(
                name="SDG_OBJECT_STORE_ENDPOINT",
                value_from=kubernetes.client.V1EnvVarSource(
                    secret_key_ref=kubernetes.client.V1SecretKeySelector(
                        name=sdg_object_store_secret,
                        key="endpoint",
                        optional=True,
                    )
                ),
            ),
            kubernetes.client.V1EnvVar(
                name="SDG_OBJECT_STORE_BUCKET",
                value_from=kubernetes.client.V1EnvVarSource(
                    secret_key_ref=kubernetes.client.V1SecretKeySelector(
                        name=sdg_object_store_secret,
                        key="bucket",
                        optional=False,
                    )
                ),
            ),
            kubernetes.client.V1EnvVar(
                name="SDG_OBJECT_STORE_ACCESS_KEY",
                value_from=kubernetes.client.V1EnvVarSource(
                    secret_key_ref=kubernetes.client.V1SecretKeySelector(
                        name=sdg_object_store_secret,
                        key="access_key",
                        optional=False,
                    )
                ),
            ),
            kubernetes.client.V1EnvVar(
                name="SDG_OBJECT_STORE_SECRET_KEY",
                value_from=kubernetes.client.V1EnvVarSource(
                    secret_key_ref=kubernetes.client.V1SecretKeySelector(
                        name=sdg_object_store_secret,
                        key="secret_key",
                        optional=False,
                    )
                ),
            ),
            kubernetes.client.V1EnvVar(
                name="SDG_OBJECT_STORE_REGION",
                value_from=kubernetes.client.V1EnvVarSource(
                    secret_key_ref=kubernetes.client.V1SecretKeySelector(
                        name=sdg_object_store_secret,
                        key="region",
                        optional=True,
                    )
                ),
            ),
            kubernetes.client.V1EnvVar(
                name="SDG_OBJECT_STORE_DATA_KEY",
                value_from=kubernetes.client.V1EnvVarSource(
                    secret_key_ref=kubernetes.client.V1SecretKeySelector(
                        name=sdg_object_store_secret,
                        key="data_key",
                        optional=False,
                    )
                ),
            ),
            kubernetes.client.V1EnvVar(
                name="SDG_OBJECT_STORE_VERIFY_TLS",
                value_from=kubernetes.client.V1EnvVarSource(
                    secret_key_ref=kubernetes.client.V1SecretKeySelector(
                        name=sdg_object_store_secret,
                        key="verify_tls",
                        optional=True,
                    )
                ),
            ),
        ],
    )

    sdg_data_preprocess_container = kubernetes.client.V1Container(
        name="sdg-data-preprocess",
        image=RHELAI_IMAGE,
        command=["/bin/sh", "-ce"],
        args=[
            PYTHON_EXECUTOR.format(
                python_code=exec_data_processing_op_command,
                python_main=exec_data_processing_op_args.strip(),
            ),
        ],
        volume_mounts=get_vol_mount(),
        security_context=get_security_context(),
    )

    main_container = None
    if strategy == "download":
        main_container = sdg_data_preprocess_container
    # For the upload strategy, the main container is the data container since we only upload the
    # trained model back to the object store
    elif strategy == "upload":
        main_container = data_container

    # Create and configure a spec section
    template = kubernetes.client.V1PodTemplateSpec(
        metadata=kubernetes.client.V1ObjectMeta(labels={"app": "data-" + strategy}),
        spec=kubernetes.client.V1PodSpec(
            restart_policy="Never",
            containers=[main_container],
            volumes=get_vol(),
        ),
    )

    if strategy == "download":
        template.spec.init_containers = [data_container]

    # Create the specification of deployment
    spec = kubernetes.client.V1JobSpec(
        template=template,
    )

    # Instantiate the job object
    job = kubernetes.client.V1Job(
        api_version="batch/v1",
        kind="Job",
        metadata=kubernetes.client.V1ObjectMeta(name=job_name, namespace=namespace),
        spec=spec,
    )

    return job


def create_eval_job(
    namespace: str,
    eval_type: str,
    judge_serving_model_secret: str,
    nproc_per_node: int = 1,
    judge_serving_model_ca_cert: str = None,
    judge_serving_model_ca_cert_cm_key: str = None,
) -> kubernetes.client.V1Job:
    """
    Create a Kubernetes Job object.

    This function generates a Kubernetes Job object configured to run Evaluation steps.

    Args:
        namespace (str): The namespace in which the job will be created.
        eval_type (str): The type of evaluation to run.
        judge_serving_model_secret (str): The name of the Kubernetes Secret containing the judge
        nproc_per_node (int): The number of processes per node.

    Returns:
        kubernetes.client.V1Job: A Kubernetes Job object configured with the specified parameters.
    """

    job_name = f"eval-{eval_type}"

    # if eval_type == "mmlu":
    #     init_containers = [
    #         kubernetes.client.V1Container(
    #             name=f"run-eval-{eval_type}",
    #             image="",
    #             command=,
    #             args=,
    #             volume_mounts=[
    #                 kubernetes.client.V1VolumeMount(
    #                     name=TRAINING_VOLUME_NAME, mount_path=TRAINING_PVC_MOUNT_PATH
    #                 ),
    #             ],
    #         )
    #     ]
    #     container = kubernetes.client.V1Container(
    #         name=f"output-eval-{eval_type}-scores",
    #         image="",
    #         command=["/bin/sh", "-c"],
    #         args=[f"cat {MMLU_SCORES_PATH}"],
    #         volume_mounts=[
    #             kubernetes.client.V1VolumeMount(
    #                 name=TRAINING_VOLUME_NAME, mount_path=TRAINING_PVC_MOUNT_PATH
    #             ),
    #         ],
    #     )

    exec_run_mt_bench_op_command = """
from typing import *

def run_mt_bench_op(
    models_path_prefix: str,
    mt_bench_output: str,
    merge_system_user_message: bool,
    # generate_answers,judgment uses a magic word for its mt_bench evaluator  - 'auto'
    # with 'auto', number of gpus allocated for serving is calculated based on environment
    # https://github.com/instructlab/eval/blob/main/src/instructlab/eval/mt_bench.py#L36
    max_workers: str,
    models_list: List[str] = None,
    models_folder: Optional[str] = None,
    device: str = None,
    best_score_file: Optional[str] = None,
) -> NamedTuple("outputs", best_model=str, best_score=float):
    import json
    import os
    import subprocess

    import torch
    from instructlab.eval.mt_bench import MTBenchEvaluator

    if judge_ca_cert := os.getenv("JUDGE_CA_CERT_PATH"):
        import httpx
        import openai

        # Create a custom HTTP client
        class CustomHttpClient(httpx.Client):
            def __init__(self, *args, **kwargs):
                # Use the custom CA certificate
                kwargs.setdefault("verify", judge_ca_cert)
                super().__init__(*args, **kwargs)

        # Create a new OpenAI class that uses the custom HTTP client
        class CustomOpenAI(openai.OpenAI):
            def __init__(self, *args, **kwargs):
                custom_client = CustomHttpClient()
                super().__init__(http_client=custom_client, *args, **kwargs)

        # Monkey patch the OpenAI class in the openai module, so that the eval lib can use it
        openai.OpenAI = CustomOpenAI

    def launch_vllm(
        model_path: str, gpu_count: int, retries: int = 120, delay: int = 10
    ) -> tuple:
        import subprocess
        import sys
        import time

        import requests
        from instructlab.model.backends.common import free_tcp_ipv4_port

        free_port = free_tcp_ipv4_port("127.0.0.1")
        port = str(free_port)
        vllm_server = f"http://127.0.0.1:{port}/v1"

        command = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--port",
            port,
            "--model",
            model_path,
        ]
        if gpu_count > 0:
            command += [
                "--tensor-parallel-size",
                str(gpu_count),
            ]

        process = subprocess.Popen(args=command)

        print(f"Waiting for vLLM server to start at {vllm_server}...")

        for attempt in range(retries):
            try:
                response = requests.get(f"{vllm_server}/models")
                if response.status_code == 200:
                    print(f"vLLM server is up and running at {vllm_server}.")
                    return process, vllm_server
            except requests.ConnectionError:
                pass

            print(
                f"Server not available yet, retrying in {delay} seconds (Attempt {attempt + 1}/{retries})..."
            )
            time.sleep(delay)

        raise RuntimeError(
            f"Failed to start vLLM server at {vllm_server} after {retries} retries."
        )

    def shutdown_vllm(process: subprocess.Popen, timeout: int = 20):
        import subprocess

        from instructlab.model.backends.vllm import wait_for_stable_vram

        try:
            process.terminate()
            process.wait(timeout=timeout)

            if process.poll() is None:
                print(f"Forcefully killing vLLM server process with PID: {process.pid}")
                process.kill()

            print(f"Successfully stopped vLLM server with PID: {process.pid}")

        except subprocess.TimeoutExpired:
            print(
                f"Timeout expired. Forcefully killing vLLM server with PID: {process.pid}"
            )
            process.kill()  # Force kill the process if over timeout
        except subprocess.NoSuchProcess:
            print(f"Process with PID {process.pid} no longer exists.")
        except Exception as e:
            print(f"Failed to stop process with PID {process.pid}. Error: {e}")
        # Note from instructlab/model/backends/vllm.py
        # vLLM relies on stable VRAM,  residual reclamation activity
        # can lead to crashes on restart. To prevent this add a
        # short delay (typically ~ 10 seconds, max 30) to verify stability.
        wait_for_stable_vram(30)

    gpu_available = torch.cuda.is_available()
    gpu_name = (
        torch.cuda.get_device_name(torch.cuda.current_device())
        if gpu_available
        else "No GPU available"
    )
    gpu_count = torch.cuda.device_count() if gpu_available else 0

    print(f"GPU Available: {gpu_available}, {gpu_name}")

    if models_list is None and models_folder:
        models_list = os.listdir(models_folder)

    judge_api_key = os.getenv("JUDGE_API_KEY", "")
    judge_model_name = os.getenv("JUDGE_NAME")
    judge_endpoint = os.getenv("JUDGE_ENDPOINT")

    scores = {}
    all_mt_bench_data = []

    # generate_answers,judgment uses a magic word for its mt_bench evaluator  - 'auto'
    # with 'auto', number of gpus allocated for serving is calculated based on environment
    # https://github.com/instructlab/eval/blob/main/src/instructlab/eval/mt_bench.py#L36
    if max_workers == "auto":
        try:
            usable_cpu_count = len(os.sched_getaffinity(0)) // 2
        except AttributeError:
            usable_cpu_count = multiprocessing.cpu_count() // 2
        max_workers = usable_cpu_count

    # modify model_list to ignore any jsonl files present in the directory
    models_list = [model for model in models_list if not model.endswith(".jsonl")]
    for model_name in models_list:
        print(f"Serving candidate model: {model_name}")
        model_path = f"{models_path_prefix}/{model_name}"

        vllm_process, vllm_server = launch_vllm(model_path, gpu_count)

        # model ID is the model_path value in vLLM
        evaluator = MTBenchEvaluator(
            model_name=model_path,
            judge_model_name=judge_model_name,
            output_dir="/tmp/eval_output",
            merge_system_user_message=merge_system_user_message,
        )

        evaluator.gen_answers(
            server_url=vllm_server,
            serving_gpus=gpu_count,
            max_workers=max_workers,
        )

        shutdown_vllm(vllm_process)

        overall_score, qa_pairs, turn_scores, error_rate = evaluator.judge_answers(
            server_url=judge_endpoint,
            api_key=judge_api_key,
            serving_gpus=gpu_count,
            max_workers=max_workers,
        )

        mt_bench_data = {
            "report_title": "SKILLS EVALUATION REPORT",
            "model": model_path,
            "judge_model": judge_model_name,
            "overall_score": overall_score,
            "turn_scores": turn_scores,
            "qa_scores": qa_pairs,
            "error_rate": error_rate,
        }

        all_mt_bench_data.append(mt_bench_data)
        scores[model_path] = overall_score

    with open(mt_bench_output, "w", encoding="utf-8") as f:
        json.dump(all_mt_bench_data, f, indent=4)

    outputs = NamedTuple("outputs", best_model=str, best_score=float)
    best_model = max(scores, key=scores.get)
    best_score = scores[best_model]
    if best_score_file:
        with open(best_score_file, "w", encoding="utf-8") as f:
            json.dump({"best_model": best_model, "best_score": best_score}, f, indent=4)

    # Rename the best model directory to "candidate_model" for the next step
    # So we know which model to use for the final evaluation
    if os.path.exists(os.path.join(models_path_prefix, "candidate_model")):
        print("candidate_model already exists. Skipping renaming")
    else:
        os.rename(
            os.path.join(models_path_prefix, best_model),
            os.path.join(models_path_prefix, "candidate_model"),
        )

    return outputs(best_model=best_model, best_score=best_score)
"""
    exec_run_mt_bench_op_args = f"""
run_mt_bench_op(best_score_file="{MT_BENCH_SCORES_PATH}",mt_bench_output="{MT_BENCH_OUTPUT_PATH}",models_folder="{CANDIDATE_MODEL_PATH_PREFIX}",models_path_prefix="{CANDIDATE_MODEL_PATH_PREFIX}", max_workers="{MAX_WORKERS}", merge_system_user_message={MERGE_SYSTEM_USER_MESSAGE})
"""
    exec_run_final_eval_op_command = """
from typing import *

def run_final_eval_op(
    mmlu_branch_output: str,
    mt_bench_branch_output: str,
    base_model_dir: str,
    tasks: str,
    taxonomy: str,
    base_branch: str,
    candidate_branch: str,
    max_workers: str,
    device: str,
    model_dtype: str,
    few_shots: int,
    batch_size: int,
    merge_system_user_message: bool,
    candidate_model: str = None,
):
    import json
    import os
    import subprocess

    import torch
    from instructlab.eval.mmlu import MMLU_TASKS, MMLUBranchEvaluator
    from instructlab.eval.mt_bench import MTBenchBranchEvaluator
    from instructlab.model.evaluate import qa_pairs_to_qna_to_avg_scores, sort_score

    if judge_ca_cert := os.getenv("JUDGE_CA_CERT_PATH"):
        import httpx
        import openai

        # Create a custom HTTP client
        class CustomHttpClient(httpx.Client):
            def __init__(self, *args, **kwargs):
                # Use the custom CA certificate
                kwargs.setdefault("verify", judge_ca_cert)
                super().__init__(*args, **kwargs)

        # Create a new OpenAI class that uses the custom HTTP client
        class CustomOpenAI(openai.OpenAI):
            def __init__(self, *args, **kwargs):
                custom_client = CustomHttpClient()
                super().__init__(http_client=custom_client, *args, **kwargs)

        # Monkey patch the OpenAI class in the openai module, so that the eval lib can use it
        openai.OpenAI = CustomOpenAI

    print("Starting Final Eval...")

    def launch_vllm(
        model_path: str, gpu_count: int, retries: int = 120, delay: int = 10
    ) -> tuple:
        import subprocess
        import sys
        import time

        import requests
        from instructlab.model.backends.common import free_tcp_ipv4_port

        free_port = free_tcp_ipv4_port("127.0.0.1")
        port = str(free_port)
        vllm_server = f"http://127.0.0.1:{port}/v1"

        command = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--port",
            port,
            "--model",
            model_path,
        ]
        if gpu_count > 0:
            command += [
                "--tensor-parallel-size",
                str(gpu_count),
            ]

        process = subprocess.Popen(args=command)

        print(f"Waiting for vLLM server to start at {vllm_server}...")

        for attempt in range(retries):
            try:
                response = requests.get(f"{vllm_server}/models")
                if response.status_code == 200:
                    print(f"vLLM server is up and running at {vllm_server}.")
                    return process, vllm_server
            except requests.ConnectionError:
                pass

            print(
                f"Server not available yet, retrying in {delay} seconds (Attempt {attempt + 1}/{retries})..."
            )
            time.sleep(delay)

        raise RuntimeError(
            f"Failed to start vLLM server at {vllm_server} after {retries} retries."
        )

    def shutdown_vllm(process: subprocess.Popen, timeout: int = 20):
        import subprocess

        from instructlab.model.backends.vllm import wait_for_stable_vram

        try:
            process.terminate()
            process.wait(timeout=timeout)

            if process.poll() is None:
                print(f"Forcefully killing vLLM server process with PID: {process.pid}")
                process.kill()

            print(f"Successfully stopped vLLM server with PID: {process.pid}")

        except subprocess.TimeoutExpired:
            print(
                f"Timeout expired. Forcefully killing vLLM server with PID: {process.pid}"
            )
            process.kill()  # Force kill the process if over timeout
        except subprocess.NoSuchProcess:
            print(f"Process with PID {process.pid} no longer exists.")
        except Exception as e:
            print(f"Failed to stop process with PID {process.pid}. Error: {e}")
        # Note from instructlab/model/backends/vllm.py
        # vLLM relies on stable VRAM,  residual reclamation activity
        # can lead to crashes on restart. To prevent this add a
        # short delay (typically ~ 10 seconds, max 30) to verify stability.
        wait_for_stable_vram(30)

    # For standalone mode
    if candidate_model is None:
        # logic to get the best model from the models folder and results
        pass

    ######################################################################
    # branch_eval_summary_to_json creates a json object from output of instructlab/eval
    # TODO: Add this to the instructlab/eval or instructlab/instructlab repository
    def branch_eval_summary_to_json(
        improvements: list[tuple[str, float, float, float]],
        regressions: list[tuple[str, float, float, float]],
        no_changes: list[tuple[str, float]],
        new=None,
    ) -> str:
        # Generates a JSON object from the _branch benchmark evaluations

        import json

        summary = {"improvements": [], "regressions": [], "no_changes": [], "new": []}

        if len(improvements) > 0:
            improvements.sort(key=sort_score, reverse=True)
            for improvement in improvements:
                task, delta, base_score, new_score = improvement
                summary["improvements"].append(
                    {
                        "task": task,
                        "base_score": round(base_score, 2),
                        "new_score": round(new_score, 2),
                        "delta": delta,
                    }
                )

        if len(regressions) > 0:
            regressions.sort(key=sort_score)
            for regression in regressions:
                task, delta, base_score, new_score = regression
                summary["regressions"].append(
                    {
                        "task": task,
                        "base_score": round(base_score, 2),
                        "new_score": round(new_score, 2),
                        "delta": delta,
                    }
                )

        if len(no_changes) > 0:
            for entry in no_changes:
                task, avg_score = entry
                summary["no_changes"].append(
                    {"task": task, "average_score": round(avg_score, 2)}
                )

        if new is not None and len(new) > 0:
            for entry in new:
                na, avg_score = entry
                summary["new"].append(
                    {"qna": qna, "average_score": round(avg_score, 2)}
                )

        return json.dumps(summary, indent=4)

    ######################################################################
    print("Checking GPUs...")
    gpu_available = torch.cuda.is_available()
    gpu_name = (
        torch.cuda.get_device_name(torch.cuda.current_device())
        if gpu_available
        else "No GPU available"
    )
    gpu_count = torch.cuda.device_count() if gpu_available else 0

    print(f"GPU Available: {gpu_available}, Using: {gpu_name}")

    # MMLU_BRANCH

    # This is very specific to 'ilab generate', necessary because the data generation and
    # model evaluation are taking place in separate environments.
    def update_test_lines_in_files(base_dir):
        import os

        import yaml

        for root, dirs, files in os.walk(base_dir):
            for file_name in files:
                if file_name.startswith("knowledge_") and file_name.endswith(
                    "_task.yaml"
                ):
                    file_path = os.path.join(root, file_name)

                    with open(file_path, "r") as file:
                        task_yaml = yaml.load(file, Loader=yaml.Loader)

                    current_test_file_path = task_yaml["dataset_kwargs"]["data_files"][
                        "test"
                    ]
                    current_test_file_path_parts = current_test_file_path.split("/")
                    new_test_file_path = f"{root}/{current_test_file_path_parts[-1]}"
                    task_yaml["dataset_kwargs"]["data_files"]["test"] = (
                        new_test_file_path
                    )
                    with open(file_path, "w", encoding="utf-8") as file:
                        yaml.dump(task_yaml, file)

    # find_node_dataset_directories to find sdg output node_datasets_*
    def find_node_dataset_directories(base_dir: str):
        import os
        import re

        # This is specific to ilab/eval output
        pattern = r"node_datasets_"
        matching_dirs = []
        regex = re.compile(pattern)

        for root, dirs, files in os.walk(base_dir):
            for directory in dirs:
                if regex.search(directory):
                    matching_dirs.append(os.path.join(root, directory))

        # From 'ilab sdg' the knowledge_*_task.yaml files have a line that references where the SDG took place.
        # This needs to be updated to run elsewhere.
        # The line is:
        #    test: /path/to/where/sdg/occured/node_datasets_*
        # TODO: update sdg repo: https://github.com/instructlab/sdg/blob/366814b3e89e28c98c0d2a276ad0759c567d2798/src/instructlab/sdg/eval_data.py#L84-%23L114
        update_test_lines_in_files(base_dir)
        return matching_dirs

    print("Starting MMLU_Branch...")

    mmlu_tasks = ["mmlu_pr"]

    node_dataset_dirs = find_node_dataset_directories(tasks)

    # This assumes generated filesystem from ilab sdg, which
    # generates a node_datasets_ directory for MMLU custom tasks data
    if node_dataset_dirs:
        tasks_dir = node_dataset_dirs[0]

        mmlu_branch_evaluators = [
            MMLUBranchEvaluator(
                model_path=candidate_model,
                tasks_dir=tasks_dir,
                tasks=mmlu_tasks,
                few_shots=few_shots,
                batch_size=batch_size,
            ),
            MMLUBranchEvaluator(
                model_path=base_model_dir,
                tasks_dir=tasks_dir,
                tasks=mmlu_tasks,
                few_shots=few_shots,
                batch_size=batch_size,
            ),
        ]
        m_paths = [candidate_model, base_model_dir]
        overall_scores = []
        individual_scores_list = []
        for i, evaluator in enumerate(mmlu_branch_evaluators):
            m_path = m_paths[i]
            print("Launching Vllm...")
            vllm_process, vllm_server = launch_vllm(m_path, gpu_count)
            overall_score, individual_scores = evaluator.run(vllm_server)
            overall_scores.append(overall_score)
            individual_scores_list.append(individual_scores)
            print("Stopping Vllm")
            shutdown_vllm(vllm_process)

        # TODO: update instructlab/instructlab model/evaluate.py
        # so this logic can be imported outside of the CLI
        overall_score = overall_scores[0]
        base_overall_score = overall_scores[1]
        individual_scores = individual_scores_list[0]
        base_individual_scores = individual_scores_list[1]

        improvements, regressions, no_changes = [], [], []
        for task, score in individual_scores.items():
            base_score = base_individual_scores[task]
            s = score["score"]
            b_s = base_score["score"]
            d = round(s - b_s, 2)
            if s > b_s:
                improvements.append((task, d, b_s, s))
            elif b_s > s:
                regressions.append((task, d, b_s, s))
            else:
                no_changes.append((task, s))

        summary = branch_eval_summary_to_json(
            improvements,
            regressions,
            no_changes,
        )

        mmlu_branch_data = {
            "report_title": "KNOWLEDGE EVALUATION REPORT",
            "max_score": "1.0",
            "model": candidate_model,
            "model_score": round(overall_score, 2),
            "base_model": base_model_dir,
            "base_model_score": round(base_overall_score, 2),
            "summary": summary,
        }

        with open(mmlu_branch_output, "w") as f:
            json.dump(mmlu_branch_data, f, indent=4)
    else:
        print("No MMLU tasks directories found, skipping MMLU_branch evaluation.")

    # MT_BENCH_BRANCH

    print("Strating MT_BENCH_BRANCH ...")

    judge_api_key = os.getenv("JUDGE_API_KEY", "")
    judge_model_name = os.getenv("JUDGE_NAME")
    judge_endpoint = os.getenv("JUDGE_ENDPOINT")

    output_dir = "/tmp/eval_output"

    # TODO: candidate_branch must be in same repo, not a fork, or, can compare main branch against candidate, base models
    base_branch = base_branch or "main"
    candidate_branch = candidate_branch or "main"

    ######################################################################
    # TODO: Update ilab/model/evaluate evaluate def logic to allow for external judge model
    # and when that happens, much of this logic can be imported from the 'evaluate' definition:
    # https://github.com/instructlab/instructlab/blob/83ca501ecdd858677380046e2a56da5b2f3f14e7/src/instructlab/model/evaluate.py#L504
    #
    # With instructlab, model_name is synonomous with model_path
    mt_bench_evaluators = [
        MTBenchBranchEvaluator(
            model_name=candidate_model,
            judge_model_name=judge_model_name,
            taxonomy_git_repo_path=taxonomy,
            branch=candidate_branch,
            output_dir=output_dir,
            merge_system_user_message=merge_system_user_message,
        ),
        MTBenchBranchEvaluator(
            model_name=base_model_dir,
            judge_model_name=judge_model_name,
            taxonomy_git_repo_path=taxonomy,
            branch=base_branch,
            output_dir=output_dir,
            merge_system_user_message=merge_system_user_message,
        ),
    ]

    # ilab/evaluate uses a magic word for its mt_bench evaluator  - 'auto'
    # with 'auto', number of gpus allocated for serving is calculated based on environment
    # https://github.com/instructlab/eval/blob/main/src/instructlab/eval/mt_bench.py#L36
    if max_workers == "auto":
        try:
            usable_cpu_count = len(os.sched_getaffinity(0)) // 2
        except AttributeError:
            usable_cpu_count = multiprocessing.cpu_count() // 2
        max_workers = usable_cpu_count

    branches = [candidate_branch, base_branch]
    m_paths = [candidate_model, base_model_dir]
    qa_pairs_and_errors = []
    for i, evaluator in enumerate(mt_bench_evaluators):
        branch = branches[i]
        m_path = m_paths[i]

        print(
            f"Generating questions and reference answers from qna files for branch {branch}..."
        )
        vllm_process, vllm_server = launch_vllm(m_path, gpu_count)

        evaluator.gen_answers(
            server_url=vllm_server,
            serving_gpus=gpu_count,
            max_workers=max_workers,
        )

        shutdown_vllm(vllm_process)

        print(f"Evaluating answers for branch {branch}...")
        overall_score, qa_pairs, error_rate = evaluator.judge_answers(
            server_url=judge_endpoint,
            api_key=judge_api_key,
            serving_gpus=gpu_count,
            max_workers=max_workers,
        )

        qa_pairs_and_errors.append((overall_score, qa_pairs, error_rate))

    overall_score, qa_pairs, error_rate = qa_pairs_and_errors[0]
    base_overall_score, base_qa_pairs, base_error_rate = qa_pairs_and_errors[1]

    qna_to_avg_scores = qa_pairs_to_qna_to_avg_scores(qa_pairs)
    base_qna_to_avg_scores = qa_pairs_to_qna_to_avg_scores(base_qa_pairs)

    improvements, regressions, no_changes, new_qnas = [], [], [], []

    for qna, avg_score in qna_to_avg_scores.items():
        base_avg_score = base_qna_to_avg_scores.get(qna)
        if base_avg_score is not None:
            if avg_score > base_avg_score:
                improvements.append(
                    (
                        qna,
                        round(avg_score - base_avg_score, 2),
                        base_avg_score,
                        avg_score,
                    )
                )
            elif avg_score == base_avg_score:
                no_changes.append((qna, avg_score))
            else:
                regressions.append(
                    (
                        qna,
                        round(avg_score - base_avg_score, 2),
                        base_avg_score,
                        avg_score,
                    )
                )
        else:
            new_qnas.append((qna, avg_score))

    error_rate = (error_rate + base_error_rate) / 2
    if error_rate > 0:
        error_rate = round(error_rate, 2)

    summary = branch_eval_summary_to_json(
        improvements,
        regressions,
        no_changes,
        new_qnas,
    )

    mt_bench_branch_data = {
        "report_title": "SKILLS EVALUATION REPORT",
        "model": candidate_model,
        "judge_model": judge_model_name,
        "max_score": "10.0",
        "overall_score": overall_score,
        "base_overall_score": base_overall_score,
        "error_rate": error_rate,
        "summary": summary,
    }

    with open(mt_bench_branch_output, "w") as f:
        json.dump(mt_bench_branch_data, f, indent=4)
"""
    exec_run_final_eval_op_args = f"""
run_final_eval_op(mmlu_branch_output="{MMLU_BRANCH_SCORES_PATH}", mt_bench_branch_output="{MT_BENCH_BRANCH_SCORES_PATH}", candidate_model="{CANDIDATE_MODEL_PATH}", taxonomy="{TAXONOMY_PATH}", tasks="{DATA_PVC_SDG_PATH}", base_branch="", candidate_branch="", device=None, base_model_dir="{DATA_PVC_MODEL_PATH}", max_workers="{MAX_WORKERS}", merge_system_user_message={MERGE_SYSTEM_USER_MESSAGE}, model_dtype="{MODEL_DTYPE}", few_shots={FEW_SHOTS}, batch_size={BATCH_SIZE})
"""

    eval_container = kubernetes.client.V1Container(
        name=f"run-eval-{eval_type}",
        image=RHELAI_IMAGE,
        command=["/bin/sh", "-ce"],
        volume_mounts=get_vol_mount(),
        security_context=get_security_context(),
        env_from=[
            kubernetes.client.V1EnvFromSource(
                secret_ref=kubernetes.client.V1SecretEnvSource(
                    name=judge_serving_model_secret
                )
            ),
        ],
        resources=kubernetes.client.V1ResourceRequirements(
            requests={"cpu": "1", "nvidia.com/gpu": nproc_per_node},
            limits={"cpu": "1", "nvidia.com/gpu": nproc_per_node},
        ),
    )
    eval_args = {
        EVAL_TYPE_MT_BENCH: [
            PYTHON_EXECUTOR.format(
                python_code=exec_run_mt_bench_op_command,
                python_main=exec_run_mt_bench_op_args.strip(),
            ),
        ],
        EVAL_TYPE_FINAL: [
            PYTHON_EXECUTOR.format(
                python_code=exec_run_final_eval_op_command,
                python_main=exec_run_final_eval_op_args.strip(),
            ),
        ],
    }
    try:
        eval_container.args = eval_args[eval_type]
    except KeyError as exc:
        raise ValueError(f"Unknown evaluation type: {eval_type}") from exc

    init_containers = [eval_container]

    output_container = kubernetes.client.V1Container(
        name=f"output-eval-{eval_type}-scores",
        image=RHELAI_IMAGE,
        command=["/bin/sh", "-c"],
        security_context=get_security_context(),
        volume_mounts=get_vol_mount(),
    )
    eval_paths = {
        EVAL_TYPE_MT_BENCH: MT_BENCH_SCORES_PATH,
        EVAL_TYPE_FINAL: MT_BENCH_BRANCH_SCORES_PATH,
    }
    try:
        output_container.args = [f"cat {eval_paths[eval_type]}"]
    except KeyError as exc:
        raise ValueError(f"Unknown evaluation type: {eval_type}") from exc

    # Create and configure a spec section
    template = kubernetes.client.V1PodTemplateSpec(
        metadata=kubernetes.client.V1ObjectMeta(labels={"app": f"eval-{eval_type}"}),
        spec=kubernetes.client.V1PodSpec(
            restart_policy="Never",
            init_containers=init_containers,
            containers=[output_container],
            volumes=get_vol(),
        ),
    )

    if judge_serving_model_ca_cert:
        # Define the volume that references the ConfigMap
        cm_volume = kubernetes.client.V1Volume(
            name="judge-ca-cert-volume",
            config_map=kubernetes.client.V1ConfigMapVolumeSource(
                name=judge_serving_model_ca_cert
            ),
        )
        # Define the volume mount to specify where the Secret should be mounted in the container
        cm_volume_mount = kubernetes.client.V1VolumeMount(
            name="judge-ca-cert-volume",
            mount_path=JUDGE_CA_CERT_PATH,  # Path where the Secret will be mounted
        )
        # Add an env var to the container to specify the path to the CA cert
        eval_container.env = [
            kubernetes.client.V1EnvVar(
                name=JUDGE_CA_CERT_ENV_VAR_NAME,
                value=os.path.join(
                    JUDGE_CA_CERT_PATH, judge_serving_model_ca_cert_cm_key
                ),
            )
        ]
        # Add the volume to the Pod spec
        eval_container.volume_mounts.append(cm_volume_mount)
        # Add the volume mount to the container
        template.spec.volumes.append(cm_volume)

    # Create the specification of deployment
    spec = kubernetes.client.V1JobSpec(
        template=template,
    )

    # Instantiate the job object
    job = kubernetes.client.V1Job(
        api_version="batch/v1",
        kind="Job",
        metadata=kubernetes.client.V1ObjectMeta(name=job_name, namespace=namespace),
        spec=spec,
    )

    return job


def log_pod_containers(
    pod: kubernetes.client.V1Pod, container_type: str, namespace: str
):
    """
    Logs the output of containers in a given pod.

    Args:
        pod (kubernetes.client.V1Pod): The pod object containing the containers.
        container_type (str): The type of containers to log (e.g., 'containers', 'init_containers').
        namespace (str): The namespace in which the pod is located.

    Returns:
        None

    Logs:
        Logs the output of each container in the specified pod to the logger. If the container logs
        cannot be retrieved

    Raises:
        kubernetes.client.rest.ApiException: If there is an error other than a 400 status error when
        retrieving the logs. due to a 400 status error, it continues to the next container.
    """
    core_v1 = kubernetes.client.CoreV1Api()
    containers = getattr(pod.spec, container_type)
    if containers is None:
        return
    for container in containers:
        try:
            pod_log = core_v1.read_namespaced_pod_log(
                name=pod.metadata.name,
                namespace=namespace,
                container=container.name,
            )
            logger.error(
                "Logs for pod %s, %s %s:\n%s",
                pod.metadata.name,
                container_type[:-1],  # Remove the trailing 's'
                container.name,
                pod_log,
            )
        except kubernetes.client.rest.ApiException as exc:
            if exc.status == 400:
                continue


def run_job(namespace: str, job: kubernetes.client.V1Job) -> str:
    """
    Create and run a Kubernetes job in the specified namespace, and wait for its completion.

    Args:
        namespace (str): The namespace in which to create the job.
        job (kubernetes.client.V1Job): The job object to be created and run.

    Returns:
        str: The last container's logs.

    Prints:
        str: The status of the job during its execution.

    The function will print the job's status as it progresses and will stop watching once the job
    either succeeds or fails. If the job fails, it will also print the logs of the failed pod.
    """
    # Create a job
    batch_v1 = kubernetes.client.BatchV1Api()
    core_v1 = kubernetes.client.CoreV1Api()
    try:
        resp = batch_v1.create_namespaced_job(body=job, namespace=namespace)
        logger.info("Job created '%s/%s'", namespace, resp.metadata.name)
    except kubernetes.client.rest.ApiException as exc:
        if exc.status == 409:
            logger.info(
                "%s '%s/%s' already exists.",
                job.kind,
                namespace,
                job.metadata.name,
            )
        else:
            raise

    # It seems that the watcher suffers from a bug where it misses job events
    # https://github.com/kubernetes-client/python/issues/2238
    # Or connections are dropped
    # https://github.com/kubernetes-client/python/issues/2238
    # Once the library supports Informer API, we can switch to it
    # https://github.com/kubernetes-client/python/issues/868
    # Wait for the job to complete
    w = kubernetes.watch.Watch()
    pod_log = None
    exit_flag = False
    while not exit_flag:  # Keep the watch active
        try:
            for event in w.stream(
                batch_v1.list_namespaced_job,
                namespace=namespace,
                timeout_seconds=60,  # Timeout after 1 minutes
            ):
                job_event = event["object"]
                if job_event.metadata.name != job.metadata.name:
                    continue

                logger.info("Job: %s - %s", job.metadata.name, job_event.status)

                # Handle job completion (successful or failed)
                if job_event.status.succeeded == 1:
                    logger.info("Job completed successfully.")
                    pods = core_v1.list_namespaced_pod(
                        namespace=namespace,
                        label_selector=f"app={job.spec.template.metadata.labels['app']}",
                    )
                    if pods.items:
                        pod_log = core_v1.read_namespaced_pod_log(
                            name=pods.items[0].metadata.name, namespace=namespace
                        )
                    else:
                        logger.error(
                            "No pods found for job %s. The job exists, but the pods are missing.",
                            job.metadata.name,
                        )
                        pod_log = None
                    w.stop()
                    exit_flag = True  # Set the flag to exit the outer loop
                    break

                elif job_event.status.failed == 1:
                    logger.error("Job failed. Pod logs:")
                    pods = core_v1.list_namespaced_pod(
                        namespace=namespace,
                        label_selector=f"app={job.spec.template.metadata.labels['app']}",
                    )
                    for pod in pods.items:
                        log_pod_containers(pod, "init_containers", namespace)
                        log_pod_containers(pod, "containers", namespace)
                    w.stop()
                    raise RuntimeError("Job failed.")

                else:
                    logger.info(
                        "Job '%s' is still running. Waiting for the next event.",
                        job.metadata.name,
                    )

        except kubernetes.client.exceptions.ApiException as e:
            logger.error("API exception occurred: %s", str(e))
            time.sleep(5)  # Backoff before retrying
        except urllib3.exceptions.ProtocolError as e:
            logger.warning("Connection broken reconnecting the watcher: %s", str(e))
            time.sleep(5)  # Backoff before retrying

        finally:
            w.stop()  # Ensure the watch is stopped after each try

    # Ensure pod logs are returned after success
    return pod_log


def create_pvc(
    name: str,
    namespace: str,
    storage_class: str,
    access_modes: list,
    size: str,
) -> kubernetes.client.V1PersistentVolumeClaim:
    """
    Create a PersistentVolumeClaim (PVC) in the specified namespace.

    Args:
        namespace (str): The namespace in which to create the PVC.
        storage_class (str): The storage class for the PVC.
        access_modes (list): The access modes for the PVC.
        size (str): The size of the PVC.

    Returns:
        kubernetes.client.V1PersistentVolumeClaim: The created PVC object.
    """
    # Create a PVC
    return kubernetes.client.V1PersistentVolumeClaim(
        metadata=kubernetes.client.V1ObjectMeta(name=name, namespace=namespace),
        spec=kubernetes.client.V1PersistentVolumeClaimSpec(
            access_modes=access_modes,
            storage_class_name=storage_class,
            resources=kubernetes.client.V1ResourceRequirements(
                requests={"storage": size}
            ),
        ),
    )


@run.command(name="sdg")
@click.pass_context
def sdg(
    ctx: click.Context,
) -> None:
    """
    Preprocesses SDG data by creating a Persistent Volume Claim (PVC) and
    initiating a job to run a pod for SDG data preprocessing.

    Steps:
        1. Creates a PVC to hold SDG data and transformed SDG data.
        2. Initiates a job to run a pod for SDG data preprocessing.
    """
    # Populate variables from context
    namespace = ctx.obj["namespace"]
    taxonomy_repo_url = ctx.obj["taxonomy_repo_url"]
    taxonomy_repo_branch = ctx.obj["taxonomy_repo_branch"]
    taxonomy_repo_pr = ctx.obj["taxonomy_repo_pr"]
    storage_class = ctx.obj["storage_class"]
    serving_endpoint = ctx.obj["serving_endpoint"]
    serving_model = ctx.obj["serving_model"]
    num_instructions_to_generate = ctx.obj["num_instructions_to_generate"]

    # check in the context
    if not taxonomy_repo_branch and not taxonomy_repo_pr:
        raise ValueError(
            "Either '--taxonomy-repo-branch' or '--taxonomy-repo-pr' "
            "must be provided to the 'run' command."
        )

    logger.info("Running setup for SDG.")
    # Request the Kubernetes API
    v1 = kubernetes.client.CoreV1Api()

    # list of PVCs to create and their details
    pvcs = [
        {
            "name": DATA_PVC_NAME,
            "namespace": namespace,
            "storage_class": storage_class,
            "access_modes": ["ReadWriteMany"],
            "size": "200Gi",
        },
    ]
    for pvc in pvcs:
        try:
            v1.create_namespaced_persistent_volume_claim(
                namespace=namespace, body=create_pvc(**pvc)
            )
            logger.info("Successfully created PVC '%s' created.", pvc.get("name"))
        except kubernetes.client.rest.ApiException as exc:
            if exc.status == 409:
                logger.info("PVC '%s' already exists.", pvc["name"])
            else:
                raise

    # Create the job to run the pod to execute the SDG data preprocessing
    # Example usage
    job = create_sdg_job(
        namespace=namespace,
        job_name="sdg",
        exec_git_clone_op_repo_url=taxonomy_repo_url,
        exec_git_clone_op_repo_branch=taxonomy_repo_branch,
        exec_git_clone_op_repo_pr=taxonomy_repo_pr,
        num_instructions_to_generate=num_instructions_to_generate,
    )
    run_job(namespace, job)
    logger.info("SDG setup completed.")


def validate_url(url: str) -> str:
    """
    Validate if the given string is a valid URL.

    Args:
        url (str): The URL string to validate.

    Returns:
        str: The original URL if valid.

    Raises:
        ValueError: If the URL is not valid.
    """
    parsed = urlparse(url)
    if not all([parsed.scheme, parsed.netloc]):
        raise ValueError(f"Invalid URL: {url}")
    return url


@run.command(name="sdg-data-fetch")
@click.pass_context
def sdg_data_fetch(
    ctx: click.Context,
) -> None:
    """
    Fetches SDG data from an object store and put in a Persistent Volume Claim (PVC)
    """
    # Populate variables from context
    namespace = ctx.obj["namespace"]
    storage_class = ctx.obj["storage_class"]
    judge_serving_model_endpoint = ctx.obj["judge_serving_model_endpoint"]
    judge_serving_model_name = ctx.obj["judge_serving_model_name"]
    judge_serving_model_api_key = ctx.obj["judge_serving_model_api_key"]
    judge_serving_model_ca_cert = ctx.obj["judge_serving_model_ca_cert"]
    judge_serving_model_ca_cert_cm_key = ctx.obj["judge_serving_model_ca_cert_cm_key"]
    judge_serving_model_secret = ctx.obj["judge_serving_model_secret"]
    sdg_object_store_endpoint = ctx.obj["sdg_object_store_endpoint"]
    sdg_object_store_bucket = ctx.obj["sdg_object_store_bucket"]
    sdg_object_store_access_key = ctx.obj["sdg_object_store_access_key"]
    sdg_object_store_secret_key = ctx.obj["sdg_object_store_secret_key"]
    sdg_object_store_region = ctx.obj["sdg_object_store_region"]
    sdg_object_store_data_key = ctx.obj["sdg_object_store_data_key"]
    sdg_object_store_verify_tls = ctx.obj["sdg_object_store_verify_tls"]
    sdg_object_store_secret = ctx.obj["sdg_object_store_secret"]
    force_pull = ctx.obj["force_pull"]
    dry_run = ctx.obj["dry_run"]

    # Check if all required arguments are provided for Data Fetch
    if not sdg_object_store_secret:
        if not all(
            [
                sdg_object_store_bucket,
                sdg_object_store_access_key,
                sdg_object_store_secret_key,
                sdg_object_store_data_key,
            ]
        ):
            # Endpoint is optional if AWS S3 is used
            raise ValueError(
                "All of '--sdg-object-store-bucket', "
                "'--sdg-object-store-access-key', '--sdg-object-store-secret-key', "
                "'--sdg-object-store-data-key' "
                "must be provided to the 'sdg-data-fetch' command. Alternatively, provide "
                "'--sdg-object-store-secret' to use a Kubernetes Secret."
            )

    # Check if all required arguments are provided for Evaluation
    if not judge_serving_model_secret:
        if not all(
            [
                judge_serving_model_endpoint,
                judge_serving_model_name,
                judge_serving_model_api_key,
            ]
        ):
            # Endpoint is optional if AWS S3 is used
            raise ValueError(
                "All of '--judge-serving-model-endpoint', "
                "'--judge-serving-model-api-key', '--judge-serving-model-name', "
                "must be provided to the 'sdg-data-fetch' command. Alternatively, provide "
                "'--judge-serving-model-secret' to use a Kubernetes Secret."
            )

    logger.info("Running setup for SDG data fetch.")

    # Request the Kubernetes API
    v1 = kubernetes.client.CoreV1Api()

    # SDG Data Fetch secret
    if (
        # Endpoint (if AWS S3 is used) and Region are optional
        all(
            [
                sdg_object_store_bucket,
                sdg_object_store_access_key,
                sdg_object_store_secret_key,
                sdg_object_store_data_key,
            ]
        )
        and not sdg_object_store_secret
    ):
        sdg_object_store_secret = SDG_OBJECT_STORE_SECRET_NAME
        secret = kubernetes.client.V1Secret(
            metadata=kubernetes.client.V1ObjectMeta(
                name=sdg_object_store_secret, namespace=namespace
            ),
            string_data={
                "bucket": sdg_object_store_bucket,
                "access_key": sdg_object_store_access_key,
                "secret_key": sdg_object_store_secret_key,
                "data_key": sdg_object_store_data_key,
            },
        )

        # Endpoint is optional if AWS S3 is used
        if sdg_object_store_endpoint:
            validate_url(sdg_object_store_endpoint)
            secret.string_data["endpoint"] = sdg_object_store_endpoint

        # Region is optional
        if sdg_object_store_region:
            secret.string_data["region"] = sdg_object_store_region

        if sdg_object_store_verify_tls:
            secret.string_data["verify_tls"] = "true"
        else:
            secret.string_data["verify_tls"] = "false"

        try:
            if dry_run:
                logger.info(
                    "Dry run: Secret would be created.\n%s", secret.metadata.name
                )
            else:
                v1.create_namespaced_secret(namespace=namespace, body=secret)
        except kubernetes.client.rest.ApiException as exc:
            if exc.status == 409:
                logger.info("Secret '%s' already exists.", secret.metadata.name)
            else:
                raise

    # If the secret option is used, verify the presence of the keys and the existence of the secret
    elif sdg_object_store_secret:
        if not dry_run:
            try:
                secret = v1.read_namespaced_secret(
                    name=sdg_object_store_secret, namespace=namespace
                )

                def decode_base64(data):
                    return base64.b64decode(data).decode("utf-8")

                if secret.data.get("endpoint"):
                    endpoint = decode_base64(secret.data.get("endpoint"))
                    validate_url(endpoint)

                if not all(
                    [
                        secret.data.get("bucket"),
                        secret.data.get("access_key"),
                        secret.data.get("secret_key"),
                        secret.data.get("data_key"),
                    ]
                ):
                    raise ValueError(
                        f"The provided secret {sdg_object_store_secret} must contain the keys:"
                        "'bucket', 'access_key', 'secret_key', 'data_key'.",
                    )
            except kubernetes.client.rest.ApiException as exc:
                if exc.status == 404:
                    raise ValueError(
                        f"Secret {sdg_object_store_secret} not found in namespace {namespace}."
                    ) from exc

    # Judge serving model secret
    if (
        all(
            [
                judge_serving_model_endpoint,
                judge_serving_model_name,
                judge_serving_model_api_key,
            ]
        )
        and not judge_serving_model_secret
    ):
        judge_serving_model_secret = JUDGE_SERVING_NAME
        secret = kubernetes.client.V1Secret(
            metadata=kubernetes.client.V1ObjectMeta(
                name=judge_serving_model_secret, namespace=namespace
            ),
            string_data={
                "JUDGE_API_KEY": judge_serving_model_api_key,
                "JUDGE_ENDPOINT": judge_serving_model_endpoint,
                "JUDGE_NAME": judge_serving_model_name,
            },
        )

        try:
            if dry_run:
                logger.info(
                    "Dry run: Secret would be created.\n%s", secret.metadata.name
                )
                print(secret)
            else:
                v1.create_namespaced_secret(namespace=namespace, body=secret)
        except kubernetes.client.rest.ApiException as exc:
            if exc.status == 409:
                logger.info("Secret '%s' already exists.", secret.metadata.name)
            else:
                raise

    # If the secret option is used, verify the presence of the keys and the existence of the secret
    elif judge_serving_model_secret:
        if not dry_run:
            try:
                secret = v1.read_namespaced_secret(
                    name=judge_serving_model_secret, namespace=namespace
                )

                if not all(
                    [
                        secret.data.get("JUDGE_API_KEY"),
                        secret.data.get("JUDGE_ENDPOINT"),
                        secret.data.get("JUDGE_NAME"),
                    ]
                ):
                    raise ValueError(
                        f"The provided secret {judge_serving_model_secret} must contain the keys:"
                        "'JUDGE_API_KEY', 'JUDGE_ENDPOINT', 'JUDGE_NAME' mind the uppercase.",
                    )

                judge_serving_model_endpoint = decode_base64(
                    secret.data.get("JUDGE_ENDPOINT")
                )
                validate_url(judge_serving_model_endpoint)

                # Validation of the configmap's existence is done in the next conditional block
                if secret.data.get("JUDGE_CA_CERT"):
                    judge_serving_model_ca_cert = decode_base64(
                        secret.data.get("JUDGE_CA_CERT")
                    )
                if secret.data.get("JUDGE_CA_CERT_CM_KEY"):
                    judge_serving_model_ca_cert_cm_key = decode_base64(
                        secret.data.get("JUDGE_CA_CERT_CM_KEY")
                    )
            except kubernetes.client.rest.ApiException as exc:
                if exc.status == 404:
                    raise ValueError(
                        f"Secret {judge_serving_model_secret} not found in namespace {namespace}."
                    ) from exc

    # If the CA cert is provided, verify the existence of the secret
    # We don't add the CA Cert Secret name into the Secret that contains the judge details
    # If provided, the Secret will be mounted as a volume in the evaluation job
    if judge_serving_model_ca_cert and not dry_run:
        try:
            cm = v1.read_namespaced_config_map(
                name=judge_serving_model_ca_cert, namespace=namespace
            )
            # Validate the presence of the key
            if not cm.data.get(judge_serving_model_ca_cert_cm_key):
                raise ValueError(
                    f"Provided ConfigMap {judge_serving_model_ca_cert} does not contain the key:"
                    f"'{judge_serving_model_ca_cert_cm_key}'."
                    "Use '--judge-serving-model-ca-cert-cm-key' to specify the key."
                )
        except kubernetes.client.rest.ApiException as exc:
            if exc.status == 404:
                raise ValueError(
                    f"ConfigMap {judge_serving_model_ca_cert} not found in namespace {namespace}."
                ) from exc

    # Set the judge secret in the context for the evaluation job
    ctx.obj["judge_serving_model_secret"] = judge_serving_model_secret

    # Set the judge CA cert in the context for the evaluation job, this handles the case where the
    # secret is not provided via the cli flag but inside the secret
    ctx.obj["judge_serving_model_ca_cert"] = judge_serving_model_ca_cert
    ctx.obj["judge_serving_model_ca_cert_cm_key"] = judge_serving_model_ca_cert_cm_key

    # list of PVCs to create and their details
    pvcs = [
        {
            "name": DATA_PVC_NAME,
            "namespace": namespace,
            "storage_class": storage_class,
            "access_modes": ["ReadWriteMany"],
            "size": "200Gi",  # Allocate size for a few models and large SDG data sets
        },
    ]
    for pvc in pvcs:
        try:
            if dry_run:
                logger.info("Dry run: PVC would be created.\n%s", create_pvc(**pvc))
            else:
                v1.create_namespaced_persistent_volume_claim(
                    namespace=namespace, body=create_pvc(**pvc)
                )
                logger.info("Successfully created PVC '%s' created.", pvc.get("name"))
        except kubernetes.client.rest.ApiException as exc:
            if exc.status == 409:
                logger.info("PVC '%s' already exists.", pvc["name"])
            else:
                raise

    # Create the job to run the pod to execute the SDG data fetch
    job = create_data_job(
        namespace=namespace,
        job_name="data-download",
        sdg_object_store_secret=sdg_object_store_secret,
        strategy="download",
        force_pull=force_pull,
    )

    if dry_run:
        logger.info("Dry run: Job would be created.\n%s", job)
        return

    # Run the job
    run_job(namespace, job)
    logger.info(
        "SDG data, model to train and taxonomy tree were successfully downloaded."
    )


@run.command(name="train")
@click.pass_context
def train(
    ctx: click.Context,
) -> None:
    """
    Run the distributed training.
    """
    namespace = ctx.obj["namespace"]
    training_phase = ctx.obj["training_phase"]
    path_to_model = ctx.obj["model_to_train"]
    nproc_per_node: int = ctx.obj["nproc_per_node"]
    training_1_epoch_num: int = ctx.obj["training_1_epoch_num"]
    training_2_epoch_num: int = ctx.obj["training_2_epoch_num"]
    dry_run = ctx.obj["dry_run"]

    if training_phase is None:
        raise ValueError("Training phase must be provided with --training-phase=[1|2]")

    # During the initial training
    if path_to_model is None:
        path_to_model = DATA_PVC_MODEL_PATH

    epoch_num = None
    if training_phase == "1":
        epoch_num = training_1_epoch_num
    elif training_phase == "2":
        epoch_num = training_2_epoch_num

    logger.info("Running multi-phased distributed training phase %s", training_phase)
    worker_replicas = PYTORCH_NNODES - 1
    pytorch_training_job_yaml = yaml.safe_load(
        PYTORCH_TRAINING_JOB.format(
            name=f"train-phase-{training_phase}",
            data_pvc_name=DATA_PVC_NAME,
            path_to_model=path_to_model,
            nproc_per_node=nproc_per_node,
            nnodes=PYTORCH_NNODES,
            image=RHELAI_IMAGE,
            worker_replicas=worker_replicas,
            epoch_num=epoch_num,
            phase_num=training_phase,
            data_pvc_model_path=DATA_PVC_MODEL_PATH,
            data_pvc_sdg_path=DATA_PVC_SDG_PATH,
            preprocessed_data_skills_path=PREPROCESSED_DATA_SKILLS_PATH,
            preprocessed_data_knowledge_path=PREPROCESSED_DATA_KNOWLEDGE_PATH,
        )
    )

    if dry_run:
        logger.info(
            "Dry run: PytorchJob would be created.\n%s", pytorch_training_job_yaml
        )
        return

    api = kubernetes.client.CustomObjectsApi()

    try:
        api.create_namespaced_custom_object(
            group="kubeflow.org",
            version="v1",
            namespace=namespace,
            plural="pytorchjobs",
            body=pytorch_training_job_yaml,
        )
    except kubernetes.client.rest.ApiException as exc:
        if exc.status == 409:
            logger.info(
                "%s '%s/%s' already exists.",
                pytorch_training_job_yaml["kind"],
                namespace,
                pytorch_training_job_yaml["metadata"]["name"],
            )
        else:
            raise

    # Get the CR status and wait for it to be completed
    core_v1 = kubernetes.client.CoreV1Api()
    w = kubernetes.watch.Watch()
    exit_flag = False
    # TODO: this block is getting really deep, would be nice to refactor one day
    while not exit_flag:  # Keep the watch active
        try:
            logger.info("Watching for PytorchJob")
            for event in w.stream(
                api.list_namespaced_custom_object,
                group="kubeflow.org",
                version="v1",
                namespace=namespace,
                plural="pytorchjobs",
                timeout_seconds=60,  # Timeout after 1 minutes
            ):
                pytorchjob_event = event["object"]
                if (
                    pytorchjob_event["metadata"]["name"]
                    != pytorch_training_job_yaml["metadata"]["name"]
                ):
                    continue
                pytorchjob_name = pytorchjob_event["metadata"]["name"]

                if (
                    "status" not in pytorchjob_event
                    or "conditions" not in pytorchjob_event["status"]
                ):
                    continue
                logger.info(
                    "PytorchJob: %s - %s",
                    pytorchjob_name,
                    pytorchjob_event["status"].get("conditions", "No conditions yet"),
                )
                for job_condition in reversed(pytorchjob_event["status"]["conditions"]):
                    if job_condition["type"] == "Running":
                        try:
                            # List PytorchJob Pods
                            pods = core_v1.list_namespaced_pod(
                                namespace=namespace,
                                label_selector=(
                                    f"training.kubeflow.org/job-name=train-phase-{training_phase}"
                                ),
                            )
                            for pod_event in pods.items:
                                if pod_event.metadata.name.startswith(pytorchjob_name):
                                    logger.info(
                                        "Pod: %s - %s",
                                        pod_event.metadata.name,
                                        pod_event.status.phase,
                                    )
                                    # First look if any container is in CrashLoopBackOff
                                    for (
                                        container_status
                                    ) in pod_event.status.container_statuses:
                                        # We fail on CrashLoopBackOff and not on Error, allowing
                                        # for retries
                                        if (
                                            container_status.state.waiting
                                            and container_status.state.waiting.reason
                                            == "CrashLoopBackOff"
                                        ):
                                            log_pod_containers(
                                                pod_event,
                                                "init_containers",
                                                namespace,
                                            )
                                            log_pod_containers(
                                                pod_event, "containers", namespace
                                            )
                                            raise RuntimeError(
                                                f"Pod {pod_event.metadata.name} failed."
                                            )

                                    # If the pod is in a failed state, log the containers and
                                    # stop the watcher
                                    if pod_event.status.phase == "Failed":
                                        log_pod_containers(
                                            pod_event, "init_containers", namespace
                                        )
                                        log_pod_containers(
                                            pod_event, "containers", namespace
                                        )
                                        w.stop()
                                        raise RuntimeError(
                                            f"Pod {pod_event.metadata.name} failed."
                                        )
                        except kubernetes.client.exceptions.ApiException as e:
                            logger.error("API exception occurred: %s", str(e))
                            time.sleep(5)  # Backoff before retrying
                    elif job_condition["type"] == "Succeeded":
                        logger.info(
                            "PytorchJob '%s' completed successfully: %s",
                            pytorchjob_name,
                            job_condition["reason"],
                        )
                        logger.info("Training phase %s completed.", training_phase)
                        w.stop()
                        exit_flag = True
                        # Break here to avoid going into other conditions, we are done
                        break
                    elif job_condition["type"] == "Failed":
                        logger.error(
                            "PytorchJob' %s' failed: %s",
                            pytorchjob_name,
                            job_condition["reason"],
                        )
                        w.stop()
                        raise RuntimeError("Job failed.")
        except kubernetes.client.exceptions.ApiException as e:
            logger.error("API exception occurred: %s", str(e))
            time.sleep(5)  # Backoff before retrying
        # Catches the following error:
        # urllib3.exceptions.ProtocolError: ("Connection broken: InvalidChunkLength
        except urllib3.exceptions.ProtocolError as e:
            logger.warning("Connection broken reconnecting the watcher %s", str(e))
            time.sleep(5)  # Backoff before retrying
        finally:
            w.stop()


@run.command(name="evaluation")
@click.pass_context
def evaluation(ctx: click.Context) -> str:
    """
    Run the evaluation phase and return the scores as a JSON string.

    Args:
        ctx (click.Context): The Click context object.
        eval_type (str): The type of evaluation to run.

    Returns:
        str: The evaluation scores as a JSON string.
    """
    namespace = ctx.obj["namespace"]
    eval_type = ctx.obj["eval_type"]
    dry_run = ctx.obj["dry_run"]
    judge_serving_model_secret = ctx.obj["judge_serving_model_secret"]
    judge_serving_model_ca_cert = ctx.obj["judge_serving_model_ca_cert"]
    judge_serving_model_ca_cert_cm_key = ctx.obj["judge_serving_model_ca_cert_cm_key"]

    # This should only happen if the script is called with the "evaluation" subcommand
    if not judge_serving_model_secret:
        raise ValueError(
            "Judge serving model secret must be provided with --judge-serving-model-secret."
        )

    if eval_type is None:
        raise ValueError(
            "Evaluation type must be provided with --eval-type=[mt-bench|final]"
        )

    logger.info("Running %s evaluation.", eval_type)

    # Create and run the evaluation job
    job = create_eval_job(
        namespace=namespace,
        eval_type=eval_type,
        judge_serving_model_secret=judge_serving_model_secret,
        judge_serving_model_ca_cert=judge_serving_model_ca_cert,
        judge_serving_model_ca_cert_cm_key=judge_serving_model_ca_cert_cm_key,
    )

    if dry_run:
        logger.info("Dry run: Job would be created.\n%s", job)
        return

    scores = run_job(namespace, job)

    if eval_type == EVAL_TYPE_MT_BENCH:
        scores = scores.replace("'", '"')
        try:
            scores_data = json.loads(scores)
            if isinstance(scores_data, dict):
                scores = json.dumps(scores_data)
            else:
                raise ValueError("Unexpected format for scores data")
        except json.JSONDecodeError as e:
            logger.error("Failed to parse scores: %s", e)
            raise

        return scores

    logger.info("Evaluation scores: %s", scores)


@run.command(name="upload-trained-model")
@click.pass_context
def upload_trained_model(ctx: click.Context):
    """
    Uploads the trained model back to the object store.

    This function retrieves the namespace and SDG object store secret from the
    provided Click context object. It then creates and runs a data job to
    upload the trained model to the object store.

    Args:
        ctx (click.Context): The Click context object containing command-line
                             parameters and options.

    Returns:
        None

    Raises:
        ValueError: If the SDG object store secret is not provided.
    """
    namespace = ctx.obj["namespace"]
    # At this stage the secret is present from previous phases so no need to check
    sdg_object_store_secret = ctx.obj["sdg_object_store_secret"]
    dry_run = ctx.obj["dry_run"]

    logger.info("Uploading the trained model back to the object store.")
    job = create_data_job(
        namespace=namespace,
        job_name="trained-model-upload",
        sdg_object_store_secret=sdg_object_store_secret,
        strategy="upload",
    )

    if dry_run:
        logger.info("Dry run: Job would be created.\n%s", job)
        return

    # Run the job
    run_job(namespace, job)
    logger.info("Successfully uploaded newly trained model back to the object store.")


if __name__ == "__main__":
    # Configs can be set in Configuration class directly or using helper utility
    try:
        kubernetes.config.load_kube_config()
    except kubernetes.config.ConfigException:
        logger.info("Failed to load kube config. Trying in-cluster config")
        kubernetes.config.load_incluster_config()

    cli()
