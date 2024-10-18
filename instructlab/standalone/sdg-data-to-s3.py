#!/usr/bin/env python3

import os
import pathlib
from urllib.parse import urlparse

import boto3
import click


@click.group()
def cli():
    """
    Command Line Interface (CLI) entry point.

    This function serves as the main entry point for the command line interface.
    It currently does not perform any operations.
    """


@click.option(
    "--object-store-endpoint",
    envvar="OBJECT_STORE_ENDPOINT",
    help=(
        "Object store endpoint for SDG if different than the official AWS S3 endpoint. "
        "Expects an URL. TLS with self-signed certificates is not supported. (SDG_OBJECT_STORE_ENDPOINT env var)"
        "e.g. https://s3.openshift-storage.svc:443"
        "Don't forget the URL scheme (http/https) and the port"
    ),
    type=str,
)
@click.option(
    "--object-store-bucket",
    envvar="OBJECT_STORE_BUCKET",
    help="Object store bucket containing SDG data. (SDG_OBJECT_STORE_BUCKET env var)",
    type=str,
    required=True,
)
@click.option(
    "--object-store-access-key",
    envvar="OBJECT_STORE_ACCESS_KEY",
    help="Object store access key for SDG. (SDG_OBJECT_STORE_ACCESS_KEY env var)",
    type=str,
    required=True,
)
@click.option(
    "--object-store-secret-key",
    envvar="OBJECT_STORE_SECRET_KEY",
    help="Object store secret key for SDG. (SDG_OBJECT_STORE_SECRET_KEY env var)",
    type=str,
    required=True,
)
@click.option(
    "--object-store-region",
    envvar="OBJECT_STORE_REGION",
    help="Region for the object store. (SDG_OBJECT_STORE_REGION env var)",
    type=str,
)
@click.option(
    "--sdg-data-archive-file-path",
    help=(
        "Path to the tarball that contains SDG data."
        "The tarball MUST NOT contain a top-level directory. "
        "To archive your SDG data, use the following command: "
        "cd /path/to/data && tar -czvf sdg.tar.gz *"
    ),
    type=pathlib.Path,
    required=True,
)
@click.option(
    "--object-store-verify-tls",
    envvar="OBJECT_STORE_VERIFY_TLS",
    help="Verify TLS for the object store. (SDG_OBJECT_STORE_VERIFY_TLS env var).",
    default=True,
    type=bool,
)
@click.command()
def upload(
    object_store_endpoint: str,
    object_store_bucket: str,
    object_store_access_key: str,
    object_store_secret_key: str,
    object_store_region: str,
    sdg_data_archive_file_path: str,
    object_store_verify_tls: bool,
):
    """
    Push data to an S3-compatible object store.
    Args:
        object_store_endpoint (str): The endpoint URL of the object store.
        object_store_bucket (str): The name of the bucket in the object store.
        object_store_access_key (str): The access key for the object store.
        object_store_secret_key (str): The secret key for the object store.
        object_store_region (str): The region where the object store is located.
        sdg_data_archive_file_path (str): The file path of the SDG data archive to be uploaded.
        object_store_verify_tls (bool): Whether to verify TLS certificates when connecting to the
        object store.

    Returns:
        None
    """

    click.echo("Pushing data to S3...")
    if object_store_endpoint:
        validate_url(object_store_endpoint)

    s3 = build_boto3_client(
        object_store_access_key,
        object_store_secret_key,
        object_store_endpoint,
        object_store_region,
        object_store_verify_tls,
    )

    s3_key = os.path.basename(sdg_data_archive_file_path)
    s3.upload_file(sdg_data_archive_file_path, object_store_bucket, s3_key)


def build_boto3_client(
    object_store_access_key: str,
    object_store_secret_key: str,
    object_store_endpoint: str = None,
    object_store_region: str = None,
    object_store_verify_tls: bool = True,
):
    """
    Creates and returns a boto3 S3 client.

    Parameters:
    object_store_access_key (str): The access key for the object store.
    object_store_secret_key (str): The secret key for the object store.
    object_store_endpoint (str, optional): The endpoint URL for the object store. Defaults to None.
    object_store_region (str, optional): The region name for the object store. Defaults to None.
    object_store_verify_tls (bool, optional): Whether to verify TLS certificates. Defaults to True.

    Returns:
    boto3.client: A boto3 S3 client configured with the provided credentials and settings.
    """
    return boto3.client(
        "s3",
        aws_access_key_id=object_store_access_key,
        aws_secret_access_key=object_store_secret_key,
        endpoint_url=object_store_endpoint,
        region_name=object_store_region,
        verify=object_store_verify_tls,
    )


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


cli.add_command(upload)

if __name__ == "__main__":
    cli()
