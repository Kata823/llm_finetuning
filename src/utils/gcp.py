# Third Party
from google.cloud import secretmanager


def access_secret_version(project_id: str, secret_id: str, version_id: str) -> str:
    """Access secret from secret manager.
    Args:
        project_id (str): Project ID of the secret manager.
        secret_id (str): Secret ID inside the secret manager of the target secret.
        version_id (str): Version ID of the secret.
    Returns:
        str: Return retrieved secret.
    """
    client = secretmanager.SecretManagerServiceClient()

    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    response = client.access_secret_version(request={"name": name})

    payload = response.payload.data.decode("UTF-8")

    return payload