import logging


def deploy_service_imperatively(manifest: dict) -> bool:
    """
    Imperatively create a kubernetes service.

    Args:
        manifest (dict): service manifest

    Returns:
        True
    """
    from kubernetes import client, config

    try:
        config.load_incluster_config()
    except:
        config.load_kube_config()

    api = client.CoreV1Api()

    name = manifest["metadata"]["name"]
    namespace = manifest["metadata"]["namespace"]

    logging.info(f"Create service '{name}' in namespace '{namespace}'.")

    api_response = api.create_namespaced_service(namespace=namespace, body=manifest)

    return True


def delete_service_imperatively(manifest: dict) -> bool:
    """
    Imperatively delete a kubernetes service.

    Args:
        manifest (dict): service manifest

    Returns:
        True if the deletion was succesfull, False otherwise.
    """
    from kubernetes import client, config

    try:
        config.load_incluster_config()
    except:
        config.load_kube_config()

    api = client.CoreV1Api()

    name = manifest["metadata"]["name"]
    namespace = manifest["metadata"]["namespace"]

    logging.info(f"Delete service '{name}' in namespace '{namespace}'.")

    api_response = api.delete_namespaced_service(name, namespace=namespace)

    return hasattr(api_response, "status") and api_response.status == "Success"
