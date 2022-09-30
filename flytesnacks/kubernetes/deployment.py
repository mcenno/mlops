import logging


def delete_deployment_imperatively(manifest: dict) -> bool:
    """
    Delete a deployment imperatively.

    Args:
        manifest (dict): deployment manifest
    Returns:
        True if the deletion was succesfull, False otherwise.
    """
    from kubernetes import client, config

    try:
        config.load_incluster_config()
    except:
        config.load_kube_config()

    api = api = client.AppsV1Api()
    name = manifest["metadata"]["name"]
    namespace = manifest["metadata"]["namespace"]

    logging.info(f"Delete deployment '{name}' in namespace '{namespace}'.")

    api_response = api.delete_namespaced_deployment(name, namespace, grace_period_seconds=0)

    return hasattr(api_response, "status") and api_response.status == "Success"


def deploy_deployment_imperatively(manifest: dict, timeout: int = 300) -> bool:
    """
    Imperatively create a kubernetes deployment and wait for it to become ready.

    Args:
        manifest (dict): deployment manifest

    Kwargs:
        timeout (int): number of seconds to wait for succesfull creation

    Returns:
        True if the deployment could be created and started successfully
        False otherwise
    """
    from time import sleep, time
    from kubernetes import client, config

    try:
        config.load_incluster_config()
    except:
        config.load_kube_config()

    api = client.AppsV1Api()

    name = manifest["metadata"]["name"]
    namespace = manifest["metadata"]["namespace"]

    logging.info(f"Create deployment '{name}' in namespace '{namespace}'.")

    api.create_namespaced_deployment(namespace=namespace, body=manifest)

    poll_interval = 5  # [seconds]
    timeout_start = time()

    goal_condition = {"reason": "MinimumReplicasAvailable", "status": "True", "type": "Available"}

    while True:
        if time() < timeout_start + timeout:
            status = api.read_namespaced_deployment_status(name=name, namespace=namespace)

            if status is not None and hasattr(status, "status"):
                status = status.status
                if (
                    hasattr(status, "conditions")
                    and status.conditions is not None
                    and any(
                        [
                            all([hasattr(c, k) and getattr(c, k) == v for k, v in goal_condition.items()])
                            for c in status.conditions
                        ]
                    )
                ):
                    return True
                else:
                    logging.info(f"Waiting for deployment condition to become '{goal_condition}'.")
            else:
                logging.info("Deployment status unavailable.")

            sleep(poll_interval)
        else:
            logging.warning(f"Deployment didn't reach the goal state in {timeout}s. Deleting the deployment.")
            delete_deployment_imperatively(manifest)
            return False
