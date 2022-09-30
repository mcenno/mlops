"""Create and delete custom resources imperatively."""

import logging
from typing import Tuple


logging.getLogger().setLevel(logging.INFO)


def delete_custom_resource_imperatively(manifest: dict) -> bool:
    """
    Delete a custom resource imperatively.

    Args:
        manifest (dict):
    Returns:
        True if the deletion was succesfull, False otherwise.
    """
    from kubernetes import client, config

    try:
        config.load_incluster_config()
    except:
        config.load_kube_config()

    api = client.CustomObjectsApi()
    name = manifest["metadata"]["name"]
    namespace = manifest["metadata"]["namespace"]
    group, version = manifest["apiVersion"].split("/")
    plural = (manifest["kind"] + "s").lower()

    logging.info(f"Delete custom resource object '{name}' of type '{manifest['kind']}' in namespace '{namespace}'.")

    api_response = api.delete_namespaced_custom_object(group, version, namespace, plural, name, grace_period_seconds=0)

    return api_response.get("status") == "Success"


def deploy_custom_resource_imperatively(
    manifest: dict, goal_condition: dict = None, goal_state: Tuple[str, str] = None, timeout: int = 300
) -> bool:
    """
    Imperatively create a custom resource and wait for it to become ready.

    Use either `goal_condition` or `goal_state`.

    Args:
        manifest (dict): custom resource manifest
    Kwargs:
        goal_condition (dict): condition to be met for the creation to be succesfull
           e.g in the form `{"type": "Ready", "status": "True"}`
        goal_state (Tuple): state to be met for the creation to be succesfull
           e.g. in the form `("state", "Available")`
        timeout (int): number of seconds to wait for succesfull creation

    Returns:
        True if the custom resource could be created and the goal_condition or
           goal_state has been achieved
        False otherwise
    """
    from time import sleep, time
    from kubernetes import client, config

    assert not (goal_condition is not None and goal_state is not None)
    assert goal_condition is not None or goal_state is not None

    try:
        config.load_incluster_config()
    except:
        config.load_kube_config()

    api = client.CustomObjectsApi()
    name = manifest["metadata"]["name"]
    namespace = manifest["metadata"]["namespace"]
    group, version = manifest["apiVersion"].split("/")
    plural = (manifest["kind"] + "s").lower()

    logging.info(f"Create custom resource object '{name}' of type '{manifest['kind']}' in namespace '{namespace}'.")

    api.create_namespaced_custom_object(group=group, version=version, namespace=namespace, plural=plural, body=manifest)

    poll_interval = 5  # [seconds]
    timeout_start = time()

    while True:
        if time() < timeout_start + timeout:
            status = api.get_namespaced_custom_object_status(group, version, namespace, plural, name=name)

            if status is not None and status.get("status") is not None:
                status = status["status"]
                if goal_state is not None:
                    if status.get(goal_state[0]) == goal_state[1]:
                        return True
                    else:
                        logging.info(
                            f"Waiting for custom resource state '{goal_state[0]}' to become '{goal_state[1]}'."
                        )
                elif goal_condition is not None:
                    if status.get("conditions") is not None and any(
                        [all([k in c and c[k] == v for k, v in goal_condition.items()]) for c in status["conditions"]]
                    ):
                        return True
                    else:
                        logging.info(f"Waiting for custom resource condition to become '{goal_condition}'.")
            else:
                logging.info("Custom resource status unavailable.")
            sleep(poll_interval)
        else:
            logging.warning(f"Custom resource didn't reach the goal state in {timeout}s. Deleting the custom resource.")
            delete_custom_resource_imperatively(manifest)
            return False
