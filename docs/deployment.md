# Deployment Guide

This document describes how to deploy a complete Helexa environment (Cortex and Neurons) using the `deploy.sh` script.

## Prerequisites

### Local Environment (The machine running the deployment)

The following tools must be installed on the machine from which you are running the deployment script:

*   **Rust Toolchain**: `cargo` is required to build the release binary.
*   **Utilities**: `jq`, `yq` (YAML processor), `envsubst` (usually part of `gettext`), `rsync`, `file`, and `strip` (binutils).
*   **SSH**: Key-based authentication to the target machines.

### Target Environment (Cortex and Neuron nodes)

*   **Operating System**: Linux-based OS with systemd.
*   **User Access**: An administrative user with `sudo` privileges. Passwordless sudo is recommended for smoother automation.
*   **SSH Access**: The public key of the deployment machine must be in the `~/.ssh/authorized_keys` of the target user.
*   **Firewall** (Optional): If `firewalld` (firewall-cmd) or `ufw` is active, the script will attempt to configure exceptions automatically.

## Configuration

Deployments are defined in YAML configuration files located in `asset/env/`.

The script expects the configuration file to follow a specific naming convention based on the orchestration name passed as an argument:

`asset/env/.<orchestration_name>.yml`

### Example Configuration

You can find a documented example in [`asset/env/example.yml`](../asset/env/example.yml).

Key sections:
*   **orchestrations**: A list of environments.
*   **cortex**: Configuration for the control plane node.
    *   Includes networking details (IP, ports), SSH credentials, and service user configuration.
*   **neurons**: A list of compute nodes.
    *   Similar configuration structure to cortex nodes.

To create a new deployment configuration for an orchestration named "production":

1.  Copy the example:
    ```bash
    cp asset/env/example.yml asset/env/.production.yml
    ```
2.  Edit `.production.yml` to match your infrastructure. Ensure the `name` field under `orchestrations` matches the orchestration name you intend to use.

## Running the Deployment

To deploy the "production" orchestration:

```bash
./script/deploy.sh production
```

### Process Overview

1.  **Build**: Compiles the Helexa binary in release mode using `cargo` and strips symbols to reduce size.
2.  **Staging**: Prepares temporary files for configuration locally.
3.  **Cortex Deployment**:
    *   Connects to the Cortex host (local or remote).
    *   Installs/Updates the `helexa` binary.
    *   Deploys the default spec file (`default.json`).
    *   Deploys and configures the systemd unit (`helexa-cortex.service`).
    *   Configures firewall rules (if `firewalld` or `ufw` is detected) for API, WebSocket, and Dashboard ports.
    *   Enables and restarts the service.
4.  **Neuron Deployment**:
    *   Iterates through defined neurons.
    *   Connects to each Neuron host.
    *   Installs/Updates the `helexa` binary.
    *   Deploys and configures the systemd unit (`helexa-neuron.service`) with environment variables pointing back to the Cortex.
    *   Configures firewall rules (if detected) for the API port.
    *   Enables and restarts the service.

## Directory Structure on Target

*   **Binaries**: `/usr/local/bin/helexa`
*   **Systemd Units**: `/etc/systemd/system/helexa-cortex.service` or `/etc/systemd/system/helexa-neuron.service`
*   **Specs**: `/usr/local/share/helexa/`
*   **Service Home**: `/var/lib/helexa` (Default home directory for the service user)

## Troubleshooting

*   **SSH Errors**: Ensure you can manually SSH into the target machines using the usernames and ports specified in the YAML file without entering a password.
*   **Permission Errors**: Ensure the target user has sudo rights.
*   **Missing Tools**: Check the "Local Environment" prerequisites if the script fails early.