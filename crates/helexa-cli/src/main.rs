// SPDX-License-Identifier: PolyForm-Shield-1.0

use std::net::SocketAddr;

use anyhow::Result;
use clap::{Parser, Subcommand};
use tracing::info;

#[derive(Parser)]
#[command(name = "helexa", version, about = "helexa cortex/neuron node")]
struct Cli {
    /// optional path to a config file (applies to all subcommands)
    #[arg(long)]
    config: Option<String>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// run a cortex node (mesh + optional orchestrator/gateway/portal roles)
    Cortex(CortexOpts),

    /// run a neuron node (model runtime worker)
    Neuron(NeuronOpts),
}

#[derive(Parser, Debug)]
struct CortexOpts {
    /// address for orchestrator control api (enables orchestrator role)
    #[arg(long)]
    orchestrator_socket: Option<SocketAddr>,

    /// address for public api gateway (enables gateway role)
    #[arg(long)]
    gateway_socket: Option<SocketAddr>,

    /// address(es) for portal frontends (enables portal role, repeatable)
    #[arg(long)]
    portal_socket: Vec<SocketAddr>,

    /// optional node identity / label for operator
    #[arg(long)]
    node_id: Option<String>,

    /// address for cortex control-plane websocket listener (neurons connect here)
    #[arg(long)]
    control_plane_socket: Option<SocketAddr>,
}

#[derive(Parser, Debug)]
struct NeuronOpts {
    /// address for neuron control channel (e.g. grpc or quic)
    #[arg(long, default_value = "0.0.0.0:9050")]
    control_socket: SocketAddr,

    /// address for local model-serving api (if any)
    #[arg(long, default_value = "127.0.0.1:8060")]
    api_socket: SocketAddr,

    /// directory for model storage / cache
    #[arg(long)]
    models_dir: Option<String>,

    /// optional node identity / label for operator
    #[arg(long)]
    node_id: Option<String>,

    /// URL of the cortex control-plane websocket endpoint this neuron should connect to
    #[arg(long)]
    cortex_control_endpoint: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    util::logging::init_tracing();

    let cli = Cli::parse();
    info!("starting helexa: {:?}", cli.command);

    match cli.command {
        Commands::Cortex(opts) => {
            let config = cortex::Config {
                orchestrator_socket: opts.orchestrator_socket,
                gateway_socket: opts.gateway_socket,
                portal_sockets: opts.portal_socket,
                node_id: opts.node_id,
                control_plane_socket: opts.control_plane_socket,
            };
            cortex::run(config).await?;
        }
        Commands::Neuron(opts) => {
            let config = neuron::Config {
                control_socket: opts.control_socket,
                api_socket: opts.api_socket,
                models_dir: opts.models_dir,
                node_id: opts.node_id,
                cortex_control_endpoint: opts.cortex_control_endpoint,
            };
            neuron::run(config).await?;
        }
    }

    Ok(())
}
