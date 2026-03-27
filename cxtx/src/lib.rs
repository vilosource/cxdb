pub mod cli;
pub mod cxdb_http;
pub mod delivery;
pub mod ledger;
pub mod provider;
pub mod proxy;
pub mod session;
pub mod turns;

use anyhow::{Context, Result};
use cli::Cli;
use delivery::DeliveryHandle;
use ledger::SessionLedgerWriter;
use provider::{resolve_upstream_base_for_env, ProviderKind};
use proxy::{ProxyServer, UpstreamConfig};
use session::SessionRuntime;
use std::process::Stdio;
use tokio::process::Command;

pub async fn run(cli: Cli) -> Result<i32> {
    let provider = cli.command.provider();
    let effective_url = cli.effective_url();
    let cxdb_url = cli
        .effective_url()
        .parse()
        .with_context(|| format!("invalid CXDB URL: {effective_url}"))?;

    let upstream_config = match provider {
        ProviderKind::Pi => {
            let openai_upstream = resolve_upstream_base_for_env(
                &["OPENAI_BASE_URL", "OPENAI_API_BASE"],
                "https://api.openai.com/v1",
            )
            .context("failed to resolve OpenAI upstream for Pi")?;
            let anthropic_upstream = resolve_upstream_base_for_env(
                &[
                    "ANTHROPIC_BASE_URL",
                    "ANTHROPIC_API_URL",
                    "ANTHROPIC_API_BASE",
                ],
                "https://api.anthropic.com",
            )
            .context("failed to resolve Anthropic upstream for Pi")?;
            UpstreamConfig::DualProtocol {
                openai_upstream,
                anthropic_upstream,
            }
        }
        _ => {
            let upstream_base = provider
                .resolve_upstream_base()
                .context("failed to resolve provider upstream base URL")?;
            UpstreamConfig::Single { upstream_base }
        }
    };

    let (listener, proxy_base_url) = ProxyServer::bind(provider, &upstream_config)
        .await
        .context("failed to reserve local reverse proxy listener")?;
    let args = provider.child_args_for_proxy(cli.command.args(), Some(&proxy_base_url));
    let allowlisted_env = provider.capture_env_allowlist();
    let session = SessionRuntime::new(provider, args.clone(), allowlisted_env)?;
    let ledger = SessionLedgerWriter::create(&session).await?;

    let proxy = ProxyServer::start_with_listener(
        provider,
        upstream_config,
        session.clone(),
        ledger.clone(),
        listener,
        proxy_base_url.clone(),
    )
    .await
    .context("failed to start local reverse proxy")?;
    let delivery = DeliveryHandle::start(
        cxdb_url,
        session.clone(),
        ledger.clone(),
        provider.client_tag().to_string(),
    )
    .await?;
    proxy.set_delivery(delivery.clone()).await;

    let mut command = Command::new(provider.command_name());
    command.args(&args);
    command.stdin(Stdio::inherit());
    command.stdout(Stdio::inherit());
    command.stderr(Stdio::inherit());
    for name in provider.upstream_base_env_names() {
        command.env_remove(name);
    }
    command.envs(provider.injected_env(&proxy_base_url));

    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(err) => {
            ledger
                .note_delivery_state("child_launch_failed", 0, Some(err.to_string()))
                .await
                .ok();
            delivery.shutdown().await.ok();
            proxy.shutdown().await.ok();
            ledger.finalize().await.ok();
            return Err(err).with_context(|| {
                format!(
                    "failed to launch {} using PATH resolution",
                    provider.command_name()
                )
            });
        }
    };
    session.set_child_pid(child.id());
    ledger.note_child_pid(child.id()).await?;

    delivery.enqueue_create_context().await?;
    delivery.enqueue_turn(session.session_start_turn()).await?;

    let status = child.wait().await?;
    let exit_code = status.code().unwrap_or(1);

    delivery
        .enqueue_turn(session.session_end_turn(exit_code, status.success()))
        .await?;
    ledger.note_child_exit(exit_code).await?;

    proxy.shutdown().await?;
    delivery.shutdown().await?;
    ledger.finalize().await?;

    Ok(exit_code)
}

pub async fn run_provider_command(
    provider: ProviderKind,
    args: Vec<String>,
    url: &str,
) -> Result<i32> {
    run(Cli::for_tests(provider, args, url)).await
}
