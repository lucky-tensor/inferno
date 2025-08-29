//! Service registration with load balancers

use inferno_shared::Result;
use serde_json::json;
use std::net::SocketAddr;

/// Service registration manager
pub struct ServiceRegistration {
    backend_addr: SocketAddr,
    load_balancer_addrs: Vec<SocketAddr>,
    metrics_port: u16,
    backend_id: String,
}

impl ServiceRegistration {
    pub fn new(
        backend_addr: SocketAddr,
        load_balancer_addrs: Vec<SocketAddr>,
        metrics_port: u16,
        backend_id: Option<String>,
    ) -> Self {
        let backend_id = backend_id.unwrap_or_else(|| format!("backend-{}", backend_addr.port()));
        Self {
            backend_addr,
            load_balancer_addrs,
            metrics_port,
            backend_id,
        }
    }

    pub async fn register(&self) -> Result<()> {
        tracing::info!(
            "Registering backend {} with {} proxy operations servers",
            self.backend_addr,
            self.load_balancer_addrs.len()
        );

        let client = reqwest::Client::new();
        let registration_payload = json!({
            "id": self.backend_id,
            "address": self.backend_addr.to_string(),
            "metrics_port": self.metrics_port
        });

        for lb_addr in &self.load_balancer_addrs {
            // Convert proxy address to operations server address (port 6100)
            let operations_addr = SocketAddr::new(lb_addr.ip(), 6100);
            let registration_url = format!("http://{}/registration", operations_addr);
            tracing::debug!("Attempting registration at: {}", registration_url);

            match client
                .post(&registration_url)
                .json(&registration_payload)
                .timeout(std::time::Duration::from_secs(5))
                .send()
                .await
            {
                Ok(response) => {
                    if response.status().is_success() {
                        tracing::info!(
                            "Successfully registered with proxy operations server: {}",
                            operations_addr
                        );
                    } else {
                        tracing::warn!(
                            "Registration failed with status {}: {}",
                            response.status(),
                            operations_addr
                        );
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        "Failed to connect to proxy operations server {}: {}",
                        operations_addr,
                        e
                    );
                    // Don't return error - continue trying other proxy operations servers
                }
            }
        }

        Ok(())
    }

    pub async fn deregister(&self) -> Result<()> {
        // Placeholder implementation - would deregister self.backend_addr from load balancers
        // For now, just reference the fields to avoid dead code warnings
        tracing::info!(
            "Deregistering backend {} from {} load balancers",
            self.backend_addr,
            self.load_balancer_addrs.len()
        );
        Ok(())
    }
}
