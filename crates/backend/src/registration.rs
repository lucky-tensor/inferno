//! Service registration with load balancers

use inferno_shared::Result;
use serde_json::json;
use std::net::SocketAddr;

/// Service registration manager
pub struct ServiceRegistration {
    backend_addr: SocketAddr,
    load_balancer_addrs: Vec<SocketAddr>,
}

impl ServiceRegistration {
    pub fn new(backend_addr: SocketAddr, load_balancer_addrs: Vec<SocketAddr>) -> Self {
        Self {
            backend_addr,
            load_balancer_addrs,
        }
    }

    pub async fn register(&self) -> Result<()> {
        tracing::info!(
            "Registering backend {} with {} load balancers",
            self.backend_addr,
            self.load_balancer_addrs.len()
        );

        let client = reqwest::Client::new();
        let registration_payload = json!({
            "service_type": "backend",
            "address": self.backend_addr.to_string(),
            "health_check_path": "/health",
            "metadata": {
                "capabilities": ["inference", "health_check"],
                "version": env!("CARGO_PKG_VERSION")
            }
        });

        for lb_addr in &self.load_balancer_addrs {
            let registration_url = format!("http://{}/register", lb_addr);
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
                        tracing::info!("Successfully registered with load balancer: {}", lb_addr);
                    } else {
                        tracing::warn!(
                            "Registration failed with status {}: {}",
                            response.status(),
                            lb_addr
                        );
                    }
                }
                Err(e) => {
                    tracing::warn!("Failed to connect to load balancer {}: {}", lb_addr, e);
                    // Don't return error - continue trying other load balancers
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
