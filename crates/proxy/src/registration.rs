//! Backend registration HTTP service
//!
//! This module provides an HTTP service for handling backend registration requests.
//! Backends POST their registration information to `/register` and this service
//! manages them in the service discovery system.

use http::{Method, StatusCode};
use http_body_util::BodyExt;
use hyper::service::{make_service_fn, service_fn};
use hyper::{Body, Request, Response, Server};
use inferno_shared::service_discovery::{BackendRegistration, ServiceDiscovery};
use serde_json;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;
use tracing::{debug, info, warn};

/// Registration service that handles backend registration requests
pub struct RegistrationService {
    /// Service discovery instance to manage backends
    service_discovery: Arc<ServiceDiscovery>,
}

impl RegistrationService {
    /// Creates a new registration service
    pub fn new() -> Self {
        let service_discovery = Arc::new(ServiceDiscovery::new());
        Self {
            service_discovery,
        }
    }

    /// Starts the registration HTTP service
    pub async fn start(&self, listen_addr: SocketAddr) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let service_discovery = Arc::clone(&self.service_discovery);
        
        // Start health checking in the background
        let _health_check_handle = service_discovery.start_health_checking().await;

        info!("Starting registration service on {}", listen_addr);

        let make_svc = make_service_fn(move |_conn| {
            let service_discovery = Arc::clone(&service_discovery);
            async move {
                Ok::<_, Infallible>(service_fn(move |req| {
                    Self::handle_request(Arc::clone(&service_discovery), req)
                }))
            }
        });

        let server = Server::bind(&listen_addr).serve(make_svc);

        if let Err(e) = server.await {
            warn!("Registration server error: {}", e);
        }

        Ok(())
    }

    /// Handles incoming HTTP requests
    async fn handle_request(
        service_discovery: Arc<ServiceDiscovery>,
        req: Request<Body>,
    ) -> Result<Response<Body>, Infallible> {
        match (req.method(), req.uri().path()) {
            (&Method::POST, "/register") => {
                Self::handle_register(service_discovery, req).await
            }
            (&Method::GET, "/health") => {
                Self::handle_health_check().await
            }
            _ => {
                let response = Response::builder()
                    .status(StatusCode::NOT_FOUND)
                    .body(Body::from("Not Found"))
                    .unwrap();
                Ok(response)
            }
        }
    }

    /// Handles backend registration requests
    async fn handle_register(
        service_discovery: Arc<ServiceDiscovery>,
        req: Request<Body>,
    ) -> Result<Response<Body>, Infallible> {
        debug!("Received registration request");

        // Read the request body
        let body_bytes = match BodyExt::collect(req.into_body()).await {
            Ok(collected) => collected.to_bytes(),
            Err(e) => {
                warn!("Failed to read request body: {}", e);
                let response = Response::builder()
                    .status(StatusCode::BAD_REQUEST)
                    .body(Body::from("Failed to read request body"))
                    .unwrap();
                return Ok(response);
            }
        };

        // Parse the JSON registration data
        let registration: BackendRegistration = match serde_json::from_slice(&body_bytes) {
            Ok(reg) => reg,
            Err(e) => {
                warn!("Failed to parse registration JSON: {}", e);
                let response = Response::builder()
                    .status(StatusCode::BAD_REQUEST)
                    .body(Body::from("Invalid JSON format"))
                    .unwrap();
                return Ok(response);
            }
        };

        debug!(
            backend_id = %registration.id,
            address = %registration.address,
            metrics_port = registration.metrics_port,
            "Attempting to register backend"
        );

        // Register the backend
        match service_discovery.register_backend(registration.clone()).await {
            Ok(()) => {
                info!(
                    backend_id = %registration.id,
                    address = %registration.address,
                    "Backend registered successfully"
                );
                let response = Response::builder()
                    .status(StatusCode::OK)
                    .header("Content-Type", "application/json")
                    .body(Body::from(r#"{"status":"success","message":"Backend registered"}"#))
                    .unwrap();
                Ok(response)
            }
            Err(e) => {
                warn!(
                    backend_id = %registration.id,
                    error = %e,
                    "Failed to register backend"
                );
                let error_response = serde_json::json!({
                    "status": "error",
                    "message": e.to_string()
                });
                let response = Response::builder()
                    .status(StatusCode::BAD_REQUEST)
                    .header("Content-Type", "application/json")
                    .body(Body::from(error_response.to_string()))
                    .unwrap();
                Ok(response)
            }
        }
    }

    /// Handles health check requests for the registration service itself
    async fn handle_health_check() -> Result<Response<Body>, Infallible> {
        let health_response = serde_json::json!({
            "status": "healthy",
            "service": "registration"
        });

        let response = Response::builder()
            .status(StatusCode::OK)
            .header("Content-Type", "application/json")
            .body(Body::from(health_response.to_string()))
            .unwrap();
        
        Ok(response)
    }

    /// Returns the service discovery instance for accessing registered backends
    pub fn service_discovery(&self) -> &ServiceDiscovery {
        &self.service_discovery
    }
}