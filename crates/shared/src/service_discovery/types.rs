//! Core data structures for service discovery
//!
//! This module contains all the fundamental data types used throughout the
//! service discovery system including node types and information structures.

use serde::{Deserialize, Serialize};
use std::time::SystemTime;

/// Serde module for SystemTime serialization
mod system_time_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    pub fn serialize<S>(time: &SystemTime, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let duration = time
            .duration_since(UNIX_EPOCH)
            .map_err(serde::ser::Error::custom)?;
        duration.as_secs().serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<SystemTime, D::Error>
    where
        D: Deserializer<'de>,
    {
        let secs = u64::deserialize(deserializer)?;
        Ok(UNIX_EPOCH + Duration::from_secs(secs))
    }
}

/// Node types in the Inferno distributed system
///
/// Each node type has specific responsibilities and capabilities:
/// - **Proxy**: Load balancer/reverse proxy handling client requests
/// - **Backend**: AI inference node processing requests  
/// - **Governator**: Cost optimization and resource management node
///
/// # Serialization
///
/// This enum is serialized as lowercase strings in JSON for consistency
/// with the protocol specification.
///
/// # Examples
///
/// ```rust
/// use inferno_shared::service_discovery::NodeType;
/// use serde_json;
///
/// let proxy_type = NodeType::Proxy;
/// let json = serde_json::to_string(&proxy_type).unwrap();
/// assert_eq!(json, "\"proxy\"");
///
/// let parsed: NodeType = serde_json::from_str("\"backend\"").unwrap();
/// assert_eq!(parsed, NodeType::Backend);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum NodeType {
    /// Load balancer/reverse proxy node
    ///
    /// Handles incoming client requests and distributes them across
    /// available backend nodes. Maintains health monitoring and
    /// implements load balancing strategies.
    ///
    /// **Capabilities**: Load balancing, health checking, request routing
    ///
    /// **Network Role**: Accepts external connections, forwards to backends
    Proxy,

    /// AI inference backend node
    ///
    /// Processes AI inference requests using GPU or CPU resources.
    /// Exposes metrics for health monitoring and capacity planning.
    ///
    /// **Capabilities**: AI inference, model serving, resource monitoring  
    ///
    /// **Network Role**: Accepts requests from proxies, returns inference results
    Backend,

    /// Cost optimization and resource management node
    ///
    /// Monitors cluster resource usage and costs, makes recommendations
    /// for scaling and optimization decisions.
    ///
    /// **Capabilities**: Cost analysis, resource optimization, scaling decisions
    ///
    /// **Network Role**: Monitors other nodes, provides optimization recommendations
    Governator,
}

impl NodeType {
    /// Returns whether this node type can serve client inference requests
    ///
    /// # Returns
    ///
    /// - `true` for Backend nodes (can process inference requests)
    /// - `false` for Proxy and Governator nodes (routing/management only)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use inferno_shared::service_discovery::NodeType;
    ///
    /// assert!(!NodeType::Proxy.can_serve_inference());
    /// assert!(NodeType::Backend.can_serve_inference());
    /// assert!(!NodeType::Governator.can_serve_inference());
    /// ```
    pub fn can_serve_inference(&self) -> bool {
        matches!(self, NodeType::Backend)
    }

    /// Returns whether this node type can act as a load balancer
    ///
    /// # Returns
    ///
    /// - `true` for Proxy nodes (primary load balancer role)
    /// - `false` for Backend and Governator nodes
    ///
    /// # Examples
    ///
    /// ```rust
    /// use inferno_shared::service_discovery::NodeType;
    ///
    /// assert!(NodeType::Proxy.can_load_balance());
    /// assert!(!NodeType::Backend.can_load_balance());
    /// assert!(!NodeType::Governator.can_load_balance());
    /// ```
    pub fn can_load_balance(&self) -> bool {
        matches!(self, NodeType::Proxy)
    }

    /// Returns default capabilities for this node type
    ///
    /// # Returns
    ///
    /// Returns a vector of capability strings that are typically
    /// supported by nodes of this type.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use inferno_shared::service_discovery::NodeType;
    ///
    /// let proxy_caps = NodeType::Proxy.default_capabilities();
    /// assert!(proxy_caps.contains(&"load_balancing".to_string()));
    /// assert!(proxy_caps.contains(&"health_checking".to_string()));
    ///
    /// let backend_caps = NodeType::Backend.default_capabilities();
    /// assert!(backend_caps.contains(&"inference".to_string()));
    /// ```
    pub fn default_capabilities(&self) -> Vec<String> {
        match self {
            NodeType::Proxy => vec![
                "load_balancing".to_string(),
                "health_checking".to_string(),
                "request_routing".to_string(),
                "service_discovery".to_string(),
            ],
            NodeType::Backend => vec![
                "inference".to_string(),
                "model_serving".to_string(),
                "metrics_reporting".to_string(),
            ],
            NodeType::Governator => vec![
                "cost_analysis".to_string(),
                "resource_optimization".to_string(),
                "scaling_decisions".to_string(),
                "cluster_monitoring".to_string(),
            ],
        }
    }

    /// Returns the string representation used in protocol messages
    ///
    /// # Returns
    ///
    /// Returns the lowercase string representation of the node type
    /// as used in JSON serialization and protocol messages.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use inferno_shared::service_discovery::NodeType;
    ///
    /// assert_eq!(NodeType::Proxy.as_str(), "proxy");
    /// assert_eq!(NodeType::Backend.as_str(), "backend");
    /// assert_eq!(NodeType::Governator.as_str(), "governator");
    /// ```
    pub fn as_str(&self) -> &'static str {
        match self {
            NodeType::Proxy => "proxy",
            NodeType::Backend => "backend",
            NodeType::Governator => "governator",
        }
    }

    /// Parses a node type from a string
    ///
    /// # Arguments
    ///
    /// * `s` - String representation of the node type (case-insensitive)
    ///
    /// # Returns
    ///
    /// Returns `Some(NodeType)` if the string is recognized, `None` otherwise.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use inferno_shared::service_discovery::NodeType;
    ///
    /// assert_eq!(NodeType::parse("proxy"), Some(NodeType::Proxy));
    /// assert_eq!(NodeType::parse("BACKEND"), Some(NodeType::Backend));
    /// assert_eq!(NodeType::parse("invalid"), None);
    /// ```
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "proxy" => Some(NodeType::Proxy),
            "backend" => Some(NodeType::Backend),
            "governator" => Some(NodeType::Governator),
            _ => None,
        }
    }
}

impl std::fmt::Display for NodeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl std::str::FromStr for NodeType {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        NodeType::parse(s).ok_or_else(|| format!("Invalid node type: {}", s))
    }
}

/// Enhanced node information for distributed service discovery
///
/// This structure contains comprehensive information about a node in the
/// Inferno distributed system, including its type, capabilities, and
/// last update timestamp for consensus operations.
///
/// # Protocol Specification
///
/// This struct represents the enhanced node information format used in
/// peer discovery and consensus operations. It includes all fields needed
/// for self-sovereign updates and distributed consensus.
///
/// # Examples
///
/// ```rust
/// use inferno_shared::service_discovery::{NodeInfo, NodeType};
/// use std::time::SystemTime;
///
/// let node = NodeInfo {
///     id: "proxy-1".to_string(),
///     address: "10.0.1.1:8080".to_string(),
///     metrics_port: 6100,
///     node_type: NodeType::Proxy,
///     is_load_balancer: true,
///     capabilities: vec!["load_balancing".to_string(), "health_checking".to_string()],
///     last_updated: SystemTime::now(),
/// };
///
/// assert_eq!(node.id, "proxy-1");
/// assert!(node.is_load_balancer);
/// assert_eq!(node.node_type, NodeType::Proxy);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeInfo {
    /// Unique node identifier
    ///
    /// Must be unique across all nodes in the distributed system.
    /// Typically includes node type and instance identifier.
    pub id: String,

    /// Network address where the node serves requests
    ///
    /// Format: "host:port" where host can be IP address or hostname
    /// and port is the service port (not metrics port).
    pub address: String,

    /// Port where the node exposes metrics and health endpoints
    ///
    /// Used for health monitoring and operational metrics.
    /// Must expose `/metrics` (JSON) and `/telemetry` (Prometheus) endpoints.
    pub metrics_port: u16,

    /// Type of node (Proxy, Backend, Governator)
    ///
    /// Determines the node's role and capabilities in the distributed system.
    pub node_type: NodeType,

    /// Whether this node can act as a load balancer
    ///
    /// True for nodes that can route requests to other nodes.
    /// Typically true for Proxy nodes, false for Backend/Governator nodes.
    pub is_load_balancer: bool,

    /// List of capabilities supported by this node
    ///
    /// Capabilities are string identifiers for features the node supports.
    /// Used for service discovery filtering and routing decisions.
    /// Examples: ["inference", "gpu", "cpu-only", "load_balancing"]
    pub capabilities: Vec<String>,

    /// Timestamp of last update to this node's information
    ///
    /// Used for consensus tie-breaking and detecting stale information.
    /// Should be updated whenever any node information changes.
    #[serde(with = "system_time_serde")]
    pub last_updated: SystemTime,
}

impl NodeInfo {
    /// Creates a new NodeInfo with default capabilities for the node type
    ///
    /// # Arguments
    ///
    /// * `id` - Unique node identifier
    /// * `address` - Network address for the node service
    /// * `metrics_port` - Port for metrics and health endpoints
    /// * `node_type` - Type of node (determines default capabilities)
    ///
    /// # Returns
    ///
    /// Returns a new NodeInfo with default capabilities and current timestamp.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use inferno_shared::service_discovery::{NodeInfo, NodeType};
    ///
    /// let backend = NodeInfo::new(
    ///     "backend-1".to_string(),
    ///     "10.0.1.5:3000".to_string(),
    ///     9090,
    ///     NodeType::Backend
    /// );
    ///
    /// assert_eq!(backend.id, "backend-1");
    /// assert!(!backend.is_load_balancer);
    /// assert!(backend.capabilities.contains(&"inference".to_string()));
    /// ```
    pub fn new(id: String, address: String, metrics_port: u16, node_type: NodeType) -> Self {
        Self {
            id,
            address,
            metrics_port,
            node_type,
            is_load_balancer: node_type.can_load_balance(),
            capabilities: node_type.default_capabilities(),
            last_updated: SystemTime::now(),
        }
    }

    /// Updates the last_updated timestamp to the current time
    ///
    /// This should be called whenever any node information changes
    /// to maintain accurate consensus timestamps.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use inferno_shared::service_discovery::{NodeInfo, NodeType};
    /// use std::time::SystemTime;
    ///
    /// let mut node = NodeInfo::new(
    ///     "backend-1".to_string(),
    ///     "10.0.1.5:3000".to_string(),
    ///     9090,
    ///     NodeType::Backend
    /// );
    ///
    /// let old_timestamp = node.last_updated;
    /// std::thread::sleep(std::time::Duration::from_millis(10));
    /// node.update_timestamp();
    /// assert!(node.last_updated > old_timestamp);
    /// ```
    pub fn update_timestamp(&mut self) {
        self.last_updated = SystemTime::now();
    }

    /// Returns the metrics endpoint URL for this node
    ///
    /// # Returns
    ///
    /// Returns a complete URL to the node's metrics endpoint.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use inferno_shared::service_discovery::{NodeInfo, NodeType};
    ///
    /// let node = NodeInfo::new(
    ///     "backend-1".to_string(),
    ///     "10.0.1.5:3000".to_string(),
    ///     9090,
    ///     NodeType::Backend
    /// );
    ///
    /// assert_eq!(node.metrics_url(), "http://10.0.1.5:9090/metrics");
    /// ```
    pub fn metrics_url(&self) -> String {
        let host = self.address.split(':').next().unwrap_or("localhost");
        format!("http://{}:{}/metrics", host, self.metrics_port)
    }

    /// Returns the telemetry endpoint URL for this node (Prometheus format)
    ///
    /// # Returns
    ///
    /// Returns a complete URL to the node's telemetry endpoint.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use inferno_shared::service_discovery::{NodeInfo, NodeType};
    ///
    /// let node = NodeInfo::new(
    ///     "backend-1".to_string(),
    ///     "10.0.1.5:3000".to_string(),
    ///     9090,
    ///     NodeType::Backend
    /// );
    ///
    /// assert_eq!(node.telemetry_url(), "http://10.0.1.5:9090/telemetry");
    /// ```
    pub fn telemetry_url(&self) -> String {
        let host = self.address.split(':').next().unwrap_or("localhost");
        format!("http://{}:{}/telemetry", host, self.metrics_port)
    }

    /// Checks if this node has a specific capability
    ///
    /// # Arguments
    ///
    /// * `capability` - The capability to check for
    ///
    /// # Returns
    ///
    /// Returns true if the node supports the specified capability.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use inferno_shared::service_discovery::{NodeInfo, NodeType};
    ///
    /// let backend = NodeInfo::new(
    ///     "backend-1".to_string(),
    ///     "10.0.1.5:3000".to_string(),
    ///     9090,
    ///     NodeType::Backend
    /// );
    ///
    /// assert!(backend.has_capability("inference"));
    /// assert!(!backend.has_capability("load_balancing"));
    /// ```
    pub fn has_capability(&self, capability: &str) -> bool {
        self.capabilities.iter().any(|c| c == capability)
    }
}

/// Peer node information for service discovery registration
///
/// This structure represents peer node information shared during
/// the registration process. It follows the exact specification format
/// from the service-discovery.md protocol documentation.
///
/// # Protocol Specification
///
/// This struct matches the peer information format returned in
/// registration responses and used for consensus operations.
/// It contains a subset of NodeInfo fields specifically needed
/// for peer-to-peer communication.
///
/// # Examples
///
/// ```rust
/// use inferno_shared::service_discovery::{PeerInfo, NodeType};
/// use std::time::SystemTime;
///
/// let peer = PeerInfo {
///     id: "proxy-1".to_string(),
///     address: "10.0.1.1:8080".to_string(),
///     metrics_port: 6100,
///     node_type: NodeType::Proxy,
///     is_load_balancer: true,
///     last_updated: SystemTime::now(),
/// };
///
/// assert_eq!(peer.id, "proxy-1");
/// assert!(peer.is_load_balancer);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PeerInfo {
    /// Unique peer identifier
    /// Must be unique across all peers in the distributed system
    pub id: String,

    /// Network address where the peer serves requests
    /// Format: "host:port" for the service endpoint
    pub address: String,

    /// Port where the peer exposes metrics and health endpoints
    /// Used for health monitoring and operational metrics
    pub metrics_port: u16,

    /// Type of peer node (Proxy, Backend, Governator)
    /// Determines the peer's role in the distributed system
    pub node_type: NodeType,

    /// Whether this peer can act as a load balancer
    /// True for nodes that can route requests to other nodes
    pub is_load_balancer: bool,

    /// Timestamp of last update to this peer's information
    /// Used for consensus tie-breaking and stale data detection
    #[serde(with = "system_time_serde")]
    pub last_updated: SystemTime,
}

impl PeerInfo {
    /// Creates a new PeerInfo from NodeInfo
    ///
    /// # Arguments
    ///
    /// * `node_info` - The NodeInfo to convert to PeerInfo
    ///
    /// # Returns
    ///
    /// Returns a new PeerInfo containing the essential fields from NodeInfo
    /// needed for peer-to-peer communication.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use inferno_shared::service_discovery::{NodeInfo, PeerInfo, NodeType};
    ///
    /// let node = NodeInfo::new(
    ///     "backend-1".to_string(),
    ///     "10.0.1.5:3000".to_string(),
    ///     9090,
    ///     NodeType::Backend
    /// );
    ///
    /// let peer = PeerInfo::from_node_info(&node);
    /// assert_eq!(peer.id, node.id);
    /// assert_eq!(peer.address, node.address);
    /// ```
    pub fn from_node_info(node_info: &NodeInfo) -> Self {
        Self {
            id: node_info.id.clone(),
            address: node_info.address.clone(),
            metrics_port: node_info.metrics_port,
            node_type: node_info.node_type,
            is_load_balancer: node_info.is_load_balancer,
            last_updated: node_info.last_updated,
        }
    }

    /// Converts this PeerInfo to a NodeInfo with default capabilities
    ///
    /// # Returns
    ///
    /// Returns a NodeInfo struct with capabilities set to defaults
    /// for the peer's node type.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use inferno_shared::service_discovery::{PeerInfo, NodeType};
    /// use std::time::SystemTime;
    ///
    /// let peer = PeerInfo {
    ///     id: "backend-1".to_string(),
    ///     address: "10.0.1.5:3000".to_string(),
    ///     metrics_port: 9090,
    ///     node_type: NodeType::Backend,
    ///     is_load_balancer: false,
    ///     last_updated: SystemTime::now(),
    /// };
    ///
    /// let node = peer.to_node_info();
    /// assert!(node.has_capability("inference"));
    /// ```
    pub fn to_node_info(&self) -> NodeInfo {
        NodeInfo {
            id: self.id.clone(),
            address: self.address.clone(),
            metrics_port: self.metrics_port,
            node_type: self.node_type,
            is_load_balancer: self.is_load_balancer,
            capabilities: self.node_type.default_capabilities(),
            last_updated: self.last_updated,
        }
    }
}

/// Backend registration information (backward compatibility)
///
/// This structure represents backend registration information for
/// backward compatibility with the legacy service discovery API.
///
/// # Migration Note
///
/// This type is provided for backward compatibility. New code should
/// use `NodeInfo` directly for enhanced functionality.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BackendRegistration {
    /// Unique backend identifier
    pub id: String,

    /// Network address where the backend serves requests
    pub address: String,

    /// Port where the backend exposes metrics
    pub metrics_port: u16,
}

impl BackendRegistration {
    /// Converts this registration to a NodeInfo with Backend type
    ///
    /// # Returns
    ///
    /// Returns a NodeInfo with NodeType::Backend and default capabilities.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use inferno_shared::service_discovery::{BackendRegistration, NodeType};
    ///
    /// let registration = BackendRegistration {
    ///     id: "backend-1".to_string(),
    ///     address: "10.0.1.5:3000".to_string(),
    ///     metrics_port: 9090,
    /// };
    ///
    /// let node_info = registration.to_node_info();
    /// assert_eq!(node_info.node_type, NodeType::Backend);
    /// ```
    pub fn to_node_info(&self) -> NodeInfo {
        NodeInfo::new(
            self.id.clone(),
            self.address.clone(),
            self.metrics_port,
            NodeType::Backend,
        )
    }

    /// Creates a BackendRegistration from NodeInfo
    ///
    /// # Arguments
    ///
    /// * `node_info` - The NodeInfo to convert
    ///
    /// # Returns
    ///
    /// Returns a BackendRegistration with the essential fields from NodeInfo.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use inferno_shared::service_discovery::{BackendRegistration, NodeInfo, NodeType};
    ///
    /// let node = NodeInfo::new(
    ///     "backend-1".to_string(),
    ///     "10.0.1.5:3000".to_string(),
    ///     9090,
    ///     NodeType::Backend
    /// );
    ///
    /// let registration = BackendRegistration::from_node_info(&node);
    /// assert_eq!(registration.id, node.id);
    /// ```
    pub fn from_node_info(node_info: &NodeInfo) -> Self {
        Self {
            id: node_info.id.clone(),
            address: node_info.address.clone(),
            metrics_port: node_info.metrics_port,
        }
    }
}
