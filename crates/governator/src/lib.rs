//! # Inferno Governator
//!
//! Cost optimization and resource governance component for Inferno distributed systems.
//! Handles real-time pricing data, performance vs cost optimization, and resource decisions.
//!
//! ## Features
//!
//! - Real-time pricing data integration
//! - Performance vs cost optimization algorithms  
//! - Binary start/stop decisions for compute nodes
//! - Multi-cloud cost comparison
//! - PostgreSQL with time-series optimization

pub mod config;
pub mod cost_analysis;
pub mod decision_engine;
pub mod storage;

pub use config::GovernatorConfig;
