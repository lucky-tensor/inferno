# Pingora Reverse Proxy

A reverse proxy built with the pingora-proxy library that decrypts payloads using the Fernet protocol and routes requests to configured backend servers.

## Overview

This application creates a reverse proxy that:

1. Receives encrypted requests containing a `MinerRequest` payload
2. Decrypts the payload using the Fernet protocol
3. Routes the decrypted request to the appropriate backend server based on the `model_id`

## Payload Structure

The encrypted payload contains a data structure with the following format:

```rust
struct MinerRequest {
    model_id: String,
    prompt: String,
}
```

## Configuration

On startup, the application reads a `config.yaml` file that defines:

- A list of API endpoints (backend servers)
- The `model_id` that each endpoint can serve

The proxy uses this configuration to determine which backend server should handle each decrypted request based on the `model_id` in the payload.

## Contributing

1. Write tests first (TDD approach)
2. Ensure all benchmarks pass without regression
3. Follow Rust security best practices
4. Update documentation for API changes
