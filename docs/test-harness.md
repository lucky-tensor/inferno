
We run:
1. "unit tests" within module definition (doc tests)
2. we run "module tests" with exported public functions in ./tests/<module-name>_test.rs
3. we run "integration tests": that client-server connections work for the different endpoints.
4. we run "e2e tests" which run the actual binary and interfaces



# e2e tests
The e2e tests should check:
1. clients calling the load balancer
2. load balancer waiting for service discovery announcement from a backend
3. a backend announcing shutdown and gracefully disconnecting from load balancer
4. a client sending a request to load balancer which gets filled by a backend
5. a client sending a request and receiving a streaming request back from load balancer and backend.
6. multiple backends announce themselves to load balancer
7. multiple clients request to load balancer, and load balancer distributes to multiple backends. We check the metrics port of the backends to see if their filled requests counter incremented.
