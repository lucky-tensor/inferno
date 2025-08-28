# service discovery protocol

The nodes can run in two modes: load balancer, or backend. Backend runs jobs. Load balancer splits jobs.
Nodes must discover each other.
As such at startup the config file will inform the node of the addresses of load balancers for the purpose of discovery.
Once started up each node announces themselves to the load balancers.
The load balancers, if online, respond with the list of nodes that are alive and their statuses. This list may include nodes previously unknown to the announcing node.
Once a connection is established the node can periodically call other nodes and request the /metrics from the nodes.
Each node keeps a list of their peers. The peers are ranked by network latency between calls.
When a node shuts down, whether load balancer or backend, they announce their shutdown so that load balancers stop sending them work. And will gracefully shut down once they have no new inbound connections.

# todo:
- [ ] what canonical Rust libraries can be used for service discovery (prefer not to reinvent something)
