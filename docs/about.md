
Inferno is a self-healing cloud platform for AI inference, designed for high-performance, reliability, and observability. It demonstrates best practices for distributed systems, with comprehensive testing and robust error recovery for AI workloads.

# History
We were working on a self-healing cloud for AI inference that could run on bare metal, optimized for batches and kernel fusion, and then Cloudflare published their architecture with a similar name and architecture. We love it. Nothing is a coincidence. https://blog.cloudflare.com/cloudflares-most-efficient-ai-inference-engine/

# Motivation
The state of the art (Summer 2025) for IT friendly turnkey serving of LLM are VLLM platforms. However to squeeze the most performance per GPU hyperscalers use proprietary kernel and memory optimization (TensorRT for nvidia). [Read more in](./state-of-the-art-2025.md).

# Trends we consider
There are some important trends when committing to an LLM stack:
- Models are getting bigger (they can't fit on a single node or GPU)
- Models are getting more specialized (you may want to use multiple models)
- More chip architectures will emerge (AMD needs to be first class)
- More data centers will come online (not only the hyperscalers)
- Users will demand omni - video, audio, text (higher throughput required)
- Other languages are catching up to Python in maturity and features.
