# Metrics Test

This test starts some instances on the worker (must first be running, for example with "just d") and sends calls.
Check the constants for easily tunable parameters.

## Usage Instructions

It's helpful to test the metric instrumentation of the worker in the following way:

### FROM ROOT
1. Start worker: `just d`
2. Start metrics client: `just metrics-client`
3. Run this test: `just metrics-test`
4. Run analysis: `just metrics-analyse`

Then check the Python script in `/benchmarks/analyse.py`  
*(For now very rudimentary, we must expand it to produce correct measurements)*  
→ verify performance

## TODOS

- Fix bug in analyse.py or inside worker? Sometimes the cold start latency (measured as timestamp running - timestamp started) comes back negative
- Add internal latency measurement
  - May need a request-id? How to map requests with the events generated in the worker atm.
  - We could put it in the context MAYBE
- CPU/MEM/DISK data
