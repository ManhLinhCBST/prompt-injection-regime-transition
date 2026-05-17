# Prompt Injection as Regime Transition

Cross-model evidence that successful prompt injection events induce persistent post-incident state drift in multi-cycle LLM runtimes.

## Key findings

- Successful prompt injection behaves like a regime transition, not just a point failure.
- Post-incident runtime entropy remains elevated for ~7-9 cycles.
- Post-leak reasoning becomes shorter, more defensive, and more exclusion-focused.
- A narrow L3 phrasing replicated across Qwen3:30B, DeepSeek-R1 14B, and Mistral 7B.
- Qwen3:8B resisted the trigger while Qwen3:30B leaked, suggesting an inverse-scaling pattern on this failure mode.

## Full writeup

See: [ml_cbst_v5_state_drift.md](ml_cbst_v5_state_drift.md)
## Example Runtime Trace

### Leak Transition

![Leak Transition](images/leak_transition_example.png)

### Entropy Recovery

![Entropy Recovery](images/entropy_recovery_trace.png)
## Status

Exploratory independent research note. Selected artifacts may be shared for verification.
