# PI0 Evaluation Findings

- The intervention framework now exposes alarm counts, rejection reasons, and accepted-risk deltas.
- Hybrid candidate generation was the only source that consistently beat the matched zero-success baseline.
- The best online ceiling in the reported sweeps stayed at `1/10`, so candidate usefulness remains the main blocker rather than alarm triggering alone.
- Follow-up tuning around the best hybrid settings did not move the ceiling higher, which points toward needing better recovery candidates and/or richer retraining data.
