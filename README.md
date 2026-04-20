# EE 202 — Electronics
**Northwestern University · McCormick School of Engineering**

Electronics and signal processing course. Project: frequency-domain audio compression algorithm implemented in MATLAB, analyzing the trade-off between compression ratio and signal reconstruction quality.

---

## Project — Audio Compression Algorithm

### Overview
Developed a signal compression algorithm for audio signals using frequency-domain analysis in MATLAB. The algorithm applies a low-pass filter at a tunable cutoff frequency to remove high-frequency content, then evaluates reconstruction quality via an average confidence metric.

### Methodology
1. Load audio signal and compute FFT
2. Apply low-pass filter at cutoff frequency `fc`
3. Reconstruct compressed signal via inverse FFT
4. Compute average confidence score between original and reconstructed signal
5. Sweep `fc` from 3 kHz to 10 kHz to characterize the accuracy vs. compression trade-off

### Results
- Optimal cutoff frequency: ~3 kHz (average confidence ≈ 0.93)
- Higher cutoff frequencies increase fidelity but reduce compression ratio
- Trade-off curve shows diminishing returns above 5 kHz

### Key Plot
`Average Confidence vs. Cutoff Frequency` — shows the compression quality curve used to select the operating point.

---

## Stack
`MATLAB` `Signal Processing Toolbox` `FFT` `DSP`

---

*Northwestern University — EE 202 · Electronics*
