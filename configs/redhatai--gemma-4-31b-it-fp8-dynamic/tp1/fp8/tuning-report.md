# FP8 Dense GEMM Tuning Report

Generated: 2026-05-22T20:36:58-05:00

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | `RedHatAI/gemma-4-31B-it-FP8-Dynamic` |
| TP size | 1 |
| Container | vllm_node |
| Total time | 1s |

## Results

| Shape (N,K) | Status | Time |
|--------------------|--------|------|
| 5376,8192 | ✅ OK | 0s |
| 5376,21504 | ✅ OK | 0s |
| 16384,5376 | ✅ OK | 0s |
| 21504,5376 | ✅ OK | 0s |

## Summary

- **Succeeded:** 4/4
- **Failed:** 0/4
