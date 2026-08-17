| 用例            | e2e s | 峰值RSS MB | Prefill t/s | Gen t/s | token接受率 |
| --------------- | ----- | ---------- | ----------- | ------- | ----------- |
| baseline        | 189.1 | 3041       | 20.98       | 14.13   |             |
| layer-chunks=9  | 715.2 | 2735       | 8.08        | 0.77    | 0.514       |
| layer-chunks=4  | 795.4 | 2450       | 9.08        | 0.75    | 0.514       |
| layer-chunks=3 | 790.3 | 2354       | 7.20        | 0.79    | 0.514       |
| layer-chunks=2 | 736.9 | 2336       | 8.53        | 0.83    | 0.514       |
| layer-chunks=0        | 469.9 | 3191       | 10.35       | 1.38    | 0.514       |


--- conflayer pp (first, cool) ---
| qwen3 4B Q4_K - Medium         |   2.32 GiB |     4.02 B | CPU        |       8 |     256 |          pp1024 |         13.86 ± 0.00 |

--- baseline pp (second, warmer) ---
| qwen3 4B Q4_K - Medium         |   2.32 GiB |     4.02 B | CPU        |       8 |     256 |          pp1024 |         17.03 ± 0.00 |

--- conflayer context confirmation (from the first run's stderr) ---
    llama_context: n_seq_max     = 32
    llama_context: n_ubatch      = 256
    llama_context: kv_unified    = true
    llama_context: n_outputs_max = 32
    llama_context:        CPU  output buffer size =    18.55 MiB
    sched_reserve:        CPU compute buffer size =    42.01 MiB
    ~llama_context:        CPU compute buffer size is  42.0060 MiB, matches expectation of  42.0060 MiB
    llama_context: n_seq_max     = 32
    llama_context: n_ubatch      = 256
    llama_context: kv_unified    = true
    llama_context: n_outputs_max = 32
    llama_context:        CPU  output buffer size =    18.55 MiB
    sched_reserve:        CPU compute buffer size =    42.01 MiB
    ~llama_context:        CPU compute buffer size is  42.0060 MiB, matches expectation of  42.0060 MiB
