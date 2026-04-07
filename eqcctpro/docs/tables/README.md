# Paper Tables

Tables for the EQCCTPro methodology section. Generated from code and trial data.

## Regenerating Tables

```bash
# From project root
python scripts/visualization/generate_paper_tables.py --output_dir docs/tables --results_root results
```

## Table Descriptions

- **table1_memory_requirements.csv**: Per-model memory budgets (MB) from `eqcctpro/parallelization.py`. Base values from isolated-process testing; ModelActor includes buffers; Ripper uses empirically calibrated multipliers.
- **table2_optimal_picking_times.csv**: Optimal configuration picking times and actor creation overhead for 228 stations (Model-Actor method), extracted from `results/trials/*/optimal_configurations_*.csv`.
