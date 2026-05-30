# Example Datasets

This directory contains lightweight example data extracted from the project datasets.
Each example keeps 200 sampled rows and is intended for GitHub demonstration, code walkthroughs,
and quick sanity checks instead of benchmark reproduction.

## Notes

- Sample size: `200` rows per dataset.
- Files preserve the original raw CSV layout as much as possible so that the existing loaders remain familiar.
- `dapt2020` is stored as a directory containing one CSV because the project loader reads a directory of CSV files.
- These samples are for examples only and should not be used to report paper metrics.

## Included Files

| Dataset | Example Path | Rows | Label Column | Source Layout |
| --- | --- | ---: | --- | --- |
| UNSW-NB15-2024 | `example_data/UNSW-NB15-2024_sample_200.csv` | 200 | `Activity` | csv |
| dapt2020 | `example_data/dapt2020_sample_200.csv` | 200 | `Activity` | Activity |
| earlycrow | `example_data/earlycrow_sample_200.csv` | 200 | `multiple_label` | csv |
| zapt | `example_data/zapt_sample_200.csv` | 200 | `label_sub` | csv |
| BAI-Net26 | `example_data/BAI-Net26_sample_200.csv` | 200 | `Activity` | csv |
| Unraveled | `example_data/Unraveled_sample_200.csv` | 200 | `Activity` | csv |

## Loader Hints

- `UNSW-NB15-2024`: pass `example_data/UNSW-NB15-2024_sample_200.csv` to `load_UNSW-NB15-2024_dataset()`.
- `BAI-Net26`: pass `example_data/BAI-Net26_sample_200.csv` to `load_BAI-Net26_dataset()`.
- `Unraveled`: pass `example_data/Unraveled_sample_200.csv` to `load_Unraveled_dataset()`.
- `earlycrow`: pass `example_data/earlycrow_sample_200.csv` to `load_earlycrow_dataset()`.
- `zapt`: pass `example_data/zapt_sample_200.csv` to `load_zapt_dataset()`.
- `dapt2020`: pass `example_data/dapt2020_sample_200.csv` to `load_dapt2020_dataset()`.


