# Data

## Source

| | |
|---|---|
| Dataset | *Stellar Classification Dataset - SDSS17* (fedesoriano, Kaggle, January 2022) |
| Original survey | Sloan Digital Sky Survey, **Data Release 17 (DR17)** |
| Kaggle page | https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17 |
| Automatic-download mirror | `https://raw.githubusercontent.com/rhthm/sdss-object-classification/6f6e95a86cb3d58520716cadefb077d3ac50c107/data/star_classification.csv` (pinned to one commit) |
| File | `star_classification.csv`, 100,000 rows × 18 columns, 16.7 MB |
| SHA-256 | `9628c95bf0b132f5a965d9bd775e6d7e1304d31ace2ab0527ed0d0bdd9fb3858` |

Each row is one object with an SDSS optical spectrum. The `class` label (GALAXY, STAR or QSO) was assigned by
the SDSS spectroscopic pipeline, so these are measured labels, not hand-made or synthetic ones.

**Authenticity check.** SDSS identifiers encode where the data came from. `spec_obj_ID` packs the plate,
fibre and observing date (MJD), and `obj_ID` packs run, rerun, camera column and field. We decoded every
integer-formatted ID in the file (97,986 spectroscopic IDs and a sample of 20,000 photometric IDs). Every
decoded value matched the file's own `plate`, `fiber_ID`, `MJD`, `run_ID`, `rerun_ID`, `cam_col` and
`field_ID` columns, which is consistent with a genuine SDSS export. About 2% of `spec_obj_ID` values are
stored in floating-point notation in the CSV and could not be decoded exactly. They are not used by the model.

## Getting the file

Automatic (download plus checksum verification):

```
python -m src.data_loader
```

Manual (if the download is blocked on your network):

1. Download `star_classification.csv` from the Kaggle page above (a free account is required).
2. Save it as `data/raw/star_classification.csv`.
3. Run `python -m src.data_loader` to check the file loads.

You can also query SDSS directly at https://skyserver.sdss.org/dr17 (SQL Search, `SpecObj` joined to `PhotoObj`).
A file produced that way will have different column names and needs adapting before use.

## Folders

* `raw/`: the original CSV, unmodified. Not committed to Git.
* `processed/sdss_star_galaxy.csv`: the cleaned binary dataset written by `python -m src.train`. Not committed.

## Cleaning steps (`src/preprocessing.py`)

1. Keep only the columns `u, g, r, i, z, redshift, class`.
2. Remove QSO (quasar) rows, leaving the binary STAR/GALAXY task: 18,961 rows.
3. Remove rows with missing or non-numeric values: 0 rows.
4. Remove rows containing the SDSS sentinel `-9999` or non-finite values: 1 row (a star with `u = g = z = -9999`).
5. Remove exact duplicate rows: 0 rows.

Result: **81,038 rows** (59,445 galaxies, 21,593 stars). The full report is written to
`reports/preprocessing_report.json`.

## Citation

If you use this data, cite SDSS DR17:

> Abdurro'uf et al. (2022), "The Seventeenth Data Release of the Sloan Digital Sky Surveys: Complete Release of
> MaNGA, MaStar, and APOGEE-2 Data", *The Astrophysical Journal Supplement Series*, 259, 35.
> doi:10.3847/1538-4365/ac4414

Funding for SDSS has been provided by the Alfred P. Sloan Foundation, the U.S. Department of Energy Office of
Science, and the Participating Institutions (see https://www.sdss.org/collaboration/citing-sdss/).
