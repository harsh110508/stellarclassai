# Test fixture

`sdss_fixture.csv` holds 792 real rows from the SDSS17 dataset, sampled with fixed seeds:

* 451 GALAXY rows: 450 distinct galaxies, plus one of them **intentionally repeated once** so the
  de-duplication step can be tested
* 301 STAR rows: 300 random stars, plus the single real row that has `-9999` sentinel magnitudes
* 40 QSO rows

The values are unmodified SDSS DR17 measurements. The fixture lets the tests run offline and quickly.
