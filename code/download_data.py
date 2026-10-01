"""Download the public data used in the audit into ./work (relative to the repository root).

Sources: NCBI GEO (FTP), UCSC Xena / GDC (TCGA-CESC) and the Ensembl REST API (gene symbol to ID map).
No patient-level data beyond these public accessions are used or distributed.

This script has NOT been run end to end since it was assembled from the original download scripts;
check that each file appears and that the sizes look sensible.
"""
import json
import shutil
import tarfile
from pathlib import Path
from urllib.request import Request, urlopen

REPO = Path(__file__).resolve().parents[1]
W = REPO / 'work'
GEO = 'https://ftp.ncbi.nlm.nih.gov/geo/'
FILES = {
    'GSE236738_RAW.tar': GEO + 'series/GSE236nnn/GSE236738/suppl/GSE236738_RAW.tar',
    'data_gse297038/GSE297038_17pairs_counts_merged.tsv.gz': GEO + 'series/GSE297nnn/GSE297038/suppl/GSE297038_17pairs_counts_merged.tsv.gz',
    'data_gse297038/GSE297038_family.soft.gz': GEO + 'series/GSE297nnn/GSE297038/soft/GSE297038_family.soft.gz',
    'data_gse297038/GSE297038_series_matrix.txt.gz': GEO + 'series/GSE297nnn/GSE297038/matrix/GSE297038_series_matrix.txt.gz',
    'validation/GSE63514_series_matrix.txt.gz': GEO + 'series/GSE63nnn/GSE63514/matrix/GSE63514_series_matrix.txt.gz',
    'validation/GSE6213_series_matrix.txt.gz': GEO + 'series/GSE6nnn/GSE6213/matrix/GSE6213_series_matrix.txt.gz',
    'validation/GPL570.annot.gz': GEO + 'platforms/GPLnnn/GPL570/annot/GPL570.annot.gz',
    'validation/GPL2895.annot.gz': GEO + 'platforms/GPL2nnn/GPL2895/annot/GPL2895.annot.gz',
    'data_gse56363/GSE56363_series_matrix.txt.gz': GEO + 'series/GSE56nnn/GSE56363/matrix/GSE56363_series_matrix.txt.gz',
    'data_gse56363/GSE56363_family.soft.gz': GEO + 'series/GSE56nnn/GSE56363/soft/GSE56363_family.soft.gz',
    'external_public/GSE70035/GSE70035_series_matrix.txt.gz': GEO + 'series/GSE70nnn/GSE70035/matrix/GSE70035_series_matrix.txt.gz',
    'external_public/GSE70035/GSE70035_family.soft.gz': GEO + 'series/GSE70nnn/GSE70035/soft/GSE70035_family.soft.gz',
    'external_public/GSE168009/GSE168009_Raw_count.txt.gz': GEO + 'series/GSE168nnn/GSE168009/suppl/GSE168009_Raw_count.txt.gz',
    'GSE224327_RAW.tar': GEO + 'series/GSE224nnn/GSE224327/suppl/GSE224327_RAW.tar',
    'data_tcga_cesc/TCGA-CESC.star_tpm.tsv.gz': 'https://gdc.xenahubs.net/download/TCGA-CESC.star_tpm.tsv.gz',
    'data_tcga_cesc/surv.tsv.gz': 'https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-CESC.survival.tsv.gz',
}


def get(url, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print('have', out.relative_to(REPO)); return
    with urlopen(Request(url, headers={'User-Agent': 'Mozilla/5.0'}), timeout=180) as r, out.open('wb') as f:
        shutil.copyfileobj(r, f)
    print('got ', out.relative_to(REPO), out.stat().st_size)


for rel, url in FILES.items():
    get(url, W / rel)

# GSE168009 series metadata (XML) from the GEO query page
get('https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE168009&targ=self&form=xml&view=quick', W / 'external_public/GSE168009/GSE168009_family.xml')

# GSE236738: unpack the per-sample matrices used by the single-cell analysis
mat = W / 'GSE236738_matrix'
if not mat.exists():
    mat.mkdir(parents=True)
    with tarfile.open(W / 'GSE236738_RAW.tar') as t:
        t.extractall(mat)

# GSE224327: unpack the per-sample matrices used for the second-cohort leakage demonstration
mat2 = W / 'data_gse224327' / 'extracted'
if not mat2.exists():
    mat2.mkdir(parents=True)
    with tarfile.open(W / 'GSE224327_RAW.tar') as t2:
        t2.extractall(mat2)

# additional public cohorts for the leakage demonstration (unpacked to work/additional_cohorts/<accession>)
for acc, suppl in (('GSE228499', 'GSE228nnn'), ('GSE292163', 'GSE292nnn'), ('GSE237425', 'GSE237nnn'), ('GSE197461', 'GSE197nnn')):
    tarpath = W / 'additional_cohorts' / acc / f'{acc}_RAW.tar'
    get(GEO + f'series/{suppl}/{acc}/suppl/{acc}_RAW.tar', tarpath)
    if not any(tarpath.parent.glob('*_matrix.mtx.gz')):
        with tarfile.open(tarpath) as t3:
            t3.extractall(tarpath.parent)

# GSE300897 (ovarian, chemo-refractory vs sensitive) and GSE173682 (endometrial/ovarian, RNA files only; the series archive is 17 GB because of ATAC data)
for fn in ('GSE300897_UMIcounts_HGSC.tsv.gz', 'GSE300897_annotation_HGSC.tsv.gz'):
    get(GEO + f'series/GSE300nnn/GSE300897/suppl/{fn}', W / 'additional_cohorts' / 'GSE300897' / fn)
for gsm, stem in (('GSM5276933', '3533EL'), ('GSM5276934', '3571DL'), ('GSM5276935', '36186L'), ('GSM5276936', '36639L'), ('GSM5276937', '366C5L'), ('GSM5276938', '37EACL'),
                  ('GSM5276939', '38FE7L'), ('GSM5276940', '3BAE2L'), ('GSM5276941', '3CCF1L'), ('GSM5276942', '3E4D1L'), ('GSM5276943', '3E5CFL')):
    for kind, ext in (('barcodes', 'tsv.gz'), ('features', 'tsv.gz'), ('matrix', 'mtx.gz')):
        fn = f'{gsm}_{kind}-{stem}.{ext}'
        get(GEO + f'samples/{gsm[:7]}nnn/{gsm}/suppl/{fn}', W / 'additional_cohorts' / 'GSE173682' / fn)

# TCGA-CESC clinical fields from the GDC API
clin = W / 'data_tcga_cesc' / 'clinical.json'
if not clin.exists():
    fields = 'submitter_id,demographic.vital_status,demographic.days_to_death,diagnoses.days_to_last_follow_up,diagnoses.days_to_recurrence,diagnoses.tumor_stage'
    flt = '%7B%22op%22:%22in%22,%22content%22:%7B%22field%22:%22project.project_id%22,%22value%22:%5B%22TCGA-CESC%22%5D%7D%7D'
    with urlopen('https://api.gdc.cancer.gov/cases?filters=' + flt + '&fields=' + fields + '&format=json&size=1000', timeout=180) as r:
        clin.write_text(json.dumps(json.load(r), indent=2), encoding='utf-8')

# TCGA-CESC clinical covariates (age, FIGO stage, histology) from the GDC API
cov = W / 'data_tcga_cesc' / 'clinical_covariates.json'
if not cov.exists():
    flt = '%7B%22op%22:%22in%22,%22content%22:%7B%22field%22:%22project.project_id%22,%22value%22:%5B%22TCGA-CESC%22%5D%7D%7D'
    fields = 'submitter_id,demographic.age_at_index,diagnoses.figo_stage,diagnoses.ajcc_clinical_stage,diagnoses.primary_diagnosis,diagnoses.tumor_grade'
    with urlopen('https://api.gdc.cancer.gov/cases?filters=' + flt + '&fields=' + fields + '&format=json&size=1000', timeout=180) as r:
        cov.write_text(json.dumps(json.load(r)), encoding='utf-8')

# gene symbol -> Ensembl ID map for the 29 program genes (used for RNA-seq cohorts); shipped with the repository
shutil.copy(REPO / 'code' / 'symbol2ensg.json', W / 'data_gse297038' / 'symbol2ensg.json')
print('done. GSE3578, GSE208654 and the Visium data are not needed for the audit.')
