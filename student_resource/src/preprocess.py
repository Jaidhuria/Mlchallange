"""
Simple Dataset Cleaning
=======================
1. Source files  → DROP rows with any null in business_name or business_address
2. Ground truth  → Fill matched_entity_ids NaN with "" (singletons are valid)
3. All text cols → lowercase, strip whitespace, collapse extra spaces
4. No numeric cols exist → no mean/std imputation needed
5. Save to dataset/processed/

Run: python src/preprocess.py
"""

import sys
import re
import pandas as pd
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

BASE      = Path(r"c:\Coding\ML challange\student_resource")
TRAIN_DIR = BASE / "dataset" / "train"
TEST_DIR  = BASE / "dataset" / "test"
OUT_DIR   = BASE / "dataset" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RE_SPACES = re.compile(r'\s{2,}')

# ──────────────────────────────────────────────
def clean_text(s: pd.Series) -> pd.Series:
    """Lowercase + strip + collapse multiple spaces."""
    return (s.fillna('')
             .str.strip()
             .str.lower()
             .str.replace(RE_SPACES, ' ', regex=True))


def process_source(path: Path, label: str) -> pd.DataFrame:
    print(f"\n[{label}] Loading ...", flush=True)
    df = pd.read_csv(path, sep='\t', encoding='utf-8', dtype=str, low_memory=False)
    orig_len = len(df)
    print(f"  Original rows : {orig_len:>10,}", flush=True)

    # ── NULL HANDLING: DROP rows with null in key text cols ──
    null_name = df['business_name'].isna().sum()
    null_addr = df['business_address'].isna().sum()
    print(f"  Null business_name    : {null_name:,} → rows DROPPED", flush=True)
    print(f"  Null business_address : {null_addr:,} → rows DROPPED", flush=True)

    df = df.dropna(subset=['business_name', 'business_address'])
    dropped = orig_len - len(df)
    print(f"  Rows after drop : {len(df):>10,}  (removed {dropped:,} rows = {dropped/orig_len*100:.2f}%)", flush=True)

    # ── DUPLICATE entity_id check ──
    dups = df['entity_id'].duplicated().sum()
    if dups:
        print(f"  Duplicate entity_ids : {dups:,} → DROPPED (keep first)", flush=True)
        df = df.drop_duplicates(subset=['entity_id'], keep='first')

    # ── TEXT NORMALISATION (vectorised) ──
    print(f"  Normalising text ...", flush=True)
    df['business_name']    = clean_text(df['business_name'])
    df['business_address'] = clean_text(df['business_address'])
    df['country']          = clean_text(df['country'])

    # ── VERIFY: no nulls remain ──
    assert df['business_name'].isna().sum()    == 0, "Null names remain!"
    assert df['business_address'].isna().sum() == 0, "Null addresses remain!"
    assert df['country'].isna().sum()          == 0, "Null country remain!"

    # ── POST-CLEAN STATS ──
    print(f"  ✓ No nulls remain", flush=True)
    print(f"  Country distribution:", flush=True)
    print(df['country'].value_counts().to_string(), flush=True)
    print(f"  business_name  len  mean={df['business_name'].str.len().mean():.1f}  "
          f"std={df['business_name'].str.len().std():.1f}  "
          f"min={df['business_name'].str.len().min()}  "
          f"max={df['business_name'].str.len().max()}", flush=True)
    print(f"  business_address len mean={df['business_address'].str.len().mean():.1f}  "
          f"std={df['business_address'].str.len().std():.1f}  "
          f"min={df['business_address'].str.len().min()}  "
          f"max={df['business_address'].str.len().max()}", flush=True)

    return df


def process_ground_truth(path: Path) -> pd.DataFrame:
    print(f"\n[ground_truth] Loading ...", flush=True)
    gt = pd.read_csv(path, sep='\t', encoding='utf-8', dtype=str)
    orig_len = len(gt)

    null_matches = gt['matched_entity_ids'].isna().sum()
    # NaN here = legitimate singleton (no match) → fill with "" not drop!
    gt['matched_entity_ids'] = gt['matched_entity_ids'].fillna('')
    print(f"  Singleton NaN → '' : {null_matches:,}  (valid — these entities have NO matches)", flush=True)
    print(f"  Total rows     : {orig_len:,}", flush=True)
    print(f"  Singletons (%) : {null_matches/orig_len*100:.2f}%", flush=True)

    # Match count stats
    gt['match_count'] = gt['matched_entity_ids'].apply(
        lambda x: len(x.split(',')) if x.strip() else 0)
    print(f"  Match count distribution:", flush=True)
    print(gt['match_count'].value_counts().sort_index().head(12).to_string(), flush=True)

    assert gt['matched_entity_ids'].isna().sum() == 0
    print(f"  ✓ No nulls remain", flush=True)
    return gt


def main():
    import time
    t0 = time.time()

    print("=" * 60)
    print("Dataset Cleaning Pipeline")
    print("=" * 60)
    print("Strategy:")
    print("  - NULL rows in source files   → DROPPED")
    print("  - NULL in ground truth matches → filled '' (singletons)")
    print("  - Text columns                → lowercase + strip + collapse spaces")
    print("  - Numeric columns             → NONE exist in this dataset")
    print("=" * 60)

    # ── Process all files ──
    results = {}

    results['train_s1'] = process_source(TRAIN_DIR / 'train_source1.tsv', 'train_s1')
    results['train_s2'] = process_source(TRAIN_DIR / 'train_source2.tsv', 'train_s2')
    results['train_s3'] = process_source(TRAIN_DIR / 'train_source3.tsv', 'train_s3')
    results['gt']       = process_ground_truth(TRAIN_DIR / 'train_ground_truth.tsv')
    results['test_s1']  = process_source(TEST_DIR / 'test_source1.tsv',  'test_s1')
    results['test_s2']  = process_source(TEST_DIR / 'test_source2.tsv',  'test_s2')
    results['test_s3']  = process_source(TEST_DIR / 'test_source3.tsv',  'test_s3')

    # ── Save ──
    print("\n" + "=" * 60)
    print("SAVING cleaned files ...", flush=True)

    save_map = {
        'train_source1_clean.tsv':      'train_s1',
        'train_source2_clean.tsv':      'train_s2',
        'train_source3_clean.tsv':      'train_s3',
        'train_ground_truth_clean.tsv': 'gt',
        'test_source1_clean.tsv':       'test_s1',
        'test_source2_clean.tsv':       'test_s2',
        'test_source3_clean.tsv':       'test_s3',
    }

    print(f"\n{'File':<45} {'Rows':>12}  {'Size MB':>8}", flush=True)
    print("-" * 68, flush=True)
    for fname, key in save_map.items():
        df  = results[key]
        out = OUT_DIR / fname
        df.to_csv(out, sep='\t', index=False, encoding='utf-8')
        mb  = out.stat().st_size / 1_048_576
        print(f"  {fname:<43} {len(df):>12,}  {mb:>7.1f} MB", flush=True)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"DONE in {elapsed/60:.1f} minutes")
    print(f"Output: {OUT_DIR}")

    # ── Final summary table ──
    print(f"\nFINAL CLEANED ROW COUNTS:")
    print(f"  train_source1 : {len(results['train_s1']):>10,}")
    print(f"  train_source2 : {len(results['train_s2']):>10,}")
    print(f"  train_source3 : {len(results['train_s3']):>10,}")
    print(f"  ground_truth  : {len(results['gt']):>10,}")
    print(f"  test_source1  : {len(results['test_s1']):>10,}")
    print(f"  test_source2  : {len(results['test_s2']):>10,}")
    print(f"  test_source3  : {len(results['test_s3']):>10,}")


if __name__ == '__main__':
    main()
