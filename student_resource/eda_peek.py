import pandas as pd
import numpy as np
import sys
import os

# Force UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

base = r'c:\Coding\ML challange\student_resource\dataset'

files = {
    'train_source1': f'{base}/train/train_source1.tsv',
    'train_source2': f'{base}/train/train_source2.tsv',
    'train_source3': f'{base}/train/train_source3.tsv',
    'train_ground_truth': f'{base}/train/train_ground_truth.tsv',
    'test_source1':  f'{base}/test/test_source1.tsv',
    'test_source2':  f'{base}/test/test_source2.tsv',
    'test_source3':  f'{base}/test/test_source3.tsv',
}

print("=" * 70)
print("STEP 1: SHAPE & COLUMNS")
print("=" * 70)
for name, path in files.items():
    df = pd.read_csv(path, sep='\t', encoding='utf-8', dtype=str)
    print(f"\n{name}:")
    print(f"  Shape     : {df.shape}")
    print(f"  Columns   : {list(df.columns)}")
    print(f"  Dtypes    : {dict(df.dtypes)}")

print("\n" + "=" * 70)
print("STEP 2: NULL / MISSING VALUE ANALYSIS")
print("=" * 70)
for name, path in files.items():
    df = pd.read_csv(path, sep='\t', encoding='utf-8', dtype=str)
    total = len(df)
    null_counts = df.isnull().sum()
    empty_str = (df == '').sum()
    print(f"\n{name} (total rows: {total:,}):")
    for col in df.columns:
        nc = null_counts[col]
        ec = empty_str[col]
        print(f"  {col:30s} | NaN: {nc:6,} ({nc/total*100:.2f}%) | Empty str: {ec:6,} ({ec/total*100:.2f}%)")

print("\n" + "=" * 70)
print("STEP 3: COUNTRY DISTRIBUTION")
print("=" * 70)
for name, path in [
    ('train_source1', f'{base}/train/train_source1.tsv'),
    ('train_source2', f'{base}/train/train_source2.tsv'),
    ('train_source3', f'{base}/train/train_source3.tsv'),
    ('test_source1',  f'{base}/test/test_source1.tsv'),
    ('test_source2',  f'{base}/test/test_source2.tsv'),
    ('test_source3',  f'{base}/test/test_source3.tsv'),
]:
    df = pd.read_csv(path, sep='\t', encoding='utf-8', dtype=str)
    print(f"\n{name}:")
    if 'country' in df.columns:
        print(df['country'].value_counts(dropna=False).to_string())

print("\n" + "=" * 70)
print("STEP 4: SAMPLE ROWS (3 per source)")
print("=" * 70)
for name, path in files.items():
    df = pd.read_csv(path, sep='\t', encoding='utf-8', dtype=str, nrows=3)
    print(f"\n{name}:")
    for _, row in df.iterrows():
        for col in df.columns:
            print(f"  {col}: {str(row[col])[:120]}")
        print()

print("\n" + "=" * 70)
print("STEP 5: NAME & ADDRESS LENGTH STATS")
print("=" * 70)
for name, path in [
    ('train_source1', f'{base}/train/train_source1.tsv'),
    ('train_source2', f'{base}/train/train_source2.tsv'),
    ('train_source3', f'{base}/train/train_source3.tsv'),
]:
    df = pd.read_csv(path, sep='\t', encoding='utf-8', dtype=str)
    for col in ['business_name', 'business_address']:
        if col in df.columns:
            lens = df[col].fillna('').str.len()
            print(f"\n{name} | {col}:")
            print(f"  min={lens.min()}, max={lens.max()}, mean={lens.mean():.1f}, median={lens.median()}, std={lens.std():.1f}")
            print(f"  zeros/empty: {(lens==0).sum():,}")

print("\n" + "=" * 70)
print("STEP 6: GROUND TRUTH ANALYSIS")
print("=" * 70)
gt = pd.read_csv(f'{base}/train/train_ground_truth.tsv', sep='\t', encoding='utf-8', dtype=str)
gt['matched_entity_ids'] = gt['matched_entity_ids'].fillna('')
gt['match_count'] = gt['matched_entity_ids'].apply(lambda x: len(x.split(',')) if x.strip() else 0)
print(f"Total S1 entities in GT : {len(gt):,}")
print(f"Singletons (0 matches)  : {(gt['match_count']==0).sum():,} ({(gt['match_count']==0).mean()*100:.1f}%)")
print(f"Has >=1 match           : {(gt['match_count']>=1).sum():,} ({(gt['match_count']>=1).mean()*100:.1f}%)")
print(f"Match count distribution:")
print(gt['match_count'].value_counts().sort_index().head(20).to_string())

# S2 vs S3 match breakdown
all_ids = ','.join(gt['matched_entity_ids'].dropna())
s2_count = all_ids.count('S2-')
s3_count = all_ids.count('S3-')
print(f"\nTotal S2 matches in GT  : {s2_count:,}")
print(f"Total S3 matches in GT  : {s3_count:,}")

print("\nDone!")
