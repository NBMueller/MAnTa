#!/usr/bin/env python3

import argparse
import gc
import os

import h5py
import loompy
import numpy as np
import pandas as pd

EPSILON = np.finfo(np.float64).resolution

CHR_ORDER = {str(i): i for i in range(1, 23)}
CHR_ORDER.update({'X':23, 'Y':24})

DEF_THRESHOLDS = {
    'min_dp': 10,
    'min_gq': 30,
    'min_cell_geno': 0.5,
    'min_mutated': 0.05,
    'min_vaf': 0.2,
    'min_var_geno': 0.5,
    'max_ref_vaf': 0.05,
    'min_het_vaf': 0.35,
    'min_hom_vaf': 0.95,
    'proximity': [25, 50, 100, 200]
}


def concat_str_arrays(arrays, sep='_'):
    out_arr = arrays[0].astype(str)
    for arr in arrays[1:]:
        out_arr = np.char.add(out_arr, sep)
        out_arr = np.char.add(out_arr, arr.astype(str))
    return out_arr

# ---------------------------------- H5 ----------------------------------------

def load_h5_snps(h5_file, thresholds=DEF_THRESHOLDS):
    with h5py.File(h5_file, 'r') as f:
        rel_loci = ~f['assays']['dna_variants']['ca']['filtered'][()]
        rel_cells = ~f['assays']['dna_variants']['ra']['filtered'][()]

        gt = f['assays']['dna_variants']['layers']['NGT'][()].T
        # (64477, 37)
        # Sixth filter: Remove variants mutated in < X % of cells; 
        #   done first in mosaic
        mut = (gt == 1) | (gt == 2)
        # Relative value
        if thresholds['min_mutated'] < 1:
            mut_var = np.mean(mut, axis=1) >= thresholds['min_mutated']
        # Absolute value
        else:
            mut_var = np.sum(mut, axis=1) >= thresholds['min_mutated']
        gt = gt[mut_var]

        DP = f['assays']['dna_variants']['layers']['DP'][:,mut_var].T
        GQ = f['assays']['dna_variants']['layers']['GQ'][:,mut_var].T
        VAF = f['assays']['dna_variants']['layers']['AF'][:,mut_var].T / 100
        AD = (DP * VAF).astype('int16')
        RO = DP - AD
        ampl = f['assays']['dna_variants']['ca']['amplicon'][mut_var].astype(str)
        chrom = f['assays']['dna_variants']['ca']['CHROM'][mut_var].astype(str)
        pos = f['assays']['dna_variants']['ca']['POS'][mut_var]
        ref = f['assays']['dna_variants']['ca']['REF'][mut_var].astype(str)
        alt = f['assays']['dna_variants']['ca']['ALT'][mut_var].astype(str)
        cells = f['assays']['dna_variants']['ra']['barcode'][()].astype(str)

    return to_output_format(gt, DP, GQ, AD, RO, ampl, chrom, pos, ref, alt, 
        cells, thresholds)


def load_h5_reads(h5_file):
    with h5py.File(h5_file, 'r') as f:
        # Get reads per ampl count
        reads_cnt = f['assays']['dna_read_counts']['layers']['read_counts'][()]
        reads_cells = f['assays']['dna_read_counts']['ra']['barcode'][()] \
            .astype(str)
        reads_ampl = f['assays']['dna_read_counts']['ca']['id'][()] \
            .astype(str)

        reads_df = pd.DataFrame(reads_cnt, index=reads_cells, columns=reads_ampl)
        reads_df.index.name = 'cell_barcode'
    return reads_df


# --------------------------------- LOOM ---------------------------------------


def load_loom_snps(loom_file, thresholds=DEF_THRESHOLDS):
    with loompy.connect(loom_file) as ds:
        gt = ds[:,:]
        # (64477, 37)
        # Sixth filter: Remove variants mutated in < X % of cells; 
        #   done first in mosaic
        mut = (gt == 1) | (gt == 2)
        # Relative value
        if thresholds['min_mutated'] < 1:
            mut_var = np.mean(mut, axis=1) >= thresholds['min_mutated']
        # Absolute value
        else:
            mut_var = np.sum(mut, axis=1) >= thresholds['min_mutated']
        gt = gt[mut_var]
        DP = ds.layers['DP'][mut_var,:][:,:]
        GQ = ds.layers['GQ'][mut_var,:][:,:]
        AD = ds.layers['AD'][mut_var,:][:,:]
        RO = ds.layers['RO'][mut_var,:][:,:]
        ampl = ds.ra['amplicon'][mut_var]
        chrom = ds.ra['CHROM'][mut_var]
        pos = ds.ra['POS'][mut_var]
        ref = ds.ra['REF'][mut_var]
        alt = ds.ra['ALT'][mut_var]
        cells = ds.col_attrs['barcode']

    return to_output_format(gt, DP, GQ, AD, RO, ampl, chrom, pos, ref, alt,
        cells, thresholds)


# ------------------------------- GENERAL --------------------------------------

def preprocess_data(in_file, out_file='', thresholds=DEF_THRESHOLDS):
    print(f'Processing file: {in_file}')
    if in_file.endswith('.loom'):
        in_type = 'loom'
        df, gt, VAF = load_loom_snps(in_file, thresholds)
    else:
        in_type = 'h5'
        df, gt, VAF = load_h5_snps(in_file, thresholds)
        df_reads = load_h5_reads(in_file)
        
    # post-process filtering
    df_f = filter_variants(df, gt, VAF, thresholds)
    df_out = filter_variants_consecutive(df_f, thresholds['proximity'])

    SNP_id = df_out['CHR'] + ':' + df_out['POS'].astype(str) + ':' \
        + df_out['REF'] + '/' + df_out['ALT']
    cells = df_out.columns[7:].values

    if not out_file:
        out_base = in_file.split('.')[0]
    else:
        if out_file.endswith('.filtered_variants.csv'):
            out_base = out_file[:-22]
        else:
            out_base = os.path.splitext(out_file)[0]

    out_dir = os.path.dirname(out_base)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f'Creating output directory: {out_dir}')

    variant_file = f'{out_base}.{in_type}.filtered_variants.csv'
    print(f'Writing variant  file to: {variant_file}')
    df_out.to_csv(variant_file, index=False, header=True)

    if in_file.endswith('.h5'):
        reads_file = f'{out_base}.h5.barcode.cell.distribution.merged.tsv'
        print(f'Writing read file to: {reads_file}')
        df_reads.to_csv(reads_file, index=True, header=True, sep='\t')
        return variant_file, reads_file
    return variant_file


def to_output_format(gt, DP, GQ, AD, RO, ampl, chrom, pos, ref, alt, cells,
        thresholds):
    VAF = np.where(DP > 0, (AD + EPSILON) / (DP + EPSILON), 0)
    filter_low_quality(gt, GQ, DP, VAF, thresholds)
    del DP
    del GQ
    gc.collect()

    keep_var, keep_cells = filter_low_fractions(gt, thresholds)

    gt = gt[keep_var]
    AD = AD[keep_var][:,keep_cells]
    RO = RO[keep_var][:,keep_cells]
    VAF = VAF[keep_var][:,keep_cells]
    cells = cells[keep_cells]

    variants_info = {
        'CHR': chrom[keep_var],
        'POS': pos[keep_var],
        'REF': ref[keep_var],
        'ALT': alt[keep_var],
        'REGION': ampl[keep_var],
        'NAME': ampl[keep_var],
        'FREQ': np.zeros(np.sum(keep_var))
    }
    
    info = concat_str_arrays([RO, AD, gt], ':').T
    for i, cell in enumerate(cells):
        variants_info[cell] = info[i]

    # Sort dataframe by chr:pos
    df = pd.DataFrame(variants_info)
    df['CHR_ORDER'] = df['CHR'].map(CHR_ORDER)
    df.sort_values(['CHR_ORDER', 'POS'], inplace=True)
    df.drop('CHR_ORDER', axis=1, inplace=True)
    gt = gt[df.index.values]
    VAF = VAF[df.index.values]
    df.reset_index(drop=True, inplace=True)

    return df, gt, VAF


def filter_low_quality(gt, GQ, DP, VAF, thresholds):
    # Filters 1-3: done second in mosaic
    # First filter: Remove genotype in cell with quality < X
    gt[GQ < thresholds['min_gq']] = 3
    # Second filter: Remove genotype in cell with read depth < X
    gt[DP < thresholds['min_dp']] = 3
    # Third filter: Remove genotype in cell with alternate allele freq < X
    gt[((gt == 1) | (gt == 2)) & (VAF < thresholds['min_vaf'])] = 3


def filter_low_fractions(gt, thresholds):
    # Fourth filter: Remove variants genotyped in < X % of cells; 
    #     done last/fourth in mosaic
    keep_var1 = np.mean(gt == 3, axis=1) < (1 - thresholds['min_var_geno'])

    # Fifth filter: Remove cells with < X %of genotypes present; 
    #     done third in mosaic
    keep_cells = np.mean(gt[keep_var1] == 3, axis=0) \
        < (1 - thresholds['min_cell_geno'])

    # Filter second time for variants present in >X percent of data
    gt = gt[:,keep_cells]
    keep_var2 = np.mean((gt == 1) | (gt == 2), axis=1) > 0.01
    # Get only variants passing both filters
    keep_var = keep_var1 & keep_var2
    return keep_var, keep_cells


def filter_variants(df, gt, VAF, thresholds):
    ms1 = (gt == 0) & (VAF > thresholds['max_ref_vaf'])
    ms2 = (gt == 1) & (VAF < thresholds['min_het_vaf'])
    ms3 = (gt == 2) & (VAF < thresholds['min_hom_vaf'])
    ms = ms1 | ms2 | ms3

    gt[ms] = 3
    df.iloc[:,7:] = df.iloc[:,7:] \
        .where(~ms, df.iloc[:,7:].map(lambda x: x[:-1] + '3'))

    keep_var1 = np.mean(gt == 3, axis=1) < (1 - thresholds['min_var_geno'])

    if thresholds['min_mutated'] < 1:
        keep_var2 = np.mean((gt == 1) | (gt == 2), axis=1) \
            >= thresholds['min_mutated']
    elif thresholds['min_mutated'] == 1:
        raise TypeError('--min_mutated cannot be equactly 1. Values <1 are ' \
            'interpreted as cell fraction, values >1 as absolute cell number.')
    else:
        keep_var2 = np.sum((gt == 1) | (gt == 2), axis=1) >= thresholds['min_mutated']

    keep = keep_var1 & keep_var2

    return df[keep].reset_index(drop=True)


def filter_variants_consecutive(df, proximity):
    keep = np.ones(df.shape[0], dtype=bool)

    chrom = df['CHR'].values
    pos = df['POS'].values

    loc = 0
    while loc < len(chrom):
        found = 0
        fa = np.argwhere((chrom == chrom[loc]) & (np.arange(len(chrom)) > loc)) \
            .flatten()[::-1]
        for jj in fa:
            for ii in range(len(proximity)):
                if (pos[loc] + proximity[ii] > pos[jj]) & (jj - loc > ii):
                    found = 1
            if found == 1:
                keep[np.arange(loc, jj + 1)] = False
                loc = jj + 1
                break
        if found == 0:
            loc += 1

    return df[keep].reset_index(drop=True)


def parse_args():
    parser = argparse.ArgumentParser(prog='mosaic preprocessing',
        usage='python preprocessing.py <DATA> [-args]',
        description='***Filter a loom or h5 file equal to MissionBio mosaic***')
    parser.add_argument('input', type=str, help='Input loom or h5 file.')
    parser.add_argument('-o', '--output', type=str, help='Output file')
    # thresholds
    parser.add_argument('-mGQ', '--min_gq', type=int,
        default=DEF_THRESHOLDS['min_gq'],
        help='Minimum GQ to consider a variant (per cell).')
    parser.add_argument('-mDP', '--min_dp', type=int,
        default=DEF_THRESHOLDS['min_dp'],
        help='Minimum depth to consider a variant (per cell) .')
    parser.add_argument('-mAF', '--min_vaf', type=float, 
        default=DEF_THRESHOLDS['min_vaf'],
        help='Minimum alternate VAF to consider a variant (per cell).')
    parser.add_argument('-mVG', '--min_var_geno', type=float,
        default=DEF_THRESHOLDS['min_var_geno'],
        help='Minimum fractions of genotyped loci to consider a variant.')
    parser.add_argument('-mCG', '--min_cell_geno', type=float,
        default=DEF_THRESHOLDS['min_cell_geno'],
        help='Minimum fractions of genotyped cells to consider a cell.')
    parser.add_argument('-mMut', '--min_mutated', type=float,
        default=DEF_THRESHOLDS['min_mutated'],
        help='Minimum fractions or cell number of het/hom cells to consider a variant.')
    parser.add_argument('-refVAF', '--max_ref_vaf', type=float,
        default=DEF_THRESHOLDS['max_ref_vaf'],
        help='Maximum VAF to consider a wildtype variant (per cell).')
    parser.add_argument('-homVAF', '--min_hom_vaf', type=float,
        default=DEF_THRESHOLDS['min_hom_vaf'],
        help='Minimum VAF to consider a homozygous variant (per cell).')
    parser.add_argument('-hetVAF', '--min_het_vaf', type=float,
        default=DEF_THRESHOLDS['min_het_vaf'],
        help='Minimum VAF to consider a heterozygous variant (per cell).')
    parser.add_argument('-p', '--proximity', nargs='+', type=int,
        default=DEF_THRESHOLDS['proximity'], help='If "i + 1" variants are within ' \
            '"proximity[i]", then the variants are removed.')

    return parser.parse_args()


if __name__ == '__main__':
    args = vars(parse_args())
    in_file = args.pop('input')
    out_file = args.pop('output')
    
    if not in_file.endswith(('.h5', '.loom')):
        raise TypeError(f'Input file {in_file} not .loom or .h5')
    preprocess_data(in_file, out_file, args)

