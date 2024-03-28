#!/usr/bin/env python3

from copy import deepcopy
from itertools import cycle
import os

import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy.cluster.hierarchy import linkage, cut_tree, leaves_list
from scipy.spatial.distance import pdist, euclidean
from scipy.special import factorial
from scipy.stats import kstest


EPSILON = np.finfo(np.float64).resolution
CHR_ORDER = {str(i): i for i in range(1, 23)}
CHR_ORDER.update({'X':23, 'Y':24})
PANEL_COLS = ['CHR', 'Start', 'End', 'Gene', 'Exon', 'Strand', 'Feature',
    'Biotype', 'Ensembl_ID', 'TSL', 'HUGO', 'Tx_overlap_%', 'Exon_overlaps_%', 
    'CDS_overlaps_%', 'Amplicon']


CHR_COLORS_raw = cycle(['#f4f4f4','#c3c4c3'])
CHR_COLORS = [(i, next(CHR_COLORS_raw)) for i in np.linspace(0, 1, 24)]

GENE_MAX = 96
GENE_COLORS_raw = cycle(['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99',
    '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928'])
GENE_COLORS = [(i, next(GENE_COLORS_raw)) for i in np.linspace(0, 1, GENE_MAX)]

CLUSTER_COLORS_raw = cycle(['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99',
    '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928'])


# Discretized
# CNV_COLORS = [
#     (0.000, '#2f66c5'), # 0.0 - 0.5: Dark Blue 
#     (0.083, '#2f66c5'), # 0.0 - 0.5: Dark Blue 
#     (0.083, '#1b5fff'), # 0.0 - 0.5: Dark Blue 
#     (0.250, '#1b5fff'), # 0.5 - 1.5: Light Blue
#     (0.250, '#ffffff'), # 1.5 - 2.5: White
#     (0.417, '#ffffff'), # 1.5 - 2.5: White
#     (0.417, '#ff0000'), # 2.5 - 3.5: Red
#     (0.583, '#ff0000'), # 2.5 - 3.5: Red
#     (0.583, '#C60000'), # 3.5 - 4.5: Darker Red
#     (0.725, '#C60000'), # 3.5 - 4.5: Darker Red
#     (0.725, '#b90000'), # 4.5 - 5.5: Dark Red
#     (0.917, '#b90000'), # 4.5 - 5.5: Dark Red
#     (0.917, '#000000'), # 5.5 - 6: Black
#     (1.000, '#000000'), # 5.5 - 6: Black
# ]

# Continuous
CNV_COLORS = [
    (0.000, '#2f66c5'), # 0 - 1: Dark Blue 
    (0.167, '#1b5fff'), # 1 - 2: Light Blue
    (0.333, '#ffffff'), # 2 - 3: White
    (0.500, '#ff0000'), # 3 - 4: Red
    (0.666, '#C60000'), # 4 - 5: Darker Red
    (0.833, '#b90000'), # 5 - 6: Dark Red
    (1.000, '#000000'), # 5.5 - 6: Black
]

SNP_COLORS = [
    (0.000, '#ffffff'), # White
    (0.500, '#ff9d00'), # Yellow
    (1.000, '#ff0000') # Red
]
LIB_COLORS = [
    (0.0, '#480000'),
    (0.2, '#921200'),
    (0.4, '#da5a00'),
    (0.6, '#ffa424'),
    (0.8, '#ffec6d'), 
    (1.0, '#ffffb6')
]


# ------------------------------------------------------------------------------

class TapestriDNA:
    def __init__(self, panel):
        self.panel = panel


    def load_sample_data(self, read_file, SNP_file):
        self.prefix = os.path.basename(read_file).split('.barcode')[0]
        self.out_dir, _ = os.path.split(read_file)
        print(f'\nLoading sample {self.prefix}')
        print(f'\tSNPs  file: {SNP_file}')
        self.SNPs = SNPData(SNP_file)
        self.cells = pd.DataFrame(np.zeros(self.SNPs.df.shape[0]),
            index=self.SNPs.df.index, columns=['cluster'])
        self.cells.index.name = 'barcode'
        self.cells['assignment'] = 'unknown'

        cell_order = self.get_cell_order()
        print(f'\treads rile: {read_file}')
        self.reads = ReadData(read_file, cell_order)
        self.depth = DepthData(read_file, cell_order)

        if self.reads.df_in.shape[0] != self.SNPs.df.shape[0]:
            print('!Warning: Number of cells in read and SNP files do not match. '\
                    'Taking only cells present in SNP data')
        print('Loading sample data - done')
        # Init functions for efficiently calculating joint dist
        self.dist_joint = np.stack([self.SNPs.dist, self.reads.dist], axis=1)
        cells1m = self.cells.shape[0] - 1
        self.cell_start_id = np.array([sum(range(cells1m, cells1m - i, -1)) \
            for i in range(cells1m + 1)])


    def safe_annotation(self):
        out_file = self.get_out_file()
        self.cells.to_csv(out_file)


    def get_out_file(self):
        return os.path.join(self.out_dir, f'{self.prefix}_annotated.csv')


    def get_cell_order(self):
        return self.cells.index.values


    def get_cluster_number(self):
        return self.cells['cluster'].nunique()


    def get_dist_cell_ids(self, cells):
        cl_cell_ids = self.SNPs.df.index.get_indexer(cells)
        
        dist_ids = []
        for i, cl_cell_id in enumerate(cl_cell_ids):
            start_id = np.where(cl_cell_ids > cl_cell_id,
                self.cell_start_id[cl_cell_id], self.cell_start_id[cl_cell_ids])
            new_ids = (start_id +  np.abs(cl_cell_ids - cl_cell_id) - 1)[i+1:]
            dist_ids.append(new_ids)
        return np.concatenate(dist_ids)


    def update_HCA(self, snp_weight, n_clusters=0, cells=[]):
        if len(cells) > 0:
            dist_in = self.dist_joint[self.get_dist_cell_ids(cells)]
        else:
            dist_in = self.dist_joint
        dist = np.average(dist_in, axis=1, weights=[snp_weight, (1 - snp_weight)])
        Z = linkage(dist, method='ward')
        order = leaves_list(Z)
        if n_clusters:
            clusters = cut_tree(Z, n_clusters=n_clusters).flatten()
            return order, clusters
        else:
            return order


    def split_cluster(self, cl_id, snp_weight):
        cl_cells = self.cells[self.cells['cluster'] == cl_id].index.values
        order, clusters = self.update_HCA(snp_weight, 2, cl_cells)

        new_cl_id = self.get_cluster_number()
        clusters = np.where(clusters, cl_id, new_cl_id)

        self.cells.loc[cl_cells] = self.cells.loc[self.SNPs.df.index[order]]
        self.cells.loc[cl_cells, 'cluster'] = clusters[order]


    def update_cluster_number(self, n_clusters, snp_weight):
        order, clusters = self.update_HCA(snp_weight, n_clusters)

        self.cells = self.cells.loc[self.SNPs.df.index[order]]
        self.cells['cluster'] = clusters[order]


    def update_assignment(self, new_assignment, cl_clType_map, snp_weight):
        self.cells['assignment'] = self.cells['cluster'].map(new_assignment)
        clType_cl_map = {j: i for i, j in cl_clType_map.items()}
        assign_int_map = {i: clType_cl_map[j] for i, j in new_assignment.items()}

        self.cells['cluster'] = self.cells['cluster'].map(assign_int_map)

        new_order = []
        idx_min = 0
        for cl_type in cl_clType_map.values():
            cl_cells = self.cells[self.cells['assignment'] == cl_type].index.values
            order = self.update_HCA(snp_weight, cells=cl_cells)
            new_order.extend(cl_cells[order])
            idx_min += cl_cells.size

        self.cells = self.cells.loc[new_order]

        healthy_cells = self.cells[self.cells['assignment'] == 'healthy']\
            .index.values
        self.reads.normalize_to_cluster(healthy_cells)


    def get_figure(self):
        fig = make_subplots(rows=5,
            cols=3,
            row_heights=[0.43, 0.03, 0.43, 0.03, 0.03],
            vertical_spacing=0.00,
            column_widths=[0.9, 0.05, 0.05],
            horizontal_spacing=0.00,
            shared_yaxes='rows',
            subplot_titles=('', 'Seq. depth', 'Clusters', '', '', '',
                '', '', '', '', '', ''),
            specs=[
                [{'r': 0.01},{'r': 0.01},{'r': 0.01}],
                [{'r': 0.01, 'b':0.01},{'r': 0.01},{}], 
                [{'r': 0.01},{'r': 0.01},{'r': 0.01}],
                [{'r': 0.01},{'r': 0.01},{}],
                [{'r': 0.01},{'r': 0.01},{}]
            ]
        )
        cell_order = self.get_cell_order()

        hm_lib_size = self.depth.get_heatmap(cell_order)
        hm_clusters = self.get_cluster_hm(cell_order)

        # First row
        hm_SNPs = self.SNPs.get_heatmap(cell_order)
        fig.append_trace(hm_SNPs, row=1, col=1)
        fig.append_trace(hm_lib_size, row=1, col=2)
        fig.append_trace(hm_clusters, row=1, col=3)
        # Second row
        SNP_ampl = self.SNPs.get_amplicons().values
        hm_SNP_genes = self.panel.get_heatmap('Gene', SNP_ampl)
        fig.append_trace(hm_SNP_genes, row=2, col=1)
        # Third row
        hm_reads = self.reads.get_heatmap(cell_order)
        fig.append_trace(hm_reads, row=3, col=1)
        fig.append_trace(hm_lib_size, row=3, col=2)
        fig.append_trace(hm_clusters, row=3, col=3)
        # Fourth row
        hm_reads_genes = self.panel.get_heatmap('Gene', self.reads.amplicons_good)
        fig.append_trace(hm_reads_genes, row=4, col=1)
        # Fifth row
        hm_reads_chrom = self.panel.get_heatmap('CHR', self.reads.amplicons_good)
        fig.append_trace(hm_reads_chrom, row=5, col=1)

        fig.update_yaxes(title_text='Cells', row=1, col=1)
        fig.update_yaxes(title_text='Gene', row=2, col=1)
        fig.update_yaxes(title_text='Cells', row=3, col=1)
        fig.update_yaxes(title_text='Gene', row=4, col=1)
        fig.update_yaxes(title_text='Chr', row=5, col=1)

        # Turn of x and y tick labels
        for fig_l in fig['layout']:
            if fig_l.startswith('yaxis') or fig_l.startswith('xaxis'):
                fig['layout'][fig_l].showticklabels = False
       
        return fig


    def get_cluster_hm(self, order):
        hm = go.Heatmap(
            z=self.cells[['cluster']],
            x=['Clusters'],
            y=order,
            colorscale=self.get_cluster_colors(),
            showscale=False
        )
        return hm


    def get_cluster_colors(self):
        colors = deepcopy(CLUSTER_COLORS_raw)
        val_range = np.linspace(0, 1, self.get_cluster_number())
        return [(i, next(colors)) for i in val_range]

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

class Data:
    def __init__(self, in_file, cell_order=[]):
        self.in_file = in_file
        self.df = self.load_data(cell_order)
        self.dist = self.get_pairwise_dists()
        self.Z = self.get_Z()


    def load_data(self, cell_order=[]):
        pass


    def get_pairwise_dists(self, rel_cells=[]):
        pass


    def get_heatmap(self, order):
        pass


    def get_Z(self):
        return linkage(self.dist, method='ward')


    def get_clusters(self, n_clusters):
        order = leaves_list(self.Z)
        clusters = cut_tree(self.Z, n_clusters=n_clusters).flatten()
        return self.df.index[order], clusters[order]

# ------------------------------------------------------------------------------


class DepthData(Data):
    def __init__(self, in_file, cell_order=[]):
        super().__init__(in_file, cell_order)
       

    def load_data(self, cell_order=[]):
        df_in = pd.read_csv(self.in_file, sep='\t', header=0, index_col=0)
        if len(cell_order) > 0:
            df_in = df_in.loc[cell_order]
        # Get library depth per cell
        df = np.log10(df_in.sum(axis=1)).to_frame(name='Library\nsize [log10]')
        return df


    def get_pairwise_dists(self, rel_cells=[]):
        # if len(rel_cells) == 0:
        #     rel_cells = self.df.index.values

        # df = self.df.loc[rel_cells].values
        # dist = []

        # for i in np.arange(df.shape[0] - 1):
        #     # Euclidean distance
        #     dist.append(np.sqrt(np.sum((df[i] - df[i+1:])**2, axis=1)))
        # return np.concatenate(dist)
        pass


    def get_Z(self):
        pass


    def get_heatmap(self, order):
        hm = go.Heatmap(
            z=self.df.loc[order],
            y=order,
            colorscale=LIB_COLORS,
            showscale=False
        )
        return hm

# ------------------------------------------------------------------------------


class ReadData(Data):
    def __init__(self, in_file, cell_order=[]):
        super().__init__(in_file, cell_order)
    

    def __str__(self):
        out_str = f'\nRead file: {self.in_file}:\n' \
            'Amplicons:\n' \
            f'\tTotal: {self.df_in.shape[1]}\n' \
            f'\t\tNoisy:\t\t\t{self.amplicons_noisy.sum()}\n' \
            f'\t\tLow coverage:\t{self.amplicons_low_cov.sum()}\n' \
            f'\tAfter filtering: {self.df.shape[1]}\n'
        return out_str


    @staticmethod
    def calc_gini(x):
        total = 0
        for i, xi in enumerate(x[:-1], 1):
            total += np.sum(np.abs(xi - x[i:]))
        return total / (len(x)**2 * np.mean(x))


    def load_data(self, cell_order=[]):
        self.df_in = pd.read_csv(self.in_file, sep='\t', header=0, index_col=0)
        if len(cell_order) > 0:
            self.df_in = self.df_in.loc[cell_order]

        # Noisy amplicons (tapestri)
        gini = self.df_in.apply(lambda i: self.calc_gini(i.values))
        self.amplicons_noisy = gini > gini.mean() + 2 * gini.std()

        dp_mean = self.df_in.mean().mean()
        # low performing amplicons (tapestri)
        self.amplicons_low_cov = self.df_in.mean() < 0.2 * dp_mean
        # High expression amplicons (tapestri)
        self.amplicons_high_cov = self.df_in.mean() > 2 * dp_mean

        # Get library depth per cell
        self.lib_depth = np.log10(self.df_in.sum(axis=1)) \
            .to_frame(name='Library\nsize [log10]')
        # Normalize per cell
        df = self.df_in.apply(lambda x: (x / x.sum()), axis=1)

        # Filter bad amplicons
        self.amplicons_good = ~(self.amplicons_noisy | self.amplicons_low_cov)

        df = df.loc[:,self.amplicons_good]
        self.df_cpm = (df * 1e6).round().astype(int) + 1
        self.norm_const = np.arange(0, self.df_cpm.max().max() * 2) * np.log(2)

        # Remove outliers: clip data to 10% and 90% quantile
        df.clip(lower=df.quantile(0.1), upper=df.quantile(0.9), axis=1,
            inplace=True)
        # Normalize per amplicon
        df = df / df.mean()

        # Normalize such that avg. cell depth = 2
        df = df.apply(lambda x: x / x.mean() * 2, axis=0)

        return df

    
    def get_pairwise_dists(self, rel_cells=[]):
        if len(rel_cells) == 0:
            dp = self.df_cpm.values
        else:
            dp = self.df_cpm.loc[rel_cells].values

        dp_fct = dp * np.log(dp)

        dist = []
        for i in np.arange(dp.shape[0] - 1):
            valid = (dp[i] > 1) & (dp[i+1:] > 1)
            dp_tot = dp[i] + dp[i+1:]
            l12 = dp_tot / 2
            logl = dp_fct[i] + dp_fct[i+1:] \
                - (dp_tot) * np.log(l12) \
                + 2 * l12 - dp_tot
            norm = self.norm_const[dp_tot]
            dist.append(np.sum(np.where(valid, logl / norm, 0), axis=1) \
                / valid.sum(axis=1))
        return np.concatenate(dist)


    # Normalize such that avg. healthy cells depth = 2
    def normalize_to_cluster(self, healthy_cells):
        self.df = self.df.apply(lambda x: x / x.loc[healthy_cells].mean() * 2, axis=0)


    def get_heatmap(self, order):
        hm = go.Heatmap(
            z=np.clip(self.df.loc[order], 0, 6),
            zmin=0,
            zmax=6,
            x=self.df.columns,
            y=order,
            colorscale=CNV_COLORS,
            colorbar={
                'title': 'Copy Number',
                'titleside': 'top',
                'tickmode': 'array',
                'tickvals': [0, 1, 2, 3, 4, 5, 6],
                'ticks': 'outside',
                'len': 0.4,
                'yanchor': 'top',
                'y': 0.55,
                'ypad': 0
            }
        )
        return hm


# ------------------------------------------------------------------------------

class SNPData(Data):
    def __init__(self, in_file, cell_order=[]): 
        super().__init__(in_file, cell_order)

        
    def load_data(self, cell_order=[]):
        self.df_in = pd.read_csv(self.in_file, dtype={'CHR': str}).T

        df = self.df_in.iloc[7:]

        self.SNPs = self.df_in.loc[['CHR', 'POS']].T
        self.SNPs['full'] = self.df_in.loc['CHR'].str.replace('chr', '') + ':' \
            + self.df_in.loc['POS'].astype(str) + ' ' + self.df_in.loc['REF'] \
             + '>' + self.df_in.loc['ALT']
        self.SNPs.set_index(['CHR', 'POS'], inplace=True)
        df.columns = self.SNPs['full']
        # row 4 = 'REGION' in input file
        self.SNP_ampl_map = {j: self.df_in.iloc[4, i] \
            for i, j in enumerate(self.SNPs['full'])}

         # Init relevant data
        self.ref = df.map(lambda x: int(x.split(':')[0]))
        self.alt = df.map(lambda x: int(x.split(':')[1]))
        self.dp = self.ref + self.alt
        self.VAF = np.clip((self.alt + EPSILON) / (self.dp + EPSILON),
            EPSILON, 1 - EPSILON)
        self.RAF = 1 - self.VAF
        self.norm_const = np.insert(
            np.arange(1, self.dp.max().max() * 2 + 1) \
                * np.log(np.arange(1, self.dp.max().max() * 2 + 1)),
            0, np.nan)
    
        # Filter SNPs that are irrelevant for clustering
        self.rel_SNPs = self.get_relevant_SNPS()
        return df


    def get_relevant_SNPS(self):
        # Identify SNPs that are symmetric/normal distributed: likely germline + ADO
        # Not informative for clustering
        VAF_z = (self.VAF - self.VAF.mean(axis=0)).fillna(0)
        p_vals_sym = VAF_z.apply(lambda x: kstest(x, -1 * x).pvalue, axis=0)
        rel_SNPs = p_vals_sym * self.SNPs.shape[0] < 0.05

        # Identify SNPs that are on the same read in most/all cells and remove 
        #   one of them from clustering
        for chrom, chrom_SNPs in self.SNPs.iloc[rel_SNPs.values].groupby('CHR'):
            if chrom_SNPs.size < 2:
                continue
            pos = chrom_SNPs.index.get_level_values('POS').values
            # SNPs at position id[x] and [x - 1] are on the same read
            same_read_SNP = np.argwhere((pos[1:] - pos[:-1]) < 275).ravel() + 1
            if same_read_SNP.size == 0:
                continue

            for SNP2_chrom_id in same_read_SNP:
                # Get row index in full data
                SNP1 = self.SNPs.loc[chrom_SNPs.index[SNP2_chrom_id - 1]]['full']
                SNP2 = self.SNPs.loc[chrom_SNPs.index[SNP2_chrom_id]]['full']
                
                # Calculate euclidean distance between VAF profiles
                valid = ~np.isnan(self.VAF[SNP1]) & ~np.isnan(self.VAF[SNP2])
                VAF_dist = euclidean(self.VAF[SNP1][valid], self.VAF[SNP2][valid]) \
                    / np.sqrt(np.sum(2**2 * valid.sum()))
                # If VAF profiles are very similar: remove second SNP
                if VAF_dist < 0.05:
                    rel_SNPs.loc[SNP2] = False

        return rel_SNPs


    def get_pairwise_dists(self, rel_cells=[]):
        if len(rel_cells) == 0:
            rel_cells = self.df.index.values

        dp = self.dp.loc[rel_cells, self.rel_SNPs].values
        alt = self.alt.loc[rel_cells, self.rel_SNPs].values
        ref = self.ref.loc[rel_cells, self.rel_SNPs].values
        VAF = self.VAF.loc[rel_cells, self.rel_SNPs].values
        RAF = self.RAF.loc[rel_cells, self.rel_SNPs].values

        dist = []
        for i in np.arange(rel_cells.size - 1):
            valid = (dp[i] > 0) & (dp[i+1:] > 0)
            dp_total = dp[i] + dp[i+1:]
            p12 = np.clip((alt[i] + alt[i+1:] + EPSILON) / (dp_total + EPSILON),
                EPSILON, 1 - EPSILON)
            p12_inv = 1 - p12
            logl = alt[i] * np.log(VAF[i] / p12) \
                + ref[i] * np.log(RAF[i] / p12_inv) \
                + alt[i+1:] * np.log(VAF[i+1:] / p12) \
                + ref[i+1:] * np.log(RAF[i+1:] / p12_inv)

            norm = self.norm_const[dp_total] \
                - self.norm_const[dp[i]] \
                - self.norm_const[dp[i+1:]]

            dist.append(np.nansum(logl / norm, axis=1) / valid.sum(axis=1))
        return np.concatenate(dist)


    def get_amplicons(self):
        return self.SNPs['full'].map(self.SNP_ampl_map)


    def get_heatmap(self, order):
        ampl = self.get_amplicons()

        hm = go.Heatmap(
            z=self.VAF.loc[order],
            zmin=0,
            zmax=1,
            x=self.SNPs['full'] + '<br>' + ampl,
            y=order,
            colorscale=SNP_COLORS,
            colorbar={ 
                'title': 'VAF',
                'titleside': 'top',
                'tickmode': 'array',
                'tickvals': [0, 0.5, 1],
                'ticks': 'outside',
                'len': 0.4,
                'yanchor': 'top',
                'y': 1,
                'ypad': 0
            }
        )
        return hm


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

class Panel:
    def __init__(self, in_file):
        print(f'Loading panel from: {in_file}')
        self.df = self.load_panel(in_file)
        

    @staticmethod
    def load_panel(in_file):
        df = pd.read_csv(in_file, comment='#', sep='\t', header=None, 
            index_col=14, names=PANEL_COLS)
        df['CHR'] = df['CHR'].str.replace('chr', '')
        return df


    def get_amplicon_idx(self):
        ampl_order = self.df.rename(df['locus'].to_dict(), axis=1).columns.values
        return sorted(range(self.df.shape[1]),
            key=lambda i:(CHR_ORDER[ampl_order[i].split(':')[0]], i))
     

    def get_heatmap(self, col, good_ampl):
        if col == 'Gene':
            colors = GENE_COLORS
            zmax = GENE_MAX
        else:
            colors = CHR_COLORS
            zmax = 24

        int_map = {j: i for i,j in enumerate(self.df.loc[:, col].unique())}
        z = self.df.loc[good_ampl, col].map(int_map).to_frame().T
        text = self.df.loc[good_ampl, ['Gene', 'CHR']] \
            .apply(lambda x: f'{x["Gene"]}<br>chr{x["CHR"]}<br>{x.name}', axis=1) \
            .to_frame().T
        hm = go.Heatmap(
            z=z.values,
            zmin=0,
            zmax=zmax,
            hoverinfo='text',
            text=text.values,
            colorscale=colors,
            showscale=False
        )
        return hm


# ------------------------------------------------------------------------------        
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    # Default for programming
    read_file = 'data/G12958/G12958.barcode.cell.distribution.merged.tsv'
    snp_file = 'data/G12958/G12958.filtered_variants.csv'
    panel_file = 'data/4387_annotated.bed'
    panel = Panel(panel_file)

    data = TapestriDNA(panel)
    data.load_sample_data(read_file, snp_file)
    data.update_cluster_number(3)
    fig = data.get_figure()
    fig.show()