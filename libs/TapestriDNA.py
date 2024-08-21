#!/usr/bin/env python3

import abc
from copy import deepcopy
from itertools import cycle
import os
import warnings

import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy.cluster.hierarchy import linkage, cut_tree, leaves_list
from scipy.spatial.distance import euclidean


EPSILON = np.finfo(np.float64).resolution # pylint: disable=E1101
CHR_ORDER = dict({str(i): i for i in range(1, 23, 1)}, **{'X': 23, 'Y': 24})
PANEL_COLS = ['CHR', 'Start', 'End', 'Gene', 'Exon', 'Strand', 'Feature',
    'Biotype', 'Ensembl_ID', 'TSL', 'HUGO', 'Tx_overlap_%', 'Exon_overlaps_%',
    'CDS_overlaps_%']


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
    (0.250, '#ffffff'), # 2 - 3: White
    (0.417, '#ffffff'), # 2 - 3: White
    (0.500, '#ff0000'), # 3 - 4: Red
    (0.666, '#C60000'), # 4 - 5: Darker Red
    (0.833, '#b90000'), # 5 - 6: Dark Red
    (1.000, '#6B0000'), # 5.5 - 6: Even darker Red
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
REL_COLORS = [
    (0.0, '#AA3939'), # red
    (0.5, '#7B9F34'), # lighter green
    (1.0, '#2D882D') # green
]

# ------------------------------------------------------------------------------

class TapestriDNA:
    def __init__(self, panel_in, read_file, snp_file):
        self.panel = panel_in

        self.prefix = os.path.basename(read_file).split('.barcode')[0]
        self.out_dir, _ = os.path.split(read_file)
        print(f'\nLoading sample {self.prefix}')
        print(f'\tSNPs  file: {snp_file}')
        self.snps = SNPData(snp_file)
        self.snps.add_panel_info(self.panel)
        self.cells = pd.DataFrame(np.zeros((self.snps.df.shape[0], 2)),
            index=self.snps.df.index, columns=['cluster', 'assignment'],
            dtype=int)
        self.cells['assignment'] = self.cells['assignment'].astype(str)
        self.cells.index.name = 'barcode'

        cell_order = self.get_cell_order()
        self.reads = ReadData(read_file, cell_order)
        self.reads.add_panel_info(self.panel)
        self.depth = DepthData(read_file, cell_order)
        # Add amplicon data to snps
        self.snps.add_amplicon_data(self.reads.meta)

        if self.reads.df.shape[0] != self.snps.df.shape[0]:
            print('!Warning: Number of cells in read and SNP files do not match. '\
                    'Taking only cells present in SNP data')
        print(f'Loading sample {self.prefix} - done')
        # Init functions for efficiently calculating joint dist
        self.dist_joint = np.stack([self.snps.dist, self.reads.dist], axis=1)
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
        cl_cell_ids = self.snps.df.index.get_indexer(cells)

        dist_ids = []
        for i, cl_cell_id in enumerate(cl_cell_ids):
            start_id = np.where(cl_cell_ids > cl_cell_id,
                self.cell_start_id[cl_cell_id], self.cell_start_id[cl_cell_ids])
            new_ids = (start_id +  np.abs(cl_cell_ids - cl_cell_id) - 1)[i+1:]
            dist_ids.append(new_ids)
        return np.concatenate(dist_ids)


    def update_dist_join(self):
        self.dist_joint = np.stack([self.snps.dist, self.reads.dist], axis=1)


    def update_hca(self, snp_weight, n_clusters=0, cells=None):
        if cells is None:
            dist_in = self.dist_joint
        else:
            dist_in = self.dist_joint[self.get_dist_cell_ids(cells)]
        dist = np.average(dist_in, axis=1, weights=[snp_weight, (1 - snp_weight)])
        link_matrix = linkage(dist, method='ward')
        order = leaves_list(link_matrix)
        if n_clusters:
            clusters = cut_tree(link_matrix, n_clusters=n_clusters).flatten()
            return order, clusters
        return order


    def split_cluster(self, cl_id, snp_weight):
        cl_cells = self.cells[self.cells['cluster'] == cl_id].index.values
        order, clusters = self.update_hca(snp_weight, 2, cl_cells)

        new_cl_id = self.get_cluster_number()
        clusters = np.where(clusters, cl_id, new_cl_id)

        self.cells.loc[cl_cells] = self.cells.loc[self.snps.df.index[order]]
        self.cells.loc[cl_cells, 'cluster'] = clusters[order]


    def update_snp_rel(self, snp, new_val, snp_weight, n_clusters):
        self.snps.set_snp_rel(snp, new_val)
        self.snps.update_pairwise_dists()
        self.update_dist_join()
        order, clusters = self.update_hca(snp_weight, n_clusters)

        self.cells = self.cells.loc[self.snps.df.index[order]]
        self.cells['cluster'] = clusters[order]


    def update_clustering(self, n_clusters, snp_weight):
        order, clusters = self.update_hca(snp_weight, n_clusters)

        self.cells = self.cells.loc[self.snps.df.index[order]]
        self.cells['cluster'] = clusters[order]
        not_set = self.cells['assignment'].astype(str).str.len() < 3
        self.cells.loc[not_set, 'assignment'] = self.cells.loc[not_set, 'cluster']
        self.cells.loc[~not_set, 'assignment'] = self.cells.loc[~not_set, 'assignment']


    def update_assignment(self, new_assignment, cl_type_map, snp_weight):
        self.cells['assignment'] = self.cells['cluster'].map(new_assignment)
        type_cl_map = {j: i for i, j in cl_type_map.items()}
        assign_int_map = {i: type_cl_map[j] for i, j in new_assignment.items()}

        self.cells['cluster'] = self.cells['cluster'].map(assign_int_map)

        new_order = []
        idx_min = 0
        for cl_type in cl_type_map.values():
            cl_cells = self.cells[self.cells['assignment'] == cl_type].index.values
            order = self.update_hca(snp_weight, cells=cl_cells)
            new_order.extend(cl_cells[order])
            idx_min += cl_cells.size

        self.cells = self.cells.loc[new_order]

        healthy_cells = self.cells[self.cells['assignment'] == 'healthy'] \
            .index.values
        self.reads.normalize_to_cluster(healthy_cells)


    def get_figure(self, show_all=True):
        fig_new = make_subplots(
            rows=6,
            cols=3,
            row_heights=[0.02, 0.42, 0.04, 0.02, 0.42, 0.02],
            vertical_spacing=0.00,
            column_widths=[0.95, 0.025, 0.025],
            horizontal_spacing=0.00,
            shared_yaxes='rows',
            subplot_titles=('', 'Seq.<br>depth', 'Clusters', '', '', '',
                '', '', '', '', '', '', ''),
            specs=[
                [{'r': 0.01},{'r': 0.01},{}],
                [{'r': 0.01},{'r': 0.01},{'r': 0.01}],
                [{'r': 0.01, 'b':0.02},{'r': 0.01},{}],
                # -------------------------------------
                [{'r': 0.01},{'r': 0.01},{}],
                [{'r': 0.01},{'r': 0.01},{'r': 0.01}],
                [{'r': 0.01},{'r': 0.01},{}],
            ]
        )
        cell_order = self.get_cell_order()

        hm_lib_size = self.depth.get_heatmap(cell_order)
        hm_clusters = self.get_cluster_hm(cell_order)

        # First row
        row = 1
        hm_snps_relevant = self.snps.get_relevant_snp_heatmap(show_all)
        fig_new.append_trace(hm_snps_relevant, row=row, col=1)
        fig_new.update_yaxes(title_text='relevant', row=row, col=1)
        # Second row
        row = 2
        hm_snps = self.snps.get_heatmap(cell_order, show_all)
        fig_new.append_trace(hm_snps, row=row, col=1)
        fig_new.append_trace(hm_lib_size, row=row, col=2)
        fig_new.append_trace(hm_clusters, row=row, col=3)
        fig_new.update_yaxes(title_text='Cells', row=2, col=1)
        # Third row
        row = 3
        snp_ampl = self.snps.get_amplicons(show_all)
        hm_snps_genes = self.panel.get_heatmap('Gene', snp_ampl)
        fig_new.append_trace(hm_snps_genes, row=row, col=1)
        fig_new.update_yaxes(title_text='Gene', tickangle=90, row=row, col=1)

        self.add_chr_vlines(fig_new, snp_ampl, row)
        # ----------------------------------------------------------------------
        # Fourth row
        row = 4
        hm_reads_relevant = self.reads.get_relevant_reads_heatmap(show_all)
        fig_new.append_trace(hm_reads_relevant, row=row, col=1)
        fig_new.update_yaxes(title_text='relevant', row=row, col=1)
        # Fifths row
        row = 5
        hm_reads = self.reads.get_heatmap(cell_order, show_all)
        fig_new.append_trace(hm_reads, row=row, col=1)
        fig_new.append_trace(hm_lib_size, row=row, col=2)
        fig_new.append_trace(hm_clusters, row=row, col=3)
        fig_new.update_yaxes(title_text='Cells', row=row, col=1)

        # Sixths row
        if show_all:
            read_ampl = self.reads.meta.index.values
        else:
            read_ampl = self.reads.meta[self.reads.meta['is_rel']].index.values

        row = 6
        hm_reads_genes = self.panel.get_heatmap('Gene', read_ampl)
        fig_new.append_trace(hm_reads_genes, row=row, col=1)
        fig_new.update_yaxes(title_text='Gene', row=row, col=1)

        self.add_chr_vlines(fig_new, read_ampl, row)

        # Turn of x and y tick labels
        for fig_new_l in fig_new['layout']:
            if fig_new_l.startswith('yaxis') or fig_new_l.startswith('xaxis'):
                fig_new['layout'][fig_new_l].showticklabels = False

        return fig_new


    def add_chr_vlines(self, fig, ampl, row_no, last_n_rows=3):
        chr_str = []
        ampl_chr = []
        ampl_p_arm = []
        for chrom, chr_ampl in self.panel.df.loc[ampl].groupby('CHR', sort=False):
            chr_str.append(chrom)
            ampl_chr.append(chr_ampl.shape[0])
            ampl_p_arm.append((chr_ampl['chr_arm'] == 'p').sum())

        # Dont show last vline; subtract 0.5 to be on line with heatmap
        chr_x = np.cumsum(ampl_chr[:-1]) - 0.5
        for i, x in enumerate(chr_x):
            # Add to last 3 plots
            for j in range(last_n_rows):
                # Add Chr vertical line
                fig.add_vline(x, row=row_no - j, col=1, line_color='black',
                    line_width=2)
                # Add Chr text annotation
                if j == last_n_rows - 1:
                    if i == 0:
                        x_annot = ampl_chr[i]/ 2 - 0.5
                    else:
                        x_annot = chr_x[i-1] + ampl_chr[i] / 2
                    fig.add_annotation(
                        x=x_annot, y=0, yshift=15, 
                        text=chr_str[i], showarrow=False,
                        row=row_no - j, col=1)

                # Chr Arm lines
                if ampl_p_arm[i] > 0 and ampl_p_arm[i] < ampl_chr[i]:
                    if i == 0:
                        x_arm = ampl_p_arm[i]
                    else:
                        x_arm = chr_x[i - 1] + ampl_p_arm[i]
                    fig.add_vline(x_arm, row=row_no - j, col=1,
                        line_color='#c3c4c3', line_width=2)
        # Add last chrom annotation
        fig.add_annotation(
            x=chr_x[-1] + ampl_chr[-1] / 2,
            y=0, yshift=15, 
            text=chr_str[-1], showarrow=False,
            row=row_no - (last_n_rows - 1), col=1)


    def get_cluster_hm(self, order):
        text = self.cells.apply(lambda x:
            f'cluster: {x.assignment}<br>barcode: {x.name}', axis=1).to_frame()

        hm = go.Heatmap(
            z=self.cells[['cluster']],
            x=['cluster'],
            y=order,
            hoverinfo='text',
            text=text,
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

class Data(metaclass=abc.ABCMeta):
    def __init__(self, in_file, cell_order=None):
        if cell_order is None:
            cell_order = []
        self._in_file = in_file
        self.meta = None
        self.df = self.load_data(cell_order)
        self.dist = self.get_pairwise_dists()


    @abc.abstractmethod
    def load_data(self, cell_order=None):
        pass


    @abc.abstractmethod
    def get_pairwise_dists(self, rel_cells=None):
        pass


    @abc.abstractmethod
    def get_heatmap(self, order, show_all=True):
        pass


    def update_pairwise_dists(self, rel_cells=None):
        self.dist = self.get_pairwise_dists(rel_cells)


    def get_linkage(self):
        return linkage(self.dist, method='ward')


# ------------------------------------------------------------------------------


class DepthData(Data):
    def __init__(self, in_file, cell_order=None):
        if cell_order is None:
            cell_order = []
        super().__init__(in_file, cell_order)
        self.text = self.df.apply(lambda x:
            f'Library size [log10]: {x.lib_size:.2f} (abs: {x.lib_size**10:.2e})'\
                f'<br>barcode: {x.name}',
            axis=1
        ).to_frame()


    def load_data(self, cell_order=None):
        df_in = pd.read_csv(self._in_file, sep='\t', header=0, index_col=0)
        if cell_order is not None:
            df_in = df_in.loc[cell_order]
        # Get library depth per cell
        df = np.log10(df_in.sum(axis=1)).to_frame(name='lib_size')
        return df


    def get_pairwise_dists(self, rel_cells=None):
        pass
        # if rel_cells is None:
        #     rel_cells = self.df.index.values

        # df = self.df.loc[rel_cells].values
        # dist = []

        # for i in np.arange(df.shape[0] - 1):
        #     # Euclidean distance
        #     dist.append(np.sqrt(np.sum((df[i] - df[i+1:])**2, axis=1)))
        # return np.concatenate(dist)


    def update_pairwise_dists(self, rel_cells=None):
        raise NotImplementedError


    def get_linkage(self):
        raise NotImplementedError


    def get_heatmap(self, order, show_all=True):
        hm = go.Heatmap(
            z=self.df.loc[order],
            x=['Library size'],
            y=order,
            hoverinfo='text',
            text=self.text.loc[order],
            colorscale=LIB_COLORS,
            showscale=False
        )
        return hm

# ------------------------------------------------------------------------------


class ReadData(Data):
    def __init__(self, in_file, cell_order=None):
        if cell_order is None:
            cell_order = []
        super().__init__(in_file, cell_order)


    def __str__(self):
        out_str = f'\nRead file: {self._in_file}:\n' \
            'Amplicons:\n' \
            f'\tTotal: {self.meta.shape[0]}\n' \
            f'\t\tNoisy:\t\t\t{self.meta["reason"].str.contains("noisy").sum()}\n' \
            f'\t\tLow coverage:\t' \
            f'{self.meta["reason"].str.contains("low coverage").sum()}\n' \
            f'\tAfter filtering: {self.meta["is_rel"].sum()}\n'
        return out_str


    @staticmethod
    def calc_gini(x):
        total = 0
        for i, xi in enumerate(x[:-1], 1):
            total += np.sum(np.abs(xi - x[i:]))
        return total / (len(x)**2 * np.mean(x))


    def load_data(self, cell_order=None):
        df_in = pd.read_csv(self._in_file, sep='\t', header=0, index_col=0)
        if cell_order is not None:
            df_in = df_in.loc[cell_order]

        self.meta = pd.DataFrame([], index=df_in.columns)
        self.meta['is_rel'] = True
        self.meta['reason'] = ''

        # Noisy amplicons (tapestri)
        gini = df_in.apply(lambda i: self.calc_gini(i.values))
        noisy = gini > gini.mean() + 2 * gini.std()
        self.meta.loc[noisy, 'is_rel'] = ~noisy
        self.meta.loc[noisy, 'reason'] += 'noisy;'

        dp_mean = df_in.mean().mean()
        # low performing amplicons (tapestri)
        low_cov = df_in.mean() < 0.2 * dp_mean
        self.meta.loc[low_cov, 'is_rel'] = ~low_cov
        self.meta.loc[low_cov, 'reason'] += 'low coverage;'

        # High expression amplicons (tapestri)
        high_cov = df_in.mean() > 2 * dp_mean
        self.meta.loc[high_cov, 'reason'] += 'high coverage;'

        # Get library depth per cell
        self.lib_depth = np.log10(df_in.sum(axis=1)) \
            .to_frame(name='Library\nsize [log10]')
        # Normalize counts per cell
        df = df_in.apply(lambda x: x / x[self.meta['is_rel']].sum(), axis=1)

        # Remove outliers per amplicon: clip data to 90% quartile
        df.clip(lower=None, upper=df.quantile(0.9), axis=1, inplace=True)

        self.df_cpm = (df * 1e6).round().astype(int) + 1
        self.norm_const = np.arange(0, self.df_cpm.max().max() * 2 + 1) \
            * np.log(2)

        # Normalize per amplicon such that avg. cell depth = 2
        df = df.apply(lambda x: x / x.mean() * 2, axis=0)

        return df


    def add_panel_info(self, panel_in):
        self.meta = pd.merge(self.meta,
                panel_in.df[['Gene', 'CHR', 'chr_arm', 'ampl_per_gene']],
            left_index=True, right_index=True)
        # change order to be same as panel (assumes panel to be sorted)
        self.meta = self.meta.loc[panel_in.df.index]
        # Generate display text
        self.meta['text'] = self.meta.apply(lambda x:
            f'{x.name}<br>Gene: {x.Gene} ({x.ampl_per_gene} ampl.)<br>' \
                f'Chrom. Arm: {x.CHR}{x.chr_arm}',
            axis=1)


    def get_pairwise_dists(self, rel_cells=None):
        if rel_cells is None:
            dp = self.df_cpm.loc[:, self.meta['is_rel']].values
        else:
            dp = self.df_cpm.loc[rel_cells, self.meta['is_rel']].values

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


    # Normalize per amplicon such that avg. healthy cells depth = 2
    def normalize_to_cluster(self, healthy_cells):
        self.df = self.df.apply(lambda x: x / x.loc[healthy_cells].mean() * 2,
            axis=0)


    def get_heatmap(self, order, show_all=True):
        if show_all:
            z = np.clip(self.df.loc[order], 0, 6)
            x = self.meta['text']
        else:
            z = np.clip(self.df.loc[order,self.meta['is_rel']], 0, 6)
            x = self.meta[self.meta['is_rel']]['text']

        hm = go.Heatmap(
            z=z,
            zmin=0,
            zmax=6,
            x=x,
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


    def get_relevant_reads_heatmap(self, show_all=True):
        text = self.meta['text'] + '<br><br>reason: ' \
                + self.meta['reason'].str.rstrip(';').values

        rel = self.meta.loc[:, ['is_rel']].T.astype(float)
        rel.loc[:,self.meta['reason'].str.contains('high coverage')] = 0.5

        if show_all:
            z = rel
            x = text
        else:
            z = rel.loc[:,self.meta['is_rel']]
            x = text.loc[self.meta['is_rel']]

        hm = go.Heatmap(
            z=z, # relevance
            x=x, # SNPs
            y=['amplicon'],
            zmin=0,
            zmax=1,
            colorscale=REL_COLORS,
            showscale=False
        )
        return hm


# ------------------------------------------------------------------------------

class SNPData(Data):
    def __init__(self, in_file, cell_order=None):
        if cell_order is None:
            cell_order = []
        super().__init__(in_file, cell_order)


    def load_data(self, cell_order=None):
        df_in = pd.read_csv(self._in_file, dtype={'CHR': str}).T

        df = df_in.iloc[7:]

        self.meta = df_in.loc[['CHR', 'POS']].T
        self.meta['full'] = df_in.loc['CHR'].str.replace('chr', '') + ':' \
            + df_in.loc['POS'].astype(str) + ' ' + df_in.loc['REF'] \
            + '>' + df_in.loc['ALT']
        self.meta.set_index(['full'], inplace=True)
        df.columns = self.meta.index
        # row 4|'NAME' in input file = amplicon
        self.meta['amplicon'] = df_in.loc['NAME'].values
         # Init relevant data
        self.ref = df.map(lambda x: int(x.split(':')[0]))
        self.alt = df.map(lambda x: int(x.split(':')[1]))
        self.dp = self.ref + self.alt
        self.vaf = np.clip((self.alt + EPSILON) / (self.dp + EPSILON),
            EPSILON, 1 - EPSILON)
        # Set uncovered loci to nans
        self.vaf.mask(self.dp == 0, inplace=True)
        self.raf = 1 - self.vaf
        self.norm_const = np.insert(
            np.arange(1, self.dp.max().max() * 2 + 1) \
                * np.log(np.arange(1, self.dp.max().max() * 2 + 1)),
            0, np.nan)

        # Filter SNPs that are irrelevant for clustering
        self.init_relevant_snps()
        self._update_meta_text()

        return df



    def _update_meta_text(self):
        text = '{name}<br>Amplicon: {amplicon}'
        if 'reason_ampl' in self.meta.columns:
            text += ' ({reason_ampl})'
        if 'Gene' in self.meta.columns:
            text += '<br>Gene: {Gene} ({ampl_per_gene} ampl.)'
        if 'chr_arm' in self.meta.columns:
            text += '<br>Chrom. Arm: {CHR}{chr_arm}'
        
        def fill_text(x, text):
            return text.format(**dict(x, **{'name': x.name}))

        self.meta['text'] = self.meta.apply(fill_text, args=(text,), axis=1) \
            .str.replace(' ()', '')


    def add_panel_info(self, panel_in):
        self.meta = self.meta.merge(
            panel_in.df[['Gene', 'chr_arm', 'ampl_per_gene']],
            left_on='amplicon', right_index=True, how='left')
        self._update_meta_text()


    def add_amplicon_data(self, ampl_meta):
        self.meta = self.meta.merge(
            ampl_meta['reason'], left_on='amplicon', right_index=True,
            how='inner', suffixes=('', '_ampl'))
        self._update_meta_text()
        

    def set_snp_rel(self, snp, new_val):
        self.meta.loc[snp, 'is_rel'] = new_val
        if 'manual' in self.meta.loc[snp, 'reason']:
            if new_val is False:
                self.meta.loc[snp, 'reason'] = self.meta.loc[snp, 'reason'] \
                    .replace('manual;', '')
        else:
            self.meta.loc[snp, 'reason'] += 'manual;'


    def init_relevant_snps(self):
        # Identify SNPs that are symmetric/normal distributed: likely germline + ADO
        # Not informative for clustering
        self.meta['is_rel'] = True
        self.meta['reason'] = ''

        self.meta['symmetry'] = 0
        for snp, snp_data in self.vaf.items():
            # Skip SNPs with just 1 value. E.g., all VAF == 1
            if snp_data.nunique() == 1:
                continue
            
            vaf_sorted = snp_data.dropna().sort_values().values
            self.meta.loc[snp, 'symmetry'] = np.corrcoef(
                    vaf_sorted, -1 * vaf_sorted[::-1])[0][1]
        
        symmetric = self.meta['symmetry'] > 0.99
        self.meta.loc[symmetric, 'is_rel'] = False
        self.meta.loc[symmetric, 'reason'] += 'symmetry;'

        # Identify SNPs that are on the same read in most/all cells and remove
        #   one of them from clustering
        for _, chrom_snps in self.meta.groupby('CHR'):
            if chrom_snps.shape[0] < 2:
                continue
            pos = chrom_snps['POS'].values
            # SNPs at position id[x] and [x - 1] are on the same read
            same_read_snps = np.argwhere((pos[1:] - pos[:-1]) < 275).ravel() + 1
            if same_read_snps.size == 0:
                continue
            for snp2_chrom_id in same_read_snps:
                # Get row index in full data
                snp1 = chrom_snps.iloc[snp2_chrom_id - 1].name
                snp2 = chrom_snps.iloc[snp2_chrom_id].name

                # Calculate euclidean distance between VAF profiles
                valid = ~np.isnan(self.vaf[snp1]) & ~np.isnan(self.vaf[snp2])
                vaf_dist = euclidean(self.vaf[snp1][valid], self.vaf[snp2][valid]) \
                    / np.sqrt(np.sum(2**2 * valid.sum()))
                # If VAF profiles are very similar: remove second SNP
                if vaf_dist < 0.05:
                    self.meta.loc[snp2, 'is_rel'] = False
                    self.meta.loc[snp2, 'reason'] += 'sameRead;'

        hom = (self.vaf.mean() > 0.99) & (self.vaf.isna().mean() < 0.05)
        self.meta.loc[hom, 'is_rel'] = False
        self.meta.loc[hom, 'reason'] += 'homozygous;'

        wt = (self.vaf.mean() < 0.01) & (self.vaf.isna().mean() < 0.05)
        self.meta.loc[wt, 'is_rel'] = False
        self.meta.loc[wt, 'reason'] += 'wildtype;'


    def get_pairwise_dists(self, rel_cells=None):
        if rel_cells is None:
            rel_cells = self.df.index.values

        dp = self.dp.loc[rel_cells, self.meta['is_rel']].values
        alt = self.alt.loc[rel_cells, self.meta['is_rel']].values
        ref = self.ref.loc[rel_cells, self.meta['is_rel']].values
        vaf = self.vaf.loc[rel_cells, self.meta['is_rel']].values
        raf = self.raf.loc[rel_cells, self.meta['is_rel']].values

        dist = []
        for i in np.arange(rel_cells.size - 1):
            valid = (dp[i] > 0) & (dp[i+1:] > 0)
            dp_total = dp[i] + dp[i+1:]
            p12 = np.clip((alt[i] + alt[i+1:] + EPSILON) / (dp_total + EPSILON),
                EPSILON, 1 - EPSILON)
            p12_inv = 1 - p12
            logl = alt[i] * np.log(vaf[i] / p12) \
                + ref[i] * np.log(raf[i] / p12_inv) \
                + alt[i+1:] * np.log(vaf[i+1:] / p12) \
                + ref[i+1:] * np.log(raf[i+1:] / p12_inv)

            norm = self.norm_const[dp_total] \
                - self.norm_const[dp[i]] \
                - self.norm_const[dp[i+1:]]

            dist.append(np.nansum(logl / norm, axis=1) / valid.sum(axis=1))
        return np.concatenate(dist)


    def get_amplicons(self, show_all=True):
        if show_all:
            return self.meta['amplicon']
        return self.meta[self.meta['is_rel']]['amplicon']


    def get_heatmap(self, order, show_all=True):
        z = self.vaf.loc[order].round(2)
        text = 'VAF: ' + z.astype(str) + '   (ref|alt: ' \
            + self.ref.loc[order].astype(str) + '|' \
            + self.alt.loc[order].astype(str) + ')<br><br>' \
            + np.repeat([self.meta['text'].values], order.size, axis=0) \
            + '<br>cell=' + np.repeat([order], z.shape[1], axis=0).T 
                
        if not show_all:
            z = z.loc[:, self.meta['is_rel']]
            text = text.loc[:, self.meta['is_rel']]

        hm = go.Heatmap(
            z=z, # VAF
            zmin=0,
            zmax=1,
            text=text,
            hoverinfo='text',
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
        go.Layout(
            plot_bgcolor='#636161'
        )
        return hm


    def get_relevant_snp_heatmap(self, show_all=True):
        text = self.meta['text'] + '<br><br>symmetry (R<sup>2</sup>): ' \
                + self.meta['symmetry'].round(2).astype(str) \
                + '<br>filter reason: ' + self.meta['reason'].str.rstrip(';').values

        if show_all:
            z = self.meta.loc[:, ['is_rel']].T.astype(int)
            x = text
        else:
            z = self.meta[self.meta['is_rel']].loc[:, ['is_rel']].T.astype(int)
            x = text.loc[self.meta['is_rel']]
        hm = go.Heatmap(
            z=z, # relevance
            x=x, # SNPs
            y=['snp'],
            zmin=0,
            zmax=1,
            colorscale=REL_COLORS,
            showscale=False
        )
        return hm


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

class Panel:
    def __init__(self):
        self.df = None
        self._chr_arm = False


    def load(self, in_file):
        print(f'Loading panel from: {in_file}')
        self.df = self.load_panel(in_file)
        # Add amplicons per gene column
        ampl_per_gene = self.df.groupby('Gene').apply(lambda x: x.shape[0])
        self.df['ampl_per_gene'] =  self.df['Gene'].map(ampl_per_gene)
        self.df['text'] = self.df.apply(lambda x:
            f'{x.name}<br>Gene: {x.Gene} ({x.ampl_per_gene} ampl.)<br>',
            axis=1)


    @staticmethod
    def load_panel(in_file):
        with open(in_file, 'r', encoding='utf-8') as f:
            for l in f.readlines():
                if l[0] == '#':
                    continue
                col_number = l.count('\t')
                break
        col_names = PANEL_COLS[:col_number] + ['Amplicon']
        df = pd.read_csv(in_file, comment='#', sep='\t', header=None,
            index_col=-1, names=col_names)
        df['CHR'] = df['CHR'].str.replace('chr', '')

        return df


    def add_chr_arm(self, ga_file):
        print(f'\t Adding gene annotation to panel from: {ga_file}')
        ga = pd.read_csv(ga_file, sep='\t', index_col=0)
        self.df['chr_arm'] = self.df['Gene'].map(ga['arm'].to_dict())

        # Check loci before and after if chr arm is not set
        for idx in np.argwhere(self.df['chr_arm'].isna()).ravel():
            if (self.df['CHR'].iloc[idx - 1] == self.df['CHR'].iloc[idx + 1]) \
                    and (self.df.iloc[idx - 1, -1] \
                        == self.df.iloc[idx + 1, -1]):
                self.df.iloc[idx, -1] = self.df.iloc[idx + 1, -1]
            else:
                print('No chromosome arm annotation for amplicon: ' \
                    f'')

        self._chr_arm = True

        self.df['text'] = self.df['text'] + 'Chrom. Arm: ' \
            + self.df.apply(lambda x: f'{x.CHR}{x.chr_arm}', axis=1)


    def get_heatmap(self, col, ampl=None):
        if ampl is None:
            ampl = self.df.index

        if col == 'Gene':
            colors = GENE_COLORS
            zmax = GENE_MAX
        else:
            colors = CHR_COLORS
            zmax = 24

        # Map gene/chr to int values for coloring
        int_map = {j: i for i,j in enumerate(self.df.loc[:, col].unique())}
        z = self.df.loc[ampl, col].map(int_map)

        hm = go.Heatmap(
            z=np.expand_dims(z, axis=0),
            zmin=0,
            zmax=zmax,
            y=[col],
            hoverinfo='text',
            text=np.expand_dims(self.df.loc[ampl, 'text'], axis=0),
            colorscale=colors,
            showscale=False
        )
        return hm


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    # Default for programming
    READ_FILE = 'data/G12958/G12958.barcode.cell.distribution.merged.tsv'
    SNP_FILE = 'data/G12958/G12958.filtered_variants.csv'
    PANEL_FILE = 'data/4387_annotated.bed'
    panel = Panel()
    panel.load(PANEL_FILE)

    data = TapestriDNA(panel, READ_FILE, SNP_FILE)
    data.update_clustering(3, 0.75)
    fig = data.get_figure()
    fig.show()
