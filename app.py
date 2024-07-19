#!/usr/bin/env python3

import argparse
from multiprocessing import Pool, cpu_count
import os

from libs.TapestriDNA import TapestriDNA, Panel

import dash_bootstrap_components as dbc
from dash import Dash, html, dcc, callback, Output, Input, State, ctx
from dash.exceptions import PreventUpdate


DEF_CLUSTERS = 3
DEF_SNP_WEIGHT = 0.8
DEF_CLUSTER_TYPES = ['doublets', 'healthy']


# ------------------------------- HTML LAYOUT ----------------------------------


modals = html.Div([
    dbc.Modal([
        dcc.Input(id='modal-cluster-input', type='text',
            style={'display': 'none'}
        ),
        dbc.ModalHeader([
            html.Div(id='modal-cluster-color'),
            dbc.ModalTitle(id='modal-cluster-title')
        ]),
        dbc.ModalBody(
            [
            dcc.Dropdown(DEF_CLUSTER_TYPES + ['tumor 1'], id='modal-cluster-dropdown',
                style={'width': '120px'}),
            html.Button('Split cluster', id='modal-cluster-split', n_clicks=0)
            ],
            style={'display': 'flex', 'justify-content': 'space-around'}),
        dbc.ModalFooter(dbc.Button('Close', id='modal-cluster-close',
            className='ms-auto')),
        ],
        id='modal-cluster', is_open=False,
    ),
    dbc.Modal([
        dcc.Input(id='modal-SNP-input', type='text', style={'display': 'none'}
        ),
        dbc.ModalHeader([
            dbc.ModalTitle('Relevant SNP:')
        ]),
        dbc.ModalBody(
            [
            dcc.Dropdown(['True', 'False'], id='modal-SNP-dropdown',
                style={'width': '120px'}),
            ],
            style={'display': 'flex', 'justify-content': 'space-around'}),
        dbc.ModalFooter(dbc.Button('Close', id='modal-SNP-close',
            className='ms-auto')),
        ],
        id='modal-SNP', is_open=False,
    ),
])

html_layout = dbc.Container(
    [
        html.H1(children='Manual annotation', style={'textAlign':'center'}),
        html.Hr(),
        html.Div([
            html.H4('Sample: ', style={'display':'inline-block',
                'margin-right': 10}),
            dcc.Dropdown(#options=samples, value=def_sample,
                id='dropdown-sample', style={'width': '200px'}),
            html.Div([
                html.Button('Save annotation', id='button-safe', 
                    style={'margin-right': 10}),
                html.Div(id='div-output'),
            ], style={'display':'flex', 'align-items': 'center'}),
        ], style={'display': 'flex', 'align-items': 'baseline', 'gap': '20px'}),
        html.Div([
            html.Div([
                html.H4('# Clusters: ', style={'display':'inline-block',
                    'margin-right': 10}),
                dcc.Input(id='input-n-clusters', type='number', 
                    placeholder='No. Clusters', value=DEF_CLUSTERS, min=1, max=30,
                    step=1, style={'width': '50px'}),
                ]),
            html.Div([
                html.H4('# Tumor clones: ', style={'display':'inline-block',
                    'margin-right': 10}),
                dcc.Input(id='input-n-clones', type='number', 
                    placeholder='No. Clones', value=1, min=1, max=10,
                    step=1, style={'width': '50px'}),
                ]),
            html.Div([
                html.H4('weights: ', style={'display':'inline-block',
                'margin-right': 10, 'margin-left': 10}),
                html.Div('SNPs - ', style={'margin-right': 3}),
                dcc.Input(id='input-snp-weights', type='number',
                    value=DEF_SNP_WEIGHT, min=0, max=1, step=0.05,
                    style={'width': '60px'}),
                html.Div(f': {1 - DEF_SNP_WEIGHT:.2f} - reads', id='div-read-weight'),
            ], style={'display': 'flex', 'align-items': 'center'}),
        ], style={'display': 'flex', 'align-items': 'center', 'gap': '20px'}),
        html.Div([
            html.Div(id='div-assignment',
                style={'display': 'flex', 'flex-wrap': 'wrap'}),
            html.Button('Apply', id='button-apply', n_clicks=0),
        ]),
        html.Hr(),
        dcc.Graph(id='graph', style={'height': '100vh'}),
        html.Div(id='hidden-div', style={'display': 'none'}),
        modals
    ],
    style={'margin-left': 0, 'margin-right': 0, 'max-width': '100vw'},
)


# ----------------------------- HELPER FUNCTIONS -------------------------------


def get_datasets(in_dir):
    datasets = {}
    for file in os.listdir(in_dir):
        file_full = os.path.join(args.in_dir, file)
        prefix = file.split('.')[0]
        if file.endswith('variants.csv') and not 'relevant' in file:
            try:
                datasets[prefix]['SNPs'] = file_full
            except KeyError:
                datasets[prefix] = {'SNPs': file_full}
        elif file.endswith('barcode.cell.distribution.merged.tsv'):
            try:
                datasets[prefix]['reads'] = file_full
            except KeyError:
                datasets[prefix] = {'reads': file_full}
        else:
            print(f'! WARNING: unknown input file: {file}')
            continue

    return datasets


def load_sample(sample):
    if not sample in data:
        new_data = TapestriDNA(panel)
        new_data.load_sample(datasets[sample]['reads'], datasets[sample]['SNPs'])
        data[sample] = new_data


def get_annotation_elements(sample, options, assignment={}):
    el = []
    colors = data[sample].get_cluster_colors()
    for cl in range(data[sample].get_cluster_number()):
        new_el = get_new_assignment_dropdown_div(cl, colors[cl][1], options)
        if len(assignment) > 0:
            new_el.children[2].value = assignment[cl]
        el.append(new_el)
    return el


def get_new_assignment_dropdown_div(cl, color, options):
    new_el = html.Div([
        html.Div(style={'height': '15px', 'width': '30px', 
            'display': 'inline-block', 'background-color': color}),
        html.H4(f'{cl}: ', style={'display':'inline-block',
            'margin-left': 10, 'margin-right': 10, 'margin-top': 0, 
            'margin-bottom': 0, 'color': color}),
        dcc.Dropdown(options, value=None, id=f'dropdown-{cl}',
            style={'width': '120px'})
    ])
    return new_el


def get_assignment_dropdown(assignments):
    dd = []
    for div_outer in assignments:
        for div_inner in div_outer['props']['children']:
            if div_inner['type'] == 'Dropdown':
                dd.append(div_inner)
    return dd


def get_annotation_options(n_clones):
    return DEF_CLUSTER_TYPES + [f'tumor {i+1}' for i in range(n_clones)]

# ----------------------------- CALLBACKS --------------------------------------

@callback(
    Output('input-n-clusters', 'value'),
    Output('div-output', 'children'),
    Input('dropdown-sample', 'value')
)
def load_data(sample):
    if sample in data:
        n_clusters = data[sample].get_cluster_number()
    else:
        load_sample(sample)
        n_clusters = DEF_CLUSTERS
    out_file = data[sample].get_out_file()
    return n_clusters, out_file


@callback(
    Output('graph', 'figure', allow_duplicate=True),
    Output('div-assignment', 'children', allow_duplicate=True),
    Input('input-n-clusters', 'value'),
    State('input-n-clones', 'value'),
    State('input-snp-weights', 'value'),
    State('dropdown-sample', 'value'),
    prevent_initial_call=True
)
def update_cluster_number(n_clusters, n_clones, snp_weight, sample):
    # Dont trigger if cluster assignment applied
    if CL_UPDATE_FLAG['apply']:
        CL_UPDATE_FLAG['apply'] = False
        raise PreventUpdate
    # Dont trigger if cluster split
    if CL_UPDATE_FLAG['split']:
        CL_UPDATE_FLAG['split'] = False
        raise PreventUpdate
    data[sample].update_clustering(n_clusters, snp_weight)
    fig = data[sample].get_figure()
    cl_options = get_annotation_options(n_clones)
    annot_el = get_annotation_elements(sample, cl_options)
    return fig, annot_el


@callback(
    Output('div-assignment', 'children', allow_duplicate=True),
    Output('modal-cluster-dropdown', 'options', allow_duplicate=True),
    Input('input-n-clones', 'value'),
    State('div-assignment', 'children'),
    State('dropdown-sample', 'value'),
    prevent_initial_call=True
)
def update_clone_number(n_clones, assignments, sample):
    assign = {}
    for dd in get_assignment_dropdown(assignments):
        cl = int(dd['props']['id'].split('-')[1])
        assign[cl] = dd['props']['value']
    cl_options = get_annotation_options(n_clones)
    annot_el = get_annotation_elements(sample, cl_options, assignment=assign)
    return annot_el, cl_options


@callback(
    Output('graph', 'figure', allow_duplicate=True),
    Output('div-read-weight', 'children'),
    Input('input-snp-weights', 'value'),
    State('input-n-clusters', 'value'),
    State('dropdown-sample', 'value'),
    prevent_initial_call=True
)
def update_weight(snp_weight, n_clusters, sample):
    data[sample].update_clustering(n_clusters, snp_weight)
    fig = data[sample].get_figure()
    return fig, f': {1 - snp_weight:.2f} - reads'


@callback(
    Output('graph', 'figure', allow_duplicate=True),
    Output('div-assignment', 'children', allow_duplicate=True),
    Output('input-n-clusters', 'value', allow_duplicate=True),
    Input('button-apply', 'n_clicks'),
    State('input-n-clones', 'value'),
    State('input-snp-weights', 'value'),
    State('div-assignment', 'children'),
    State('dropdown-sample', 'value'),
    State('graph', 'figure'),
    prevent_initial_call=True
)
def update_cluster_assignments(n_clicks, n_clones, snp_weight,
            assignments, sample, old_fig):
    assign = {}
    for dd in get_assignment_dropdown(assignments):
        cl = int(dd['props']['id'].split('-')[1])
        assign[cl] = dd['props']['value']

    if any([i == None for i in assign.values()]):
        return old_fig, assignments

    cl_options = get_annotation_options(n_clones)
    used_cl_types = [i for i in cl_options if i in assign.values()]
    new_cl = {i: j for i, j in enumerate(used_cl_types)}

    data[sample].update_assignment(assign, new_cl, snp_weight)
    new_fig = data[sample].get_figure()

    new_assign = get_annotation_elements(sample, cl_options, assignment=new_cl)
    new_cl_total = data[sample].get_cluster_number()

    CL_UPDATE_FLAG['apply'] = True
    return new_fig, new_assign, new_cl_total


@callback(
    Output('modal-cluster', 'is_open', allow_duplicate=True),
    Output('div-assignment', 'children', allow_duplicate=True),
    Output('modal-cluster-dropdown', 'value', allow_duplicate=True),
    Input('modal-cluster-dropdown', 'value'),
    State('modal-cluster-title', 'children'),
    State('div-assignment', 'children'),
    prevent_initial_call=True
)
def set_cluster_assignment(cl_type, title, assignments):
    cl_id = title.split(' ')[1]
    for dd in get_assignment_dropdown(assignments):
        dd_id = dd['props']['id'].split('-')[1]
        if dd_id == cl_id:
            dd['props']['value'] = cl_type
    return False, assignments, None


@callback(
    Output('modal-SNP', 'is_open', allow_duplicate=True),
    Output('graph', 'figure', allow_duplicate=True),
    Input('modal-SNP-dropdown', 'value'),
    State('modal-SNP-input','value'),
    State('input-snp-weights', 'value'),
    State('input-n-clusters', 'value'),
    State('dropdown-sample', 'value'),
    prevent_initial_call=True
)
def set_SNP_relevant(val, SNP_raw, snp_weight, n_clusters, sample):
    if not SNP_raw:
        raise PreventUpdate
    SNP, old_val = SNP_raw.split('|')
    # Do nothing if value stays the same
    if (val == 'False' and old_val == '0') \
            or (val == 'True' and old_val == '1'):
        raise PreventUpdate
    new_val = True if val == 'True' else False
    data[sample].update_SNP_rel(SNP, new_val, snp_weight, n_clusters)
    fig = data[sample].get_figure()
    return False, fig


@callback(
    Output('modal-cluster', 'is_open', allow_duplicate=True),
    Output('graph', 'figure', allow_duplicate=True),
    Output('div-assignment', 'children', allow_duplicate=True),
    Output('input-n-clusters', 'value', allow_duplicate=True),
    Input('modal-cluster-split', 'n_clicks'),
    State('modal-cluster-title', 'children'),
    State('input-n-clones', 'value'),
    State('input-snp-weights', 'value'),
    State('div-assignment', 'children'),
    State('dropdown-sample', 'value'),
    prevent_initial_call=True
)
def split_cluster(n, title, n_clones, snp_weight, assignments, sample):
    cl_id = int(title.split(' ')[1])
    data[sample].split_cluster(cl_id, snp_weight)
    fig = data[sample].get_figure()
    # Update assignment
    new_cl_total = data[sample].get_cluster_number()
    new_cl_id = new_cl_total  - 1
    new_color = data[sample].get_cluster_colors()[new_cl_id][1]
    cl_options = get_annotation_options(n_clones)
    new_assign_el = get_new_assignment_dropdown_div(new_cl_id, new_color, cl_options)
    assignments.append(new_assign_el)

    CL_UPDATE_FLAG['split'] = True
    return False, fig, assignments, new_cl_total


@callback(
    Output('hidden-div', 'children'),
    Input('button-safe', 'n_clicks'),
    State('dropdown-sample', 'value'),
    prevent_initial_call=True
)
def safe_assignment(n_clicks, sample):
    data[sample].safe_annotation()
    return


@callback(
    Output('modal-cluster-input', 'value'),
    Output('modal-SNP-input', 'value'),
    Input('graph','clickData'),
    prevent_initial_call=True
)
def open_modal(clicked_raw):
    clicked = clicked_raw['points'][0]
    if clicked['x'] == 'cluster':
        return clicked['z'], ''
    elif clicked['y'] == 'SNP':
        SNP = clicked['x'].split('<br>')[0]
        return '', f'{SNP}|{clicked["z"]}'
    else:
        return '', ''


@callback(
    Output('modal-cluster', 'is_open'),
    Output('modal-cluster-color', 'style'),
    Output('modal-cluster-title', 'children'),
    Input('modal-cluster-input','value'),
    State('dropdown-sample', 'value'),
    prevent_initial_call=True
)
def open_cluster_modal(cluster, sample):
    if cluster == '':
        return False, {}, ''
    colors = data[sample].get_cluster_colors()
    header_style={
        'height': '20px',
        'width': '30px',
        'margin-right': '5px',
        'display': 'inline-block',
        'background-color': colors[cluster][1]
    }
    title = f'Cluster {cluster} - type'
    return True, header_style, title


@callback(
    Output('modal-SNP', 'is_open'),
    Output('modal-SNP-dropdown', 'value'),
    Input('modal-SNP-input','value'),
    State('dropdown-sample', 'value'),
    prevent_initial_call=True
)
def open_SNP_modal(SNP, sample):
    if not SNP:
        return False, ''
    if SNP.split('|')[1] == '0':
        return True, 'False'
    else:
        return True, 'True'


@ callback(
    Output('modal-cluster', 'is_open', allow_duplicate=True),
    Output('modal-SNP', 'is_open', allow_duplicate=True),
    Input('modal-cluster-close', 'n_clicks'),
    Input('modal-SNP-close', 'n_clicks'),
    prevent_initial_call=True
)
def close_modal(n1, n2):
    return False, False


# ------------------------------------------------------------------------------


def main(args):
    # Load panel
    global panel
    panel = Panel(args.panel_file)
    if args.gene_annotation:
        if os.path.isfile(args.gene_annotation):
            panel.add_chrArm(args.gene_annotation)
        else:
            print(f'!Warning: Gene annotation file {args.gene_annotation} '\
                'does not exist.')

    # Load datasets
    global datasets
    datasets = get_datasets(args.in_dir)
    
    # Add samples to dropdown menu in html layout
    samples = sorted(list(datasets.keys()))
    html_layout.children[2].children[1].options = samples
    html_layout.children[2].children[1].value = samples[0]

    # Check if correct panel file is loaded for the (first) sample(s)
    with open(datasets[samples[0]]['reads'], 'r') as f:
        reads_header = f.readline()
    ampl = reads_header.strip().split('\t')[1:]
    assert len(set(panel.df.index) - set(ampl)) == 0, \
        'Amplicons between panel and samples do not match'

    # Init dash app
    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        meta_tags=[{
            'name': 'viewport',
            'content': 'width=device-width, initial-scale=1.0'
        }]
    )
    app.layout = html_layout

    # Load sample data
    global data
    data = {}

    # If multiple cores are available:
    #   Load all data async except first sample (which is loaded when app is called)
    # cores = min(args.cores, cpu_count())
    # if cores > 1:
    #     with Pool(processes=cores - 1) as pool:
    #         for sample in samples[1:]:
    #             pool.apply_async(load_sample, (sample,))

    # Set tracker to count apply button clicks (to prevent callback)
    global CL_UPDATE_FLAG
    CL_UPDATE_FLAG = {'apply': False, 'split': False}

    # Run dash app
    app.run(debug=True)


def parse_args():
    parser = argparse.ArgumentParser(prog='run_annotation',
        usage='python run_annotation.py -snp <DATA> [-args]',
        description='*** Run a dash app to annotate tapestri clusters ***')
    parser.add_argument('-i', '--in_dir', type=str, required=True,
        help='Input directory with barcode and SNP files.')
    parser.add_argument('-reads', '--read_file', type=str,
        help='Reads per barcode distribution from Tapestri processing (.tsv).')
    parser.add_argument('-snps', '--snp_file', type=str,
        help='Variant file from mosaic preprocessing (.csv).')
    parser.add_argument('-p', '--panel_file', type=str, 
        default='data/4387_annotated.bed',
        help='Annotated Tapestri panel bed file')
    parser.add_argument('-ga', '--gene_annotation', type=str,
        default='data/gencode.v19.annotation.cytoBand.tsv',
        help='Gencode mapping: gene - chromosome arm (.tsv).')
    parser.add_argument('-n', '--cores', type=int, default=1,
        help='# cores for parallel data loading in the background. Default = 1')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)