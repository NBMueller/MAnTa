#!/usr/bin/env python3

import argparse
import os
from libs.TapestriDNA import TapestriDNA, get_cluster_colors

from dash import Dash, html, dcc, callback, Output, Input, State, ctx


DEF_CLUSTERS = 3
CLUSTER_TYPES = ['tumor', 'doublets', 'healthy']


def get_annotation_elements(n_clusters, used_types=[]):
    el = []
    colors = get_cluster_colors(n_clusters)
    for cl in range(n_clusters):
        new_el = html.Div([
            html.Div(style={'height': '15px', 'width': '30px', 
                'display': 'inline-block', 'background-color':colors[cl][1]}),
            html.H4(f'{cl}: ', style={'display':'inline-block',
                'margin-left': 10, 'margin-right': 10, 'margin-top': 0, 
                'margin-bottom': 0, 'color': colors[cl][1]}),
            
        ])
        if len(used_types) > 0:
            new_el.children.append(
                dcc.Dropdown(CLUSTER_TYPES, value=used_types[cl],
                    id=f'dropdown-{cl}', style={'width': '120px'})
            )
        else:
            new_el.children.append(
                dcc.Dropdown(CLUSTER_TYPES, id=f'dropdown-{cl}',
                    style={'width': '120px'})
            )
        el.append(new_el)
    return el


@callback(
    Output('input-n-clusters', 'value'),
    Output('div-output', 'children'),
    Input('dropdown-patient', 'value')
)
def load_data(patient):
    global data
    data = TapestriDNA(datasets[patient]['reads'], datasets[patient]['SNPs'],
        panel_file)
    out_file = data.get_out_file()
    return DEF_CLUSTERS, out_file


@callback(
    Output('graph', 'figure', allow_duplicate=True),
    Output('div-assignment', 'children', allow_duplicate=True),
    Input('input-n-clusters', 'value'),
    prevent_initial_call=True
)
def update_cluster_number(n_clusters):
    data.update_cluster_number(n_clusters)
    fig = data.get_figure()
    annot_el = get_annotation_elements(n_clusters)
    return fig, annot_el


@callback(
    Output('graph', 'figure', allow_duplicate=True),
    Output('div-assignment', 'children', allow_duplicate=True),
    Input('button-apply', 'n_clicks'),
    State('div-assignment', 'children'),
    prevent_initial_call=True
)
def update_cluster_assignment(n_clicks, assignments):
    assign = {}
    for div_outer in assignments:
        for div_inner in div_outer['props']['children']:
            if div_inner['type'] == 'Dropdown':
                cl = int(div_inner['props']['id'].split('-')[1])
                cl_type = div_inner['props']['value']
                assign[cl] = cl_type
    used_types = [i for i in CLUSTER_TYPES if i in assign.values()]

    data.update_assignment(assign, used_types)
    fig = data.get_figure()
    annot_el = get_annotation_elements(len(used_types), used_types)

    return fig, annot_el


@callback(
    Output('hidden-div', 'children'),
    Input('button-safe', 'n_clicks'),
    prevent_initial_call=True
)
def safe_assignment(n_clicks):
    data.safe_annotation()
    return


def get_datasets(in_dir):
    datasets = {}
    for file in os.listdir(in_dir):
        if file.endswith('annotated.csv'):
            continue
        file_full = os.path.join(args.in_dir, file)
        prefix = file.split('.')[0]
        if not prefix in datasets:
            datasets[prefix] = {}
        if file.endswith('variants.csv'):
            datasets[prefix]['SNPs'] = file_full
        elif file.endswith('barcode.cell.distribution.merged.tsv'):
            datasets[prefix]['reads'] = file_full
        else:
            raise IOError('Unknown input file: {file_full}')
    return datasets


def main(args):
    global datasets
    datasets = get_datasets(args.in_dir)

    app = Dash(__name__)
    app.layout = html.Div([
        html.H1(children='Manual annotation', style={'textAlign':'center'}),
        html.Hr(),
        html.Div([
            html.H4('Patient: ', style={'display':'inline-block',
                'margin-right': 10}),
            dcc.Dropdown(options=sorted(list(datasets.keys())), 
                value=sorted(list(datasets.keys()))[0],
                id=f'dropdown-patient', style={'width': '200px'})
        ], style={'display': 'flex', 'align-items': 'baseline'}),
        html.Div([
            html.H4('# Clusters: ', style={'display':'inline-block',
                'margin-right': 10}),
            dcc.Input(id='input-n-clusters', type='number', 
                placeholder='No. Clusters', value=DEF_CLUSTERS, min=1, max=30,
                step=1, style={'width': '50px'})
        ]),
        html.Div([
            html.Div(id='div-assignment',
                style={'display': 'flex', 'flex-wrap': 'wrap'}),
            html.Button('Apply', id='button-apply', n_clicks=0),
        ]),
        html.Div([
            html.Button('Save annotation', id='button-safe', n_clicks=0, 
                style={'margin-right': 10}),
            html.Div(id='div-output'),
        ], style={'display':'flex'}),
        html.Hr(),
        dcc.Graph(id='graph', style={'height': '100vh'}),
        html.Div(id='hidden-div', style={'display': 'none'})
    ])

    global panel_file
    panel_file = args.panel_file
    app.run(debug=True)


def parse_args():
    parser = argparse.ArgumentParser(prog='run_annotation',
        usage='python run_annotation.py -snp <DATA> [-args]',
        description='*** Run a dash app to annotate tapestri clusters ***')
    parser.add_argument('-i', '--in_dir', type=str, required=True,
        help='Input directory with barcode and SNP files.'),
    parser.add_argument('-reads', '--read_file', type=str,
        help='Reads per barcode distribution from Tapestri processing (.tsv).'),
    parser.add_argument('-snps', '--snp_file', type=str,
        help='Variant file from mosaic preprocessing (.csv).'),
    parser.add_argument('-p', '--panel_file', type=str, 
        default='data/4387_annotated.bed',
        help='Tapestri panel bed file')
    parser.add_argument('-o', '--output', type=str, default='',
        help='Output directory. default = <DIR:SNP_FILE>')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)