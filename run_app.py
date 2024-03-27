#!/usr/bin/env python3

import argparse
import os
from libs.TapestriDNA import TapestriDNA, Panel

import dash_bootstrap_components as dbc
from dash import Dash, html, dcc, callback, Output, Input, State, ctx


DEF_CLUSTERS = 3
DEF_SNP_WEIGHT = 0.8
CLUSTER_TYPES = ['tumor', 'doublets', 'healthy']


# ------------------------------- HTML LAYOUT ----------------------------------


modal = html.Div([
    dbc.Modal([
        dbc.ModalHeader([
            html.Div(id='modal-color'),
            dbc.ModalTitle(id='modal-title')
        ]),
        dbc.ModalBody(
            [
            dcc.Dropdown(CLUSTER_TYPES, id='model-dropdown',
                style={'width': '120px'}),
            html.Button('Split cluster', id='modal-split', n_clicks=0)
            ],
            style={'display': 'flex', 'justify-content': 'space-around'}),
        dbc.ModalFooter(dbc.Button('Close', id='modal-close', className='ms-auto')),
        ],
        id='modal-cluster', is_open=False,
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
                id='dropdown-patient', style={'width': '200px'})
        ], style={'display': 'flex', 'align-items': 'baseline'}),
        html.Div([
            html.H4('# Clusters: ', style={'display':'inline-block',
                'margin-right': 10}),
            dcc.Input(id='input-n-clusters', type='number', 
                placeholder='No. Clusters', value=DEF_CLUSTERS, min=1, max=30,
                step=1, style={'width': '50px'}),
            html.Div([
                html.H4('weights: ', style={'display':'inline-block',
                'margin-right': 10, 'margin-left': 10}),
                html.Div('SNPs - ', style={'margin-right': 3}),
                dcc.Input(id='input-snp-weights', type='number',
                    value=DEF_SNP_WEIGHT, min=0, max=1, step=0.05,
                    style={'width': '60px'}),
                html.Div(f': {1 - DEF_SNP_WEIGHT:.2f} - reads', id='div-read-weight'),
            ], style={'display': 'flex', 'align-items': 'center'}),
        ], style={'display': 'flex', 'align-items': 'center'}),
        html.Div([
            html.Div(id='div-assignment',
                style={'display': 'flex', 'flex-wrap': 'wrap'}),
            html.Button('Apply', id='button-apply', n_clicks=0),
        ]),
        html.Div([
            html.Button('Save annotation', id='button-safe', 
                style={'margin-right': 10}),
            html.Div(id='div-output'),
        ], style={'display':'flex'}),
        html.Hr(),
        dcc.Graph(id='graph', style={'height': '100vh'}),
        html.Div(id='hidden-div', style={'display': 'none'}),
        modal
    ],
    style={'margin-left': 0, 'margin-right': 0, 'max-width': '100vw'},
)


# ----------------------------- HELPER FUNCTIONS -------------------------------


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
            print('! WARNING: unknown input file: {file_full} !')
    return datasets


def get_annotation_elements(assignment={}):
    el = []
    colors = data.get_cluster_colors()
    for cl in range(data.get_cluster_number()):
        new_el = get_new_assignment_dropdown_div(cl, colors[cl][1])
        if len(assignment) > 0:
            new_el.children[2].value = assignment[cl]
        el.append(new_el)
    return el


def get_new_assignment_dropdown_div(cl, color):
    new_el = html.Div([
        html.Div(style={'height': '15px', 'width': '30px', 
            'display': 'inline-block', 'background-color': color}),
        html.H4(f'{cl}: ', style={'display':'inline-block',
            'margin-left': 10, 'margin-right': 10, 'margin-top': 0, 
            'margin-bottom': 0, 'color': color}),
        dcc.Dropdown(CLUSTER_TYPES, value=None, id=f'dropdown-{cl}',
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

# ----------------------------- CALLBACKS --------------------------------------

@callback(
    Output('input-n-clusters', 'value'),
    Output('div-output', 'children'),
    Input('dropdown-patient', 'value')
)
def load_data(patient):
    data.load_sample_data(datasets[patient]['reads'], datasets[patient]['SNPs'])
    out_file = data.get_out_file()
    return DEF_CLUSTERS, out_file


@callback(
    Output('graph', 'figure', allow_duplicate=True),
    Output('div-assignment', 'children', allow_duplicate=True),
    Output('div-read-weight', 'children'),
    Input('input-n-clusters', 'value'),
    Input('input-snp-weights', 'value'),
    prevent_initial_call=True
)
def update_cluster_number(n_clusters, snp_weight):
    data.update_cluster_number(n_clusters, snp_weight)
    fig = data.get_figure()
    annot_el = get_annotation_elements()
    return fig, annot_el, f': {1 - snp_weight:.2f} - reads'


@callback(
    Output('graph', 'figure', allow_duplicate=True),
    Output('div-assignment', 'children', allow_duplicate=True),
    Input('button-apply', 'n_clicks'),
    State('div-assignment', 'children'),
    State('graph', 'figure'),
    prevent_initial_call=True
)
def update_cluster_assignments(n_clicks, assignments, old_fig):
    assign = {}
    for dd in get_assignment_dropdown(assignments):
        cl = int(dd['props']['id'].split('-')[1])
        assign[cl] = dd['props']['value']

    if any([i == None for i in assign.values()]):
        return old_fig, assignments

    used_cl_types = [i for i in CLUSTER_TYPES if i in assign.values()]
    new_cl = {i: j for i, j in enumerate(used_cl_types)}

    data.update_assignment(assign, new_cl)
    new_fig = data.get_figure()
    new_assign = get_annotation_elements(new_cl)

    return new_fig, new_assign


@callback(
    Output('modal-cluster', 'is_open', allow_duplicate=True),
    Output('div-assignment', 'children', allow_duplicate=True),
    Output('model-dropdown', 'value', allow_duplicate=True),
    Input('model-dropdown', 'value'),
    State('modal-title', 'children'),
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
    Output('modal-cluster', 'is_open', allow_duplicate=True),
    Output('graph', 'figure', allow_duplicate=True),
    Output('div-assignment', 'children', allow_duplicate=True),
    Input('modal-split', 'n_clicks'),
    State('modal-title', 'children'),
    State('input-snp-weights', 'value'),
    State('div-assignment', 'children'),
    prevent_initial_call=True
)
def split_cluster(n, title, snp_weight, assignments):
    cl_id = int(title.split(' ')[1])
    data.split_cluster(cl_id, snp_weight)
    fig = data.get_figure()
    # Update assignment
    new_cl_id = data.get_cluster_number()  - 1
    new_color = data.get_cluster_colors()[new_cl_id][1]
    new_assign_el = get_new_assignment_dropdown_div(new_cl_id, new_color)
    assignments.append(new_assign_el)
    return False, fig, assignments   



@callback(
    Output('hidden-div', 'children'),
    Input('button-safe', 'n_clicks'),
    prevent_initial_call=True
)
def safe_assignment(n_clicks):
    data.safe_annotation()
    return


@callback(
    Output('modal-cluster', 'is_open'),
    Output('modal-color', 'style'),
    Output('modal-title', 'children'),
    Input('graph','clickData'),
    State('input-n-clusters', 'value'),
    prevent_initial_call=True
)
def open_model(clicked, n_clusters):
    colors = data.get_cluster_colors()
    cluster = clicked['points'][0]['z']
    header_style={
        'height': '20px',
        'width': '30px',
        'margin-right': '5px',
        'display': 'inline-block',
        'background-color':colors[cluster][1]
    }
    title = f'Cluster {cluster} - type'
    return True, header_style, title


@ callback(
    Output('modal-cluster', 'is_open', allow_duplicate=True),
    Input('modal-close', 'n_clicks'),
    prevent_initial_call=True
)
def close_modal(n_clicks):
    return False


# ------------------------------------------------------------------------------


def main(args):
    # Load panel
    panel = Panel(args.panel_file)

    # Load datasets
    global datasets
    datasets = get_datasets(args.in_dir)
    
    # Add samples to dropdown menu in html layout
    samples = sorted(list(datasets.keys()))
    html_layout.children[2].children[1].options = samples
    html_layout.children[2].children[1].value = samples[0]

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
    data = TapestriDNA(panel)

    # Run dash app
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
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)