import dash_bootstrap_components as dbc
import dash_daq as daq
from dash import dash_table
from dash import dcc
from dash import html
import pandas as pd
import plotly.graph_objs as go
import random
from math import log10
from shapash.utils.utils import truncate_str
from shapash.webapp.utils.ShapashGraph import ShapashGraph
from shapash.webapp.utils.explanations import Explanations
from shapash.webapp.utils.utils import round_to_k

class AppLayout():

    def __init__(self, app, explainer, settings):
        self.explainer = explainer
        self.app = app
        self.logo = self.app.get_asset_url('shapash-fond-fonce.png')
        self.color = explainer.plot._style_dict["webapp_button"]
        self.bkg_color = explainer.plot._style_dict["webapp_bkg"]
        self.title_menu_color = explainer.plot._style_dict["webapp_title"]
        
        # SETTINGS
        settings_ini = {
            'rows': 1000,
            'points': 1000,
            'violin': 10,
            'features': 20,
        }

        if settings is not None:
            for k, v in settings_ini.items():
                settings_ini[k] = settings[k] if k in settings and isinstance(
                    settings[k], int) and 0 < settings[k] else v
        self.settings = settings_ini.copy()

        self.predict_col = ['_predict_']
        self.explainer.features_imp = self.explainer.state.compute_features_import(
            self.explainer.contributions)
        if self.explainer._case == 'classification':
            self.label = self.explainer.check_label_name(len(self.explainer._classes) - 1, 'num')[1]
            self.selected_feature = self.explainer.features_imp[-1].idxmax()
            self.max_threshold = int(max([x.applymap(lambda x: round_to_k(x, k=1)).max().max()
                                          for x in self.explainer.contributions]))
        else:
            self.label = None
            self.selected_feature = self.explainer.features_imp.idxmax()
            self.max_threshold = int(self.explainer.contributions.applymap(
                lambda x: round_to_k(x, k=1)).max().max())

        # DATA
        self.explanations = Explanations()  # To get explanations of "?" buttons
        self.dataframe = pd.DataFrame()
        self.round_dataframe = pd.DataFrame()
        self.list_index = []
        self.subset = None
        self.last_click_data = None
        
        self.init_data()

        # COMPONENTS
        self.components = {
            'menu': {},
            'table': {},
            'graph': {},
            'filter': {},
            'settings': {}
        }
        self.init_components()

        # LAYOUT
        self.skeleton = {
            'navbar': {},
            'body': {}
        }
        self.make_skeleton()
        self.app.layout = html.Div([self.skeleton['navbar'], self.skeleton['body']])


    @staticmethod
    def create_input_modal(id, label, tooltip):
        return dbc.Row(
            [
                dbc.Label(label, id=f'{id}_label', html_for=id, width=8),
                dbc.Col(
                    dbc.Input(id=id, type="number", value=0),
                    width=4),
                dbc.Tooltip(tooltip, target=f'{id}_label', placement='bottom'),
            ], className="g-3"
        )


    def init_data(self):
        """
        Method which initializes data from explainer object
        """
        if hasattr(self.explainer, 'y_pred'):
            self.dataframe = self.explainer.x_init.copy()
            if isinstance(self.explainer.y_pred, (pd.Series, pd.DataFrame)):
                self.predict_col = self.explainer.y_pred.columns.to_list()[0]
                self.dataframe = self.dataframe.join(self.explainer.y_pred)
            elif isinstance(self.explainer.y_pred, list):
                self.dataframe = self.dataframe.join(
                    pd.DataFrame(data=self.explainer.y_pred,
                                 columns=[self.predict_col],
                                 index=self.explainer.x_init.index)
                    )
            else:
                raise TypeError('y_pred must be of type pd.Series, pd.DataFrame or list')
        else:
            raise ValueError('y_pred must be set when calling compile function.')

        self.dataframe['_index_'] = self.explainer.x_init.index
        self.dataframe.rename(columns={f'{self.predict_col}': '_predict_'}, inplace=True)
        col_order = ['_index_', '_predict_'] + self.dataframe.columns.drop(['_index_', '_predict_']).tolist()
        random.seed(79)
        self.list_index = \
            random.sample(
                population=self.dataframe.index.tolist(),
                k=min(self.settings['rows'], len(self.dataframe.index.tolist()))
            )
        self.dataframe = self.dataframe[col_order].loc[self.list_index].sort_index()
        self.round_dataframe = self.dataframe.copy()
        for col in list(self.dataframe.columns):
            typ = self.dataframe[col].dtype
            if typ == float:
                std = self.dataframe[col].std()
                if std != 0:
                    digit = max(round(log10(1 / std) + 1) + 2, 0)
                    self.round_dataframe[col] = \
                        self.dataframe[col].map(f'{{:.{digit}f}}'.format).astype(float)

    
    
    def adjust_menu(self):
        """
        Override menu from explainer object depending on
        classification or regression case.
        """
        on_style = {'backgroundColor': self.color[0],
                    'color': self.bkg_color,
                    'margin-right': '0.5rem'}
        off_style = {'backgroundColor': self.color[1],
                     'color': self.bkg_color,
                     'margin-right': '0.5rem'}
        if self.explainer._case == 'classification':
            self.components['menu']['select_label'].options = \
                [
                    {'label': f'{self.explainer.label_dict[label] if self.explainer.label_dict else label}',
                     'value': label}
                    for label in self.explainer._classes
            ]
            self.components['menu']['classification_badge'].style = on_style
            self.components['menu']['regression_badge'].style = off_style
            self.components['menu']['select_label'].value = self.label

        elif self.explainer._case == 'regression':
            self.components['menu']['classification_badge'].style = off_style
            self.components['menu']['regression_badge'].style = on_style
            self.components['menu']['select_collapse'].is_open = False

        else:
            raise ValueError(f'No rule defined for explainer case : {self.explainer._case}')

    def init_components(self):
        """
        Initialize components (graph, table, filter, settings, ...) and insert it inside
        components containers which are created by init_skeleton
        """

        self.components['settings']['input_rows'] = self.create_input_modal(
            id='rows',
            label="Number of rows for subset",
            tooltip="Set max number of lines for subset (datatable). \
                        Filter will be apply on this subset."
        )

        self.components['settings']['input_points'] = self.create_input_modal(
            id='points',
            label="Number of points for plot",
            tooltip="Set max number of points in feature contribution plots."
        )

        self.components['settings']['input_features'] = self.create_input_modal(
            id='features',
            label="Number of features to plot",
            tooltip="Set max number of features to plot in features \
                        importance and local explanation plots."
        )

        self.components['settings']['input_violin'] = self.create_input_modal(
            id='violin',
            label="Max number of labels for violin plot",
            tooltip="Set max number of labels to display a violin plot \
                        for feature contribution plot (otherwise a scatter \
                                                    plot is displayed)."
        )

        self.components['settings']['name'] = dbc.Row(
            [
                dbc.Checklist(
                    options=[{"label": "Use domain name for \
                                features name.", "value": 1}], value=[], inline=True,
                    id="name",
                    style={"margin-left": "20px"}
                ),
                dbc.Tooltip("Replace technical feature names by \
                            domain names if exists.",
                            target='name', placement='bottom'),
            ], className="g-3",
        )

        self.components['settings']['modal'] = dbc.Modal(
            [
                dbc.ModalHeader("Settings"),
                dbc.ModalBody(
                    dbc.Form(
                        [
                            self.components['settings']['input_rows'],
                            self.components['settings']['input_points'],
                            self.components['settings']['input_features'],
                            self.components['settings']['input_violin'],
                            self.components['settings']['name']
                        ]
                    )
                ),
                dbc.ModalFooter(
                    dbc.Button("Apply", id="apply", className="ml-auto")
                ),
            ],
            id="modal"
        )

        self.components['menu'] = dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div(
                            daq.BooleanSwitch(
                                id='bool_groups',
                                on=True,
                                style={'display': 'none'} if self.explainer.features_groups is None else {},
                                color=self.color[0],
                                label={
                                    'label': 'Groups',
                                    'style': {
                                        'fontSize': 18,
                                        'color': self.color[0],
                                        'fontWeight': 'bold',
                                        "margin-left": "5px"
                                    },
                                },
                                labelPosition="right"
                            ),
                            style={"margin-right": "35px"}
                        )
                    ],
                    width="auto", align="center",
                ),
                dbc.Col(
                    [
                        html.H4(
                            [dbc.Badge("Regression", id='regression_badge',
                                        style={"margin-right": "5px",
                                                "margin-left": "0px"},
                                        color=''),
                                dbc.Badge("Classification", id='classification_badge', color='')
                                ], style={"margin-right": "5px"}
                        ),
                    ],
                    width="auto", align="center", style={'padding': 'auto'}
                ),
                dbc.Col(
                    dbc.Collapse(
                        dbc.Row(
                            [
                                # 2 columns to have class beside the dropdown buttons
                                dbc.Col([
                                    dbc.Label("Class:", style={'color': 'white', 'margin': '0px'}),
                                ], align="center"),
                                dbc.Col([
                                    dcc.Dropdown(
                                        id="select_label",
                                        options=[], value=None,
                                        clearable=False, searchable=False,
                                        style={"verticalAlign": "middle",
                                                "zIndex": '1010',
                                                "min-width": '160px',
                                                'height': '100%'}
                                    )
                                ], style={"margin-right": "17px",
                                            "padding": "0px",
                                            "width": "auto"})
                            ],
                            style={"margin": "0px"}
                        ),
                        is_open=True, id='select_collapse'
                    ),
                    width="auto", align="center", style={'padding': '0px'}
                ),
                dbc.Col([
                    html.Div(
                        [
                            html.Img(id='settings', title='settings', alt='Settings',
                                        src=self.app.get_asset_url('settings.png'),
                                        height='40px',
                                        style={'cursor': 'pointer'}),
                            self.components['settings']['modal'],
                        ]
                    )],
                    align="center", width="auto", style={'padding': '0px'}
                )
            ],
            className="g-0", justify="end"
        )

        self.adjust_menu()

        self.components['table']['dataset'] = dash_table.DataTable(
            id='dataset',
            data=self.round_dataframe.to_dict('records'),
            tooltip_data=[
                {
                    column: {'value': str(value), 'type': 'text'}
                    for column, value in row.items()
                } for row in self.dataframe.to_dict('rows')
            ], tooltip_duration=2000,

            columns=[{"name": '_index_', "id": '_index_'},
                        {"name": '_predict_', "id": '_predict_'}] +
                    [{"name": i, "id": i} for i in self.explainer.x_init],
            editable=False, row_deletable=False,
            style_as_list_view=True,
            virtualization=True,
            page_action='none',
            fixed_rows={'headers': True, 'data': 0},
            fixed_columns={'headers': True, 'data': 0},
            sort_action='custom', sort_mode='multi', sort_by=[],
            active_cell={'row': 0, 'column': 0, 'column_id': '_index_'},
            style_table={'overflowY': 'auto', 'overflowX': 'auto'},
            style_header={'height': '30px'},
            style_cell={
                'minWidth': '70px', 'width': '120px', 'maxWidth': '200px',
            },
        )

        self.components['graph']['global_feature_importance'] = ShapashGraph(
            figure=go.Figure(), id='global_feature_importance'
        )

        self.components['graph']['feature_selector'] = ShapashGraph(
            figure=go.Figure(), id='feature_selector'
        )

        # Component for the graph prediction picking
        self.components['graph']['prediction_picking'] = ShapashGraph(
            figure=go.Figure(), id='prediction_picking'
        )

        self.components['graph']['detail_feature'] = ShapashGraph(
            figure=go.Figure(), id='detail_feature'
        )

        # Component create to filter the dataset
        self.components['filter']['filter_dataset'] = dbc.Col(
            [dbc.Row(
                html.Div(
                    id='main',
                    children=[
                        html.Div(
                                id='filters',
                                children=[
                                    # Create Add Filter button
                                    dbc.Button(
                                        id='add_dropdown_button',
                                        children='Add Filter',
                                        color='warning',
                                        size='sm',
                                        style={'margin-right': '20px'}
                                    ),
                                    # Create reset Filter button (disabled of no filters applied)
                                    dbc.Button(
                                        id='reset_dropdown_button',
                                        children='Reset all existing filters',
                                        color='warning', disabled=True,
                                        size='sm',
                                        style={'margin-right': '20px'}
                                    ),
                                    # Create explanation button
                                    dbc.Button(
                                        "?",
                                        id="open_filter",
                                        size='sm',
                                        color="warning",
                                        ),
                                    # Create popover on the explanation button
                                    dbc.Popover(
                                        "Click here to know how you can apply filters.",
                                        target="open_filter",
                                        body=True,
                                        trigger="hover",
                                    ),
                                    # Modal associated to the explanation button
                                    dbc.Modal(
                                            [
                                            dbc.ModalHeader(
                                                dbc.ModalTitle("Filters explanation")
                                                ),
                                            dbc.ModalBody([
                                                html.Div(
                                                    dcc.Markdown(
                                                        self.explanations.filter
                                                        )
                                                    )
                                                ]),
                                            dbc.ModalFooter(
                                                dbc.Button(
                                                    "Close",
                                                    id="close_filter",
                                                    color="warning"
                                                    )
                                                ),
                                            ],
                                            id="modal_filter",
                                            centered=True,
                                            size='lg'
                                                ),
                                    # Div which will contains the filters
                                    html.Div(
                                        id='dropdowns_container',
                                        children=[]
                                    )
                                ]
                            )
                        ]
                    )
                ),
                dbc.Row(
                    html.Div([
                        html.Br(),
                        # Create Apply Filter button (Hidden if no filter to apply)
                        dbc.Button(
                                id='apply_filter',
                                children='Apply filters',
                                color='warning',
                                size='sm',
                                style={'display': 'none'}
                                )
                            ],
                        )
                    )
                ], style={'maxheight': '22rem', 'height': '21rem', 'zIndex': 800}
            )

        self.components['filter']['index'] = dbc.Col(dbc.Row(
            [
                dbc.Label("Index", align="center", width=4),
                dbc.Col([
                    dbc.Input(
                        id="index_id", type="text", size="s", placeholder="Id must exist",
                        debounce=True, persistence=True, style={'textAlign': 'right'}
                    )], width={"size": 5},
                    style={'padding': "0px"}
                ),
                dbc.Col([
                    html.Img(id='validation', alt='Validate', title='Validate index',
                                src=self.app.get_asset_url('reload.png'),
                                height='30px', style={'cursor': 'pointer'},
                                )], width={"size": 2},
                        style={'padding': "0px"}, align="center"
                        )
            ])
        )

        self.components['filter']['threshold'] = dbc.Col(
            [
                dbc.Label("Threshold", html_for="slider", id='threshold_label'),
                dcc.Slider(
                    min=0, max=self.max_threshold, value=0, step=0.1,
                    marks={f'{round(self.max_threshold * mark / 4)}': f'{round(self.max_threshold * mark / 4)}'
                            for mark in range(5)},
                    id="threshold_id",
                )
            ],
            className='filter_dashed'
        )

        self.components['filter']['max_contrib'] = dbc.Col(
            [
                dbc.Label(
                    "Features to display: ", id='max_contrib_label'),
                dcc.Slider(
                    min=1, max=min(self.settings['features'], len(self.dataframe.columns) - 2),
                    step=1, value=min(self.settings['features'], len(self.dataframe.columns) - 2),
                    id="max_contrib_id",
                )
            ],
            className='filter_dashed'
        )

        self.components['filter']['positive_contrib'] = dbc.Col(
            [dbc.Row(
                [
                    dbc.Label("Contributions to display:", style={'font-size': '95%'}),
                ]
            ),
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Checklist(
                                options=[{"label": "Positive", "value": 1}], value=[1], inline=True,
                                id="check_id_positive",
                                # define the font-size style
                                style={'font-size': '82%'}
                            ), style={'display': 'inline-block'}
                        ),
                        dbc.Col(
                            dbc.Checklist(
                                options=[{"label": "Negative", "value": 1}], value=[1], inline=True,
                                id="check_id_negative",
                                # define the font-size style
                                style={'font-size': '82%'}
                                ), style={'display': 'inline-block'}, align="center"
                            )
                    ], className="g-0", justify="center"
                )
            ],
            className='filter_dashed'
        )

        self.components['filter']['masked_contrib'] = dbc.Col(
            [
                dbc.Label(
                    "Feature(s) to mask:"),
                dcc.Dropdown(options=[{'label': key, 'value': value} for key, value in sorted(
                    self.explainer.inv_features_dict.items(), key=lambda item: item[0])],
                    value='', multi=True, searchable=True,
                    id="masked_contrib_id"
                ),
            ],
            className='filter_dashed'
        )


    def make_skeleton(self):
        """
        Describe the app skeleton (bootstrap grid) and initialize components containers
        """
        self.skeleton['navbar'] = dbc.Container(
            [
                dbc.Row([
                        dbc.Col(
                            html.A(
                                dbc.Row([
                                    dbc.Col([
                                        html.Img(src=self.logo, height="40px")], className='col-1'),
                                    dbc.Col([
                                        html.H4("Shapash Monitor", id="shapash_title")]),
                                    ],
                                    align="center", style={'color': self.title_menu_color}
                                ),
                                href="https://github.com/MAIF/shapash", target="_blank",
                            ),
                            # Change md=3 to md=2
                            md=2, align="center", width="100%", style={'padding': 'auto'}
                        ),
                        dbc.Col([
                            html.A(
                                dbc.Row([
                                        html.H3(truncate_str(self.explainer.title_story, maxlen=40),
                                                id="shapash_title_story",
                                                style={'text-align': 'center'})]
                                        ),
                                href="https://github.com/MAIF/shapash", target="_blank",
                            )],
                            # Change md=3 to md=4
                            md=4, align="center", width="100%", style={'padding': 'auto'}
                        ),
                        dbc.Col([
                            self.components['menu']
                            ], align="end", md=6, width='100%'
                        )
                        ],
                        style={'padding': "5px 15px",
                                "verticalAlign": "middle",
                                "width": "auto",
                                "justify": "end"}
                        )
            ],
            fluid=True, style={'height': '100%', 'backgroundColor': self.bkg_color
                                }
        )

        self.skeleton['body'] = dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card([
                                        html.Div(
                                            # To drow the global_feature_importance graph
                                            self.draw_component('graph', 'global_feature_importance'),
                                            id="card_global_feature_importance",
                                            # Position must be absolute to add the explanation button
                                            style={"position": 'absolute'}
                                            ),
                                        html.Div([
                                                # Create explanation button on feature importance graph
                                                dbc.Button(
                                                    "?",
                                                    id="open_feature_importance",
                                                    size='sm',
                                                    color="warning"
                                                    ),
                                                # Create popover for this button
                                                dbc.Popover(
                                                    "Click here to have more \
                                                    information on Feature Importance graph.",
                                                    target="open_feature_importance",
                                                    body=True,
                                                    trigger="hover",
                                                        ),
                                                # Create modal associated to this button
                                                dbc.Modal([
                                                        # Modal title
                                                        dbc.ModalHeader(
                                                            dbc.ModalTitle("Feature importance")
                                                            ),
                                                        dbc.ModalBody([
                                                            html.Div(
                                                                # Add explanation
                                                                dcc.Markdown(
                                                                    self.explanations.feature_importance
                                                                    )
                                                                    ),
                                                            # Here to add link in the modal
                                                            html.A('Click here for more details',
                                                                    href="https://github.com/MAIF/shapash/blob/master/tutorial/plot/tuto-plot03-features-importance.ipynb",
                                                                    # open new brother tab
                                                                    target="_blank",
                                                                    style={'color': self.color[0]})
                                                        ]),
                                                        # button to close the modal
                                                        dbc.ModalFooter(
                                                            dbc.Button(
                                                                "Close",
                                                                id="close_feature_importance",
                                                                color="warning"
                                                            )
                                                        ),
                                                    ],
                                                    id="modal_feature_importance",
                                                    centered=True,
                                                    size='lg'
                                                )
                                                ],
                                                # position must be relative
                                                style={'position': 'relative', 'left': '96%'})
                                    ])
                            ],
                            md=5,
                            style={'padding': '10px 10px 0px 10px'},
                        ),
                        dbc.Col(
                            [
                                # Tabs that contain 3 children tab (Dataset,
                                # Dataset Filters and True Value Vs Pedicted Values)
                                dbc.Tabs([
                                    # Tab which contains the datatable component
                                    dbc.Tab(
                                        # draw datatable component
                                        self.draw_component('table', 'dataset'),
                                        # Tab name
                                        label='Dataset',
                                        className="card",
                                        id='card_dataset',
                                        # Style of the tab
                                        style={'cursor': 'pointer'},
                                        label_style={'color': "black", 'height': '30px',
                                                        'padding': '0px 5px'},
                                        # Style when the tab is activated
                                        active_tab_class_name="fw-bold fst-italic",
                                        active_label_style={'border-top': '3px solid',
                                                            'border-top-color': self.color[0]
                                                            }
                                        ),
                                    # Tab which contains components to filter the dataset
                                    dbc.Tab(
                                        dbc.Card(
                                            dbc.CardBody(
                                                html.Div(
                                                    # draw the component
                                                    self.draw_filter_table(),
                                                    id='card_filter_dataset',
                                                    # To add scroll in overflow y
                                                    style={'overflow-y': "scroll",
                                                            'overflow-x': 'hidden'})
                                            ), style={'height': '24.1rem'},
                                        ),
                                        # Tab name
                                        label='Dataset Filters',
                                        # Style of the tab
                                        label_style={'color': "black", 'height': '30px',
                                                        'padding': '0px 5px'},
                                        tab_style={'border-left': '2px solid #ddd',
                                                    'border-right': '2px solid #ddd'},
                                        # Style when the tab is activated
                                        active_tab_class_name="fw-bold fst-italic",
                                        active_label_style={'border-top': '3px solid',
                                                            'border-top-color': self.color[0]
                                                            }
                                            ),
                                    # Tab which contains prediction picking graph
                                    # and its explanation button
                                    dbc.Tab(
                                        dbc.Card([
                                            html.Div(
                                                # draw prediction picking graph
                                                self.draw_component('graph', 'prediction_picking'),
                                                id="card_prediction_picking",
                                                # Position must be absolute to add
                                                # the explanation button
                                                style={"position": 'absolute'}
                                            ),
                                            html.Div([
                                                    # Create explanation button
                                                    dbc.Button(
                                                        "?",
                                                        id="open_prediction_picking",
                                                        size='sm',
                                                        color="warning"
                                                        ),
                                                    # Create popover for this button
                                                    dbc.Popover(
                                                            "Click here to have more \
                                                            information on Prediction Picking graph.",
                                                            target="open_prediction_picking",
                                                            body=True,
                                                            trigger="hover",
                                                        ),
                                                    # Create modal associated to this button
                                                    dbc.Modal([
                                                                # Modal title
                                                                dbc.ModalHeader(
                                                                dbc.ModalTitle("True values Vs Predicted values")
                                                                ),
                                                                dbc.ModalBody([
                                                                    html.Div(
                                                                        # explanation
                                                                        dcc.Markdown(self.explanations.prediction_picking)
                                                                        ),
                                                                    # Here to add link in the modal
                                                                    html.Div(html.Img(src="https://github.com/MAIF/shapash/blob/master/docs/_static/shapash_select_subset.gif?raw=true")),
                                                                    # Here to add a link in the modal
                                                                    html.A('Click here for more details',
                                                                            href="https://github.com/MAIF/shapash/blob/master/tutorial/plot/tuto-plot06-prediction_plot.ipynb",
                                                                            # open new brother tab
                                                                            target="_blank",
                                                                            style={'color': self.color[0]})
                                                                    ]),
                                                                dbc.ModalFooter(
                                                                    # button to close the modal
                                                                    dbc.Button(
                                                                        "Close",
                                                                        id="close_prediction_picking",
                                                                        color="warning"
                                                                    )
                                                                ),
                                                        ],
                                                        id="modal_prediction_picking",
                                                        centered=True,
                                                        size='lg'
                                                    )
                                                    # Position must be relative
                                                ],  style={'position': 'relative', 'left': '97%'})
                                            ]),
                                        # Tab name
                                        label='True Values Vs Predicted Values',
                                        # Style of the tab
                                        label_style={'color': "black", 'height': '30px',
                                                        'padding': '0px 5px'},
                                        # Style when the tab is activated
                                        active_tab_class_name="fw-bold fst-italic",
                                        active_label_style={'border-top': '3px solid',
                                                            'border-top-color': self.color[0]
                                                            }
                                    )
                                ], id="tabs"
                                )
                            ],
                            md=7,
                            style={'padding': '10px 10px'},
                        ),
                    ], style={'padding': '10px 10px 0px 10px',
                                'height': '100%'}, align="center"
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                # Card which contains feature selector graph
                                # and explanation button
                                dbc.Card([
                                        html.Div(
                                            # Draw feature selector graph
                                            self.draw_component('graph', 'feature_selector'),
                                            id='card_feature_selector',
                                            # Position must be absolute to
                                            # add explanation button
                                            style={"position": 'absolute'}
                                                ),
                                            html.Div([
                                                # Create explanation button
                                                dbc.Button(
                                                    "?",
                                                    id="open_feature_selector",
                                                    size='sm',
                                                    color="warning"
                                                    ),
                                                # popover of this button
                                                dbc.Popover(
                                                            "Click here to have more \
                                                            information on Feature Selector graph.",
                                                            target="open_feature_selector",
                                                            body=True,
                                                            trigger="hover",
                                                        ),
                                                # Modal of this button
                                                dbc.Modal([
                                                        dbc.ModalHeader(
                                                            dbc.ModalTitle("Feature selector")
                                                            ),
                                                        dbc.ModalBody([
                                                            html.Div(
                                                                # explanations
                                                                dcc.Markdown(self.explanations.feature_selector)
                                                                ),
                                                            # Here to add link
                                                            html.A('Click here for more details',
                                                                    href="https://github.com/MAIF/shapash/blob/master/tutorial/plot/tuto-plot02-contribution_plot.ipynb",
                                                                    # open new brother tab
                                                                    target="_blank",
                                                                    style={'color': self.color[0]})
                                                                ]),
                                                        dbc.ModalFooter(
                                                            # Button to close modal
                                                            dbc.Button(
                                                                "Close",
                                                                id="close_feature_selector",
                                                                color="warning"
                                                            )
                                                        ),
                                                    ],
                                                    id="modal_feature_selector",
                                                    centered=True,
                                                    size='lg'
                                                )
                                                ],
                                                    # position must be relative
                                                    style={'position': 'relative',
                                                        'left': '96%'})
                                            ])
                            ],
                            md=5,
                            align="center",
                            style={'padding': '0px 10px'},
                        ),
                        dbc.Col(
                            [
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                # Card that contains detail feature graph
                                                # and explanation button
                                                dbc.Card([
                                                    html.Div(
                                                        # draw detail_feature graph
                                                        self.draw_component('graph', 'detail_feature'),
                                                        id='card_detail_feature',
                                                        style={"position": 'absolute'}
                                                    ),
                                                    html.Div([
                                                        # Create explanation button
                                                        dbc.Button(
                                                            "?",
                                                            id="open_detail_feature",
                                                            size='sm',
                                                            color="warning"
                                                            ),
                                                        # Popover of this button
                                                        dbc.Popover(
                                                            "Click here to have more \
                                                            information on Detail Feature graph.",
                                                            target="open_detail_feature",
                                                            body=True,
                                                            trigger="hover",
                                                        ),
                                                        # Modal of this button
                                                        dbc.Modal(
                                                            [
                                                                dbc.ModalHeader(
                                                                    dbc.ModalTitle("Detail feature")
                                                                    ),
                                                                dbc.ModalBody([
                                                                    html.Div(
                                                                        # explanations
                                                                        dcc.Markdown(self.explanations.detail_feature)
                                                                        ),
                                                                    # Here to add link on the modal
                                                                    html.A('Click here for more details',
                                                                        href="https://github.com/MAIF/shapash/blob/master/tutorial/plot/tuto-plot01-local_plot-and-to_pandas.ipynb",
                                                                        # open new brother tab
                                                                        target="_blank",
                                                                        style={'color': self.color[0]})
                                                                ]),
                                                                dbc.ModalFooter(
                                                                    # Button to close the modal
                                                                    dbc.Button(
                                                                        "Close",
                                                                        id="close_detail_feature",
                                                                        color="warning"
                                                                        )
                                                                ),
                                                            ],
                                                            id="modal_detail_feature",
                                                            centered=True,
                                                            size='lg'
                                                        )

                                                        ],
                                                            # Position must be relative
                                                            style={
                                                            'position': 'relative',
                                                            'left': '96%'
                                                            })
                                                ])
                                            ],
                                            md=8,
                                            align="center",
                                        ),
                                        dbc.Col(
                                            [
                                                html.Div(
                                                    self.draw_filter(),
                                                    className="card_filter",
                                                    id='card_filter',
                                                ),
                                            ],
                                            md=4,
                                            align="center",
                                        ),
                                    ],
                                ),
                            ],
                            md=7,
                            align="center",
                            style={'padding': '0px 10px'},
                        ),
                    ],
                    style={'padding': '10px 5px 10px 10px'},
                ),
            ],
            className="mt-12",
            fluid=True,
            # To drop the x scroll-bar
            style={'overflow-x': 'hidden'}
        )

    def draw_component(self,
                       component_type,
                       component_id,
                       title=None):
        """
        Method which return a component from a type and id.
        It's the method to insert component inside component container.
        Parameters
        ----------
        component_type : string
            Type of the component. Can be table, graph, ...
        component_id : string
            Id of the component. It must be unique.
        title : string, optional
            by default None
        Returns
        -------
        list
            list of components
            (combining for example Graph + embed button to get fullscreen
             details)
        """
        component = [html.H4(title)] if title else []
        component.append(self.components[component_type][component_id])
        component.append(
            html.A(
                html.I("fullscreen",
                       className="material-icons tiny",
                       style={'marginTop': '8px', 'marginLeft': '1px'}
                       ),
                id=f"ember_{component_id}",
                className="dock-expand",
                **{'data-component-type': component_type},
                # Get components'id
                **{'data-component-id': component_id}
            )
        )
        return component
        
    def draw_filter_table(self):
        """
        Method which returns the filter dataset components block.
        Returns
        -------
            component
        """
        return self.components['filter']['filter_dataset']
    
    def draw_filter(self):
        """
        Method which returns filter components block for local
        contributions plot.
        Returns
        -------
        list
            list of components
        """
        _filter = [
            dbc.Container(
                [
                    dbc.Row([self.components['filter']['index']],
                            align="center", style={"height": "4rem"}
                            ),
                    dbc.Row([self.components['filter']['threshold']],
                            align="center", style={"height": "5rem"}
                            ),
                    dbc.Row([self.components['filter']['max_contrib']],
                            align="center", style={"height": "5rem"}
                            ),
                    dbc.Row([self.components['filter']['positive_contrib']],
                            align="center", style={"height": "4rem"}
                            ),
                    dbc.Row([self.components['filter']['masked_contrib']],
                            align="center"),
                ],
            ),
        ]
        return _filter