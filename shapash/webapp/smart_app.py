"""
Main class of Web application Shapash
"""

import datetime
import re

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash import ALL, MATCH, dcc, html, no_update
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from flask import Flask

from shapash.utils.utils import truncate_str
from shapash.webapp.layout import AppLayout
from shapash.webapp.utils.ShapashGraph import ShapashGraph
from shapash.webapp.utils.utils import check_row

# FIXME: un clic sur une barre des features importance ne réinitialise pas le diagramme, même sur une version de la webapp non modifiée
# FIXME: refactorer le code
# - appliquer un modèle style MVC
#   - un module d'initialisation de l'IHM en fonction de l'explainer
#   - un ou plusieurs modules de gestion des callbacks
#   - un éventuel module de tranformation des données
#   
# - réduire le nombre d'inputs par callback en en créant de nouveaux (réduction de la complexité)
# - développer des TU validant le fonctionnement des callbacks

class SmartApp:
    """
        Bridge pattern decoupling the application part from SmartExplainer and SmartPlotter.
        Attributes
        ----------
        explainer: object
            SmartExplainer instance to point to.
    """

    def __init__(self, explainer, settings: dict = None):
        """
        Init on class instantiation, everything to be able to run the app on server.
        Parameters
        ----------
        explainer : SmartExplainer
            SmartExplainer object
        settings : dict
            A dict describing the default webapp settings values to be used
            Possible settings (dict keys) are 'rows', 'points', 'violin', 'features'
            Values should be positive ints
        """
        # APP
        self.server = Flask(__name__)
        self.app = dash.Dash(
            server=self.server,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
        )
        self.app.title = 'Shapash Monitor'
        if explainer.title_story:
            self.app.title += ' - ' + explainer.title_story
        self.explainer = explainer

        self.layout = AppLayout(self.app, self.explainer, settings)

        # CALLBACK
        self.callback_fullscreen_buttons()
        self.init_callback_settings()
        self.callback_generator()

    @staticmethod
    def select_point(figure,
                     click_data):
        """
        Method which set the selected point in graph component
        corresponding to click_data.
        """
        if click_data:
            curve_id = click_data['points'][0]['curveNumber']
            point_id = click_data['points'][0]['pointIndex']
            # for curve in range(
            #         len(self.layout.components['graph'][graph].figure['data'])):
            #     self.layout.components['graph'][graph].figure['data'][curve].selectedpoints = \
            #         [point_id] if curve == curve_id else []
            for curve in range(
                    len(figure['data'])):
                figure['data'][curve].selectedpoints = \
                    [point_id] if curve == curve_id else []

    def callback_fullscreen_buttons(self):
        """
        Initialize callbacks for each fullscreen button
        the callback alter style of the component (height, ...)
        Returns
        -------
        dict
            Style of the component
        """
        app = self.app
        components_to_init = dict([(graph, 'graph') for graph in self.layout.components['graph'].keys()])
        components_to_init['dataset'] = 'table'
        for component_id, component_type in components_to_init.items():
            component_property = 'style' if component_type == "graph" else "style_table"

            @app.callback(
                [
                    Output(f'card_{component_id}', 'style'),
                    Output(f'{component_id}', component_property),
                ],
                [
                    Input(f'ember_{component_id}', 'n_clicks'),
                    Input(f'ember_{component_id}', 'data-component-type'),
                    Input(f'ember_{component_id}', 'data-component-id')
                ]
            )
            def ember(click,
                      data_component_type,
                      data_component_id):
                """
                Function used to set style of cards and components.
                Prediction picking graph style is different than the other
                graph because it is placed in a tab.
                ---------------------------------------------------------------
                click: click on zoom button
                data_component_type: component type
                data_component_id: component id
                --------------------------------------------------------------
                return style of cards and style of components
                """
                click = 2 if click is None else click
                toggle_on = True if click % 2 == 0 else False
                if toggle_on:
                    # Style for graph
                    style_component = {
                        'height': '21.6rem'
                    }
                    this_style_card = {
                        'height': '22rem', 'zIndex': 900,
                    }
                    # Style for prediction picking graph
                    if data_component_id == 'prediction_picking':
                        style_component = {
                            'height': '20.6rem',
                        }
                        this_style_card = {
                            'height': '20.8rem', 'zIndex': 901,
                        }
                    # Style for the Dataset
                    if data_component_type == 'table':
                        style_component = {
                            'maxHeight': '23rem',
                        }
                        this_style_card = {
                            'height': '24.1rem', 'zIndex': 900,
                        }
                    return this_style_card, style_component

                else:
                    # Style when zoom button is clicked
                    this_style_card = {
                        'height': '70vh',
                        'width': 'auto',
                        'zIndex': 998,
                        'position': 'fixed', 'top': '55px',
                        'bottom': 0, 'left': 0, 'right': 0,
                    }
                    style_component = {
                        'height': '89vh', 'maxHeight': '89vh',
                    }
                    return this_style_card, style_component

    def init_callback_settings(self):
        app = self.app
        self.layout.components['settings']['input_rows']['rows'].value = self.layout.settings['rows']
        self.layout.components['settings']['input_points']['points'].value = self.layout.settings['points']
        self.layout.components['settings']['input_features']['features'].value = self.layout.settings['features']
        self.layout.components['settings']['input_violin']['violin'].value = self.layout.settings['violin']

        for id in self.layout.settings.keys():
            @app.callback(
                [Output(f'{id}', 'valid'),
                 Output(f'{id}', 'invalid')],
                [Input(f'{id}', "value")]
            )
            def update_valid(value):
                """
                actualise valid and invalid icon in input component
                Parameters
                ----------
                value : int
                    value of input component
                Returns
                -------
                    tuple of boolean
                """
                patt = re.compile('^[0-9]*[1-9][0-9]*$')
                if patt.match(str(value)):
                    return True, False
                else:
                    return False, True

        @app.callback(
            Output("modal", "is_open"),
            [
                Input("settings", "n_clicks"),
                Input("apply", "n_clicks")],
            [
                State('rows', 'valid'),
                State('points', 'valid'),
                State('features', 'valid'),
                State('violin', 'valid'),
            ],
        )
        def toggle_modal(n1,
                         n2,
                         rows,
                         points,
                         features,
                         violin):
            """
            open modal /close modal (only if all input are valid)
            """
            ctx = dash.callback_context
            if ctx.triggered[0]['prop_id'] == 'settings.n_clicks':
                if n1 is not None:
                    return True
            else:
                if n2 is not None:
                    if all([rows, points, features, violin]):
                        return False
                    else:
                        return True
            return False

    def callback_generator(self):
        app = self.app

        @app.callback(
            [
                Output('dataset', 'data'),
                Output('dataset', 'tooltip_data'),
                Output('dataset', 'columns'),
                Output('dataset', 'active_cell'),
            ],
            [
                Input('prediction_picking', 'selectedData'),
                Input('modal', 'is_open'),
                Input('apply_filter', 'n_clicks'),
                Input('reset_dropdown_button', 'n_clicks'),
                Input({'type': 'del_dropdown_button', 'index': ALL}, 'n_clicks')
            ],
            [
                State('rows', 'value'),
                State('name', 'value'),
                State({'type': 'var_dropdown', 'index': ALL}, 'value'),
                State({'type': 'var_dropdown', 'index': ALL}, 'id'),
                State({'type': 'dynamic-str', 'index': ALL}, 'value'),
                State({'type': 'dynamic-str', 'index': ALL}, 'id'),
                State({'type': 'dynamic-bool', 'index': ALL}, 'value'),
                State({'type': 'dynamic-bool', 'index': ALL}, 'id'),
                State({'type': 'dynamic-date', 'index': ALL}, 'start_date'),
                State({'type': 'dynamic-date', 'index': ALL}, 'end_date'),
                State({'type': 'dynamic-date', 'index': ALL}, 'id'),
                State({'type': 'lower', 'index': ALL}, 'value'),
                State({'type': 'lower', 'index': ALL}, 'id'),
                State({'type': 'upper', 'index': ALL}, 'value'),
                State({'type': 'upper', 'index': ALL}, 'id'),
                State('dropdowns_container', 'children')
            ]
        )
        def update_datatable(selected_data,
                             is_open,
                             nclicks_apply,
                             nclicks_reset,
                             nclicks_del,
                             rows,
                             name,
                             val_feature,
                             id_feature,
                             val_str_modality,
                             id_str_modality,
                             val_bool_modality,
                             id_bool_modality,
                             start_date,
                             end_date,
                             id_date,
                             val_lower_modality,
                             id_lower_modality,
                             val_upper_modality,
                             id_upper_modality,
                             children):
            """
            This function is used to update the datatable according to sorting,
            filtering and settings modifications.
            ------------------------------------------------------------------
            selected_data: selected data in prediction picking graph
            is_open: modal
            nclicks_apply: click on Apply Filter button
            nclicks_reset: click on Reset All Filter button
            nclicks_del: click on delete button
            rows: number of rows for subset
            name: name for features name
            val_feature: feature selected to filter
            id_feature: id of feature selected to filter
            val_str_modality: string modalities selected
            id_str_modality: id of string modalities selected
            val_bool_modality: boolean modalities selected
            id_bool_modality: id of boolean modalities selected
            start_date: start dates selected
            end_date: end dates selected
            id_date: id of dates selected
            val_lower_modality: lower values of numeric filter
            id_lower_modality: id of lower modalities of numeric filter
            val_upper_modality: upper values of numeric filter
            id_upper_modality: id of upper values of numeric filter
            children: children of dropdown container
            ------------------------------------------------------------------
            return
            data: available dataset
            tooltip_data: tooltip of the dataset
            columns: columns of the dataset
            active_cell: activated cell
            """
            ctx = dash.callback_context
            active_cell = no_update
            df = self.layout.round_dataframe
            columns = self.layout.components['table']['dataset'].columns
            
            data = self.layout.components['table']['dataset'].data = df.to_dict('records')
            tooltip_data = self.layout.components['table']['dataset'].tooltip_data

            if ctx.triggered[0]['prop_id'] == 'modal.is_open':
                if is_open:
                    raise PreventUpdate
                else:
                    #self.settings['rows'] = rows
                    #self.layout.init_data()
                    active_cell = {'row': 0, 'column': 0, 'column_id': '_index_'}
                    #self.settings_ini['rows'] = self.settings['rows']
                    if name == [1]:
                        columns = [
                            {"name": '_index_', "id": '_index_'},
                            {"name": '_predict_', "id": '_predict_'}] + \
                            [{"name": self.explainer.features_dict[i], "id": i} for i in self.explainer.x_init]
            elif ((ctx.triggered[0]['prop_id'] == 'prediction_picking.selectedData') and
                  (selected_data is not None)):
                row_ids = []
                # If some data have been selected in prediction picking graph
                if selected_data is not None and len(selected_data) > 1:
                    for p in selected_data['points']:
                        row_ids.append(p['customdata'])
                    df = self.layout.round_dataframe.loc[row_ids]
                else:
                    df = self.layout.round_dataframe
            # If click on reset button
            elif ctx.triggered[0]['prop_id'] == 'reset_dropdown_button.n_clicks':
                df = self.layout.round_dataframe
            # If click on Apply filter
            elif ((ctx.triggered[0]['prop_id'] == 'apply_filter.n_clicks') | (
                    (ctx.triggered[0]['prop_id'] == 'prediction_picking.selectedData') and
                  (selected_data is None))):
                # get list of ID
                feature_id = [id_feature[i]['index'] for i in range(len(id_feature))]
                str_id = [id_str_modality[i]['index'] for i in range(len(id_str_modality))]
                bool_id = [id_bool_modality[i]['index'] for i in range(len(id_bool_modality))]
                lower_id = [id_lower_modality[i]['index'] for i in range(len(id_lower_modality))]
                date_id = [id_date[i]['index'] for i in range(len(id_date))]
                df = self.layout.round_dataframe
                # If there is some filters
                if len(feature_id) > 0:
                    for i in range(len(feature_id)):
                        # String filter
                        if feature_id[i] in str_id:
                            position = np.where(np.array(str_id) == feature_id[i])[0][0]
                            if ((position is not None) & (val_str_modality[position] is not None)):
                                df = df[df[val_feature[i]].isin(val_str_modality[position])]
                            else:
                                df = df
                        # Boolean filter
                        elif feature_id[i] in bool_id:
                            position = np.where(np.array(bool_id) == feature_id[i])[0][0]
                            if ((position is not None) & (val_bool_modality[position] is not None)):
                                df = df[df[val_feature[i]] == val_bool_modality[position]]
                            else:
                                df = df
                        # Date filter
                        elif feature_id[i] in date_id:
                            position = np.where(np.array(date_id) == feature_id[i])[0][0]
                            if((position is not None) &
                               (start_date[position] < end_date[position])):
                                df = df[((df[val_feature[i]] >= start_date[position]) &
                                         (df[val_feature[i]] <= end_date[position]))]
                            else:
                                df = df
                        # Numeric filter
                        elif feature_id[i] in lower_id:
                            position = np.where(np.array(lower_id) == feature_id[i])[0][0]
                            if((position is not None) & (val_lower_modality[position] is not None) &
                               (val_upper_modality[position] is not None)):
                                if (val_lower_modality[position] < val_upper_modality[position]):
                                    df = df[(df[val_feature[i]] >= val_lower_modality[position]) &
                                            (df[val_feature[i]] <= val_upper_modality[position])]
                                else:
                                    df = df
                            else:
                                df = df
                        else:
                            df = df
                else:
                    df = df
                if len(df) == 0:
                    raise ValueError(
                        "Your dataframe is empty. It must have at least one row"
                         )
            elif None not in nclicks_del:
                df = self.layout.round_dataframe
            else:
                raise dash.exceptions.PreventUpdate
            # self.layout.components['table']['dataset'].data = df.to_dict('records')
            data = df.to_dict('records')
            # self.layout.components['table']['dataset'].tooltip_data = [
            tooltip_data = [
                {
                    column: {'value': str(value), 'type': 'text'}
                    for column, value in row.items()
                } for row in df.to_dict('rows')
            ]
            return (
                data,
                tooltip_data,
                # self.layout.components['table']['dataset'].data,
                # self.layout.components['table']['dataset'].tooltip_data,
                columns,
                active_cell,
            )

        @app.callback(
            [
                Output('global_feature_importance', 'figure'),
                Output('global_feature_importance', 'clickData')
            ],
            [
                Input('select_label', 'value'),
                Input('dataset', 'data'),
                Input('prediction_picking', 'selectedData'),
                Input('apply_filter', 'n_clicks'),
                Input('reset_dropdown_button', 'n_clicks'),
                Input({'type': 'del_dropdown_button', 'index': ALL}, 'n_clicks'),
                Input('modal', 'is_open'),
                Input('card_global_feature_importance', 'n_clicks'),
                Input('bool_groups', 'on'),
                Input('ember_global_feature_importance', 'n_clicks')
            ],
            [
                State('global_feature_importance', 'clickData'),
                State('global_feature_importance', 'figure'),
                State('features', 'value')
            ]
        )
        def update_feature_importance(label,
                                      data,
                                      selected_data,
                                      apply_filters,
                                      reset_filter,
                                      nclicks_del,
                                      is_open,
                                      n_clicks,
                                      bool_group,
                                      click_zoom,
                                      clickData,
                                      figure,
                                      features):
            """
            update feature importance plot according label, click on graph,
            filters applied and subset selected in prediction picking graph.
            ------------------------------------------------------------
            label: label of data
            data: dataset
            selected_data : data selected on prediction picking graph
            apply_filters: click on apply filter button
            reset_filter: click on reset filter button
            nclicks_del: click on del button
            is_open: modal
            n_clicks: click on features importance card
            bool_group: display groups
            click_zoom: click on zoom button
            clickData: click on features importance graph
            features: features value
            -------------------------------------------------------------
            return
            figure of Features Importance graph
            click on Features Importance graph
            """
            ctx = dash.callback_context
            # Zoom is False by Default. It becomes True if we click on it
            click = 2 if click_zoom is None else click_zoom
            if click % 2 == 0:
                zoom_active = False
            else:
                zoom_active = True
            selection = None
            list_index = self.layout.list_index
            #graph_gfi = self.layout.components['graph']['global_feature_importance']
            selected_feature = self.explainer.inv_features_dict.get(
                clickData['points'][0]['label'].replace('<b>', '').replace('</b>', '')
            ) if clickData else None
            if ctx.triggered[0]['prop_id'] == 'modal.is_open':
                if is_open:
                    raise PreventUpdate
                #else:
                #    self.settings['features'] = features
                #    self.settings_ini['features'] = self.settings['features']
            elif ctx.triggered[0]['prop_id'] == 'select_label.value':
                #self.layout.label = label
                selection = None
            elif ctx.triggered[0]['prop_id'] == 'dataset.data':
                #self.layout.list_index = [d['_index_'] for d in data]
                list_index = [d['_index_'] for d in data]
            elif ctx.triggered[0]['prop_id'] == 'bool_groups.on':
                clickData = None  # We reset the graph and clicks if we toggle the button
            # If we have selected data on prediction picking graph
            elif ((ctx.triggered[0]['prop_id'] == 'prediction_picking.selectedData') and
                  (selected_data is not None) and (len(selected_data) > 1)):
                row_ids = []
                if selected_data is not None and len(selected_data) > 1:
                    for p in selected_data['points']:
                        row_ids.append(p['customdata'])
                    selection = row_ids
                else:
                    selection = None
                #when group
                if self.explainer.features_groups and bool_group:
                    list_sub_features = [f for group_features in self.explainer.features_groups.values()
                                      for f in group_features]
                    if selected_feature not in list_sub_features:
                        selected_feature = None
            # If click on a single point on prediction picking, do nothing
            elif ((ctx.triggered[0]['prop_id'] == 'prediction_picking.selectedData') and
                  (selected_data is not None) and (len(selected_data) == 1)):
                  pass
            # If we have dubble click on prediction picking to remove the selected subset
            elif ((ctx.triggered[0]['prop_id'] == 'prediction_picking.selectedData') and
                  (selected_data is None)):
                # If there is some filters applied
                #if (len([d['_index_'] for d in data]) != len(self.layout.list_index)):
                if (len([d['_index_'] for d in data]) != len(list_index)):
                    selection = [d['_index_'] for d in data]
                else:
                    selection = None
                #when group
                if self.explainer.features_groups and bool_group:
                    list_sub_features = [f for group_features in self.explainer.features_groups.values()
                                      for f in group_features]
                    if selected_feature not in list_sub_features:
                        selected_feature = None
            # If we click on reset filter button
            elif ctx.triggered[0]['prop_id'] == 'reset_dropdown_button.n_clicks':
                selection = None
                #when group
                if self.explainer.features_groups and bool_group:
                    list_sub_features = [f for group_features in self.explainer.features_groups.values()
                                      for f in group_features]
                    if selected_feature not in list_sub_features:
                        selected_feature = None
            # If we click on Apply button
            elif ctx.triggered[0]['prop_id'] == 'apply_filter.n_clicks':
                selection = [d['_index_'] for d in data]
                #when group
                if self.explainer.features_groups and bool_group:
                    list_sub_features = [f for group_features in self.explainer.features_groups.values()
                                      for f in group_features]
                    if selected_feature not in list_sub_features:
                        selected_feature = None
            # If we click on the last del button
            elif (('del_dropdown_button' in ctx.triggered[0]['prop_id']) &
                  (None not in nclicks_del)):
                selection = None
                #when group
                if self.explainer.features_groups and bool_group:
                    list_sub_features = [f for group_features in self.explainer.features_groups.values()
                                      for f in group_features]
                    if selected_feature not in list_sub_features:
                        selected_feature = None
            #FIXME: self.explainer.features_groups ?
            elif (ctx.triggered[0]['prop_id'] == 'card_global_feature_importance.n_clicks'
                  and self.explainer.features_groups 
                  and bool_group):
                row_ids = []
                if selected_data is not None and len(selected_data) > 1:
                    # we plot prediction picking subset
                    for p in selected_data['points']:
                        row_ids.append(p['customdata'])
                    selection = row_ids
                #elif (len([d['_index_'] for d in data]) != len(self.layout.list_index)):
                elif (len([d['_index_'] for d in data]) != len(list_index)):
                    selection = [d['_index_'] for d in data]
                else:
                    selection = None
                # When we click twice on the same bar this will reset the graph
                if self.layout.last_click_data == clickData:
                    selected_feature = None
                list_sub_features = [f for group_features in self.explainer.features_groups.values()
                                      for f in group_features]
                if selected_feature in list_sub_features:
                    self.layout.last_click_data = clickData
                    raise PreventUpdate
            else:
                # Zoom management to generate graph which have global axis
                # if len(self.layout.components['graph']['global_feature_importance'].figure['data']) == 1:
                if len(figure['data']) == 1:
                    selection = None
                else:
                    row_ids = []
                    if selected_data is not None and len(selected_data) > 1:
                        # we plot prediction picking subset
                        for p in selected_data['points']:
                            row_ids.append(p['customdata'])
                        selection = row_ids
                    else:
                        # we plot filter subset
                        selection = [d['_index_'] for d in data]
                self.layout.last_click_data = clickData

            group_name = selected_feature if (self.explainer.features_groups is not None
                                              and selected_feature in self.explainer.features_groups.keys()) else None

            # self.layout.components['graph']['global_feature_importance'].figure = \
            figure = \
                self.explainer.plot.features_importance(
                    max_features=features,
                    selection=selection,
                    label=label,
                    group_name=group_name,
                    display_groups=bool_group,
                    zoom=zoom_active
                )
            # Adjust graph with adding x axis title
            # self.layout.components['graph']['global_feature_importance'].adjust_graph(x_ax='Contribution')
            figure = ShapashGraph.adjust_graph_static(figure, x_ax='Contribution')
            # self.layout.components['graph']['global_feature_importance'].figure.layout.clickmode = 'event+select'
            figure.layout.clickmode = 'event+select'
            if selected_feature:
                if self.explainer.features_groups is None:
                    # self.select_point('global_feature_importance', clickData)
                    self.select_point(figure, clickData)
                elif selected_feature not in self.explainer.features_groups.keys():
                    # self.select_point('global_feature_importance', clickData)
                    self.select_point(figure, clickData)

            # font size can be adapted to screen size
            # nb_car = max([len(self.layout.components['graph']['global_feature_importance'].figure.data[0].y[i]) for i in
            #               range(len(self.layout.components['graph']['global_feature_importance'].figure.data[0].y))])
            # self.layout.components['graph']['global_feature_importance'].figure.update_layout(
            #     yaxis=dict(tickfont={'size': min(round(500 / nb_car), 12)})
            # )

            nb_car = max([len(figure.data[0].y[i]) for i in
                          range(len(figure.data[0].y))])
            figure.update_layout(
                yaxis=dict(tickfont={'size': min(round(500 / nb_car), 12)})
            )

            self.layout.last_click_data = clickData
            return figure, clickData

        @app.callback(
            Output(component_id='feature_selector', component_property='figure'),
            [
                Input('global_feature_importance', 'clickData'),
                Input('prediction_picking', 'selectedData'),
                Input('dataset', 'data'),
                Input('apply_filter', 'n_clicks'),
                Input('reset_dropdown_button', 'n_clicks'),
                Input({'type': 'del_dropdown_button', 'index': ALL}, 'n_clicks'),
                Input('select_label', 'value'),
                Input('modal', 'is_open'),
                Input('ember_feature_selector', 'n_clicks')
            ],
            [
                State('points', 'value'),
                State('violin', 'value'),
                State('global_feature_importance', 'figure'),
                State('feature_selector', 'fs_figure')
            ]
        )
        def update_feature_selector(feature,
                                    selected_data,
                                    data,
                                    apply_filters,
                                    reset_filter,
                                    nclicks_del,
                                    label,
                                    is_open,
                                    click_zoom,
                                    points,
                                    violin,
                                    figure,
                                    fs_figure):
            """
            Update feature plot according to label, data,
            selected feature on features importance graph,
            filters and settings modifications
            --------------------------------------------
            feature: click on feature importance graph
            selected_data: Data selected on prediction picking graph
            data: dataset
            apply_filters: click on apply filter button
            reset_filter: click on reset filter button
            nclicks_del: click del button
            label: selected label
            is_open: modal
            click_zoom: click on zoom button
            points: points value in setting
            violin: violin value in setting
            ---------------------------------------------
            return
            figure: feature selector graph
            """
            # Zoom is False by Default. It becomes True if we click on it
            click = 2 if click_zoom is None else click_zoom
            subset = None
            if click % 2 == 0:
                zoom_active = False
            else:
                zoom_active = True  # To check if zoom is activated
            list_index = self.layout.list_index
            #selected_feature = self.layout.selected_feature
            if feature is not None:
                selected_feature = feature['points'][0]['label'].replace('<b>', '').replace('</b>', '')
            else:
                selected_feature = self.layout.selected_feature
            ctx = dash.callback_context
            if ctx.triggered[0]['prop_id'] == 'modal.is_open':
                if is_open:
                    raise PreventUpdate
                #else:
#                    self.settings['points'] = points
#                    self.settings_ini['points'] = self.settings['points']
#                    self.settings['violin'] = violin
#                    self.settings_ini['violin'] = self.settings['violin']

            #elif ctx.triggered[0]['prop_id'] == 'select_label.value':
            #    self.layout.label = label
            elif ctx.triggered[0]['prop_id'] == 'global_feature_importance.clickData':
                if feature is not None:
                    # Removing bold
                    # self.layout.selected_feature = feature['points'][0]['label'].replace('<b>', '').replace('</b>', '')
                    selected_feature = feature['points'][0]['label'].replace('<b>', '').replace('</b>', '')
                    # if feature['points'][0]['curveNumber'] == 0 and \
                    #           len(self.layout.components['graph']['global_feature_importance'].figure['data']) == 2:
                    if feature['points'][0]['curveNumber'] == 0 and \
                              len(figure['data']) == 2:
                        if selected_data is not None and len(selected_data) > 1:
                            row_ids = []
                            for p in selected_data['points']:
                                row_ids.append(p['customdata'])
                            #self.layout.subset = row_ids
                            subset = row_ids
                        else:
                            # self.layout.subset = [d['_index_'] for d in data]
                            subset = [d['_index_'] for d in data]
                    else:
                        # self.layout.subset = self.layout.list_index
                        subset = self.layout.list_index
            # If we have selected data on prediction picking graph
            elif ((ctx.triggered[0]['prop_id'] == 'prediction_picking.selectedData') and
                  (selected_data is not None)):
                row_ids = []
                if selected_data is not None and len(selected_data) > 1:
                    for p in selected_data['points']:
                        row_ids.append(p['customdata'])
                    # self.layout.subset = row_ids
                    subset = row_ids
            # if we have click on reset button
            elif ctx.triggered[0]['prop_id'] == 'reset_dropdown_button.n_clicks':
                # self.layout.subset = None
                subset = None
            # If we have clik on Apply filter button
            elif ctx.triggered[0]['prop_id'] == 'apply_filter.n_clicks':
                # self.layout.subset = [d['_index_'] for d in data]
                subset = [d['_index_'] for d in data]
            # If we have click on the last del button
            elif (('del_dropdown_button' in ctx.triggered[0]['prop_id']) &
                  (None not in nclicks_del)):
                # self.layout.subset = None
                subset = None
            else:
                if ctx.triggered[0]['prop_id'] == 'dataset.data':
                    list_index = [d['_index_'] for d in data]
                # Zoom management to generate graph which have global axis
                # if len(self.layout.components['graph']['global_feature_importance'].figure['data']) == 1:
                if len(figure['data']) == 1:
                    #self.layout.subset = self.layout.list_index
                    subset = list_index
                # elif len(self.layout.components['graph']['global_feature_importance'].figure['data']) == 2:
                elif len(figure['data']) == 2:
                    if feature['points'][0]['curveNumber'] == 0:
                        if selected_data is not None and len(selected_data) > 1:
                            row_ids = []
                            for p in selected_data['points']:
                                row_ids.append(p['customdata'])
                            # self.layout.subset = row_ids
                            subset = row_ids
                        else:
                            # self.layout.subset = [d['_index_'] for d in data]
                            subset = [d['_index_'] for d in data]
                    else:
                        #self.layout.subset = self.layout.list_index
                        subset = list_index
                else:
                    row_ids = []
                    if selected_data is not None and len(selected_data) > 1:
                        # we plot prediction picking subset
                        for p in selected_data['points']:
                            row_ids.append(p['customdata'])
                        # self.layout.subset = row_ids
                        subset = row_ids
                    else:
                        # we plot filter subset
                        # self.layout.subset = [d['_index_'] for d in data]
                        subset = [d['_index_'] for d in data]

            #graph_fs = self.layout.components['graph']['feature_selector']

            # self.layout.components['graph']['feature_selector'].figure = \
            #graph_fs.figure = \
            fs_figure = \
                self.explainer.plot.contribution_plot(
                    # col=self.layout.selected_feature,
                    col=selected_feature,
                    # selection=self.layout.subset,
                    selection=subset,
                    label=label,
                    violin_maxf=violin,
                    max_points=points,
                    zoom=zoom_active
                )

            # self.layout.components['graph']['feature_selector'].figure['layout'].clickmode = 'event+select'
            fs_figure['layout'].clickmode = 'event+select'
            # Adjust graph with adding x and y axis titles
            # self.layout.components['graph']['feature_selector'].adjust_graph(
            fs_figure = ShapashGraph.adjust_graph_static(fs_figure,
                #x_ax=truncate_str(self.layout.selected_feature, 110),
                x_ax=truncate_str(selected_feature, 110),
                y_ax='Contribution')
            return fs_figure

        @app.callback(
            [
                Output('index_id', 'value'),
                Output("index_id", "n_submit")
            ],
            [
                Input('feature_selector', 'clickData'),
                Input('prediction_picking', 'clickData'),
                Input('dataset', 'active_cell'),
                Input('apply_filter', 'n_clicks'),
                Input('reset_dropdown_button', 'n_clicks'),
                Input({'type': 'del_dropdown_button', 'index': ALL}, 'n_clicks')
            ],
            [
                State('dataset', 'data'),
                State('index_id', 'value')  # Get the current value of the index
            ]
        )
        def update_index_id(click_data,
                            prediction_picking,
                            cell,
                            apply_filters,
                            reset_filter,
                            nclicks_del,
                            data,
                            current_index_id):
            """
            This function is used to update index value according to
            active cell, filters and click data on feature plot or on
            prediction picking graph.
            ----------------------------------------------------------------
            click_data: click on feature selector
            prediction_picking: click on prediction picking graph
            cell: selected sell on dataset
            apply_filters: click on Apply filter button
            reset_filter: click on reset filter button
            nclicks_del: click on del button
            data: dataset
            current_index_id: the current value of the index
            ----------------------------------------------------------------
            return
            selected index id
            boolean n_submit
            """
            ctx = dash.callback_context
            selected = None
            if ctx.triggered[0]['prop_id'] != 'dataset.data':
                if ctx.triggered[0]['prop_id'] == 'feature_selector.clickData':
                    selected = click_data['points'][0]['customdata'][1]
                    self.click_graph = True # FIXME: à quoi sert self.click_graph ?
                elif ctx.triggered[0]['prop_id'] == 'prediction_picking.clickData':
                    selected = prediction_picking['points'][0]['customdata']
                    self.click_graph = True
                elif ctx.triggered[0]['prop_id'] == 'dataset.active_cell':
                    if cell is not None:
                        selected = data[cell['row']]['_index_']
                    else:
                        # Get actual value in field to refresh the selected value
                        selected = current_index_id
                elif ctx.triggered[0]['prop_id'] == '.':
                    selected = data[0]['_index_']
                # If click on Reset apply button
                elif ctx.triggered[0]['prop_id'] == 'reset_dropdown_button.n_clicks':
                    # Get the row index value
                    selected = data[cell['row']]['_index_']
                # If click on Apply filter button
                elif ctx.triggered[0]['prop_id'] == 'apply_filter.n_clicks':
                    # get the first index on the dataset
                    selected = data[0]['_index_']
                # If click on the last del button
                elif (('del_dropdown_button' in ctx.triggered[0]['prop_id']) &
                      (None not in nclicks_del)):
                    # Get the row index value
                    selected = data[cell['row']]['_index_']
                else:
                    selected = data[0]['_index_']
            else:
                raise PreventUpdate
            return selected, True

        @app.callback(
            Output('threshold_label', 'children'),
            [Input('threshold_id', 'value')])
        def update_threshold_label(value):
            """
            update threshold label
            """
            return f'Threshold: {value}'

        @app.callback(
            Output('max_contrib_label', 'children'),
            [Input('max_contrib_id', 'value')])
        def update_max_contrib_label(value):
            """
            update max_contrib label
            """
            #self.layout.components['filter']['max_contrib']['max_contrib_id'].value = value
            return f'Features to display: {value}'

        @app.callback(
            [Output('max_contrib_id', 'value'),
             Output('max_contrib_id', 'max'),
             Output('max_contrib_id', 'marks')
             ],
            [Input('modal', 'is_open')],
            [State('features', 'value'),
            State('max_contrib_id', 'value')]
        )
        def update_max_contrib_id(is_open,
                                  features,
                                  value):
            """
            update max contrib component layout after settings modifications
            """
            ctx = dash.callback_context
            if ctx.triggered[0]['prop_id'] == 'modal.is_open':
                if is_open:
                    raise PreventUpdate
                else:
                    max_value = min(features, len(self.layout.dataframe.columns) - 2)
                    if max_value // 5 == max_value / 5:
                        nb_marks = min(int(max_value // 5), 10)
                    elif max_value // 4 == max_value / 4:
                        nb_marks = min(int(max_value // 4), 10)
                    elif max_value // 3 == max_value / 3:
                        nb_marks = min(int(max_value // 3), 10)
                    elif max_value // 7 == max_value / 7:
                        nb_marks = min(int(max_value // 6), 10)
                    else:
                        nb_marks = 2
                    marks = {f'{round(max_value * feat / nb_marks)}': f'{round(max_value * feat / nb_marks)}'
                             for feat in range(1, nb_marks + 1)}
                    marks['1'] = '1'
                    #if max < self.layout.components['filter']['max_contrib']['max_contrib_id'].value:
                    if max_value < value:
                        value = max_value
                    else:
                        value = no_update

                    return value, max_value, marks

        @app.callback(
            Output(component_id='detail_feature', component_property='figure'),
            [
                Input('threshold_id', 'value'),
                Input('max_contrib_id', 'value'),
                Input('check_id_positive', 'value'),
                Input('check_id_negative', 'value'),
                Input('masked_contrib_id', 'value'),
                Input('select_label', 'value'),
                Input('dataset', 'active_cell'),
                Input('feature_selector', 'clickData'),
                Input('prediction_picking', 'clickData'),
                Input("validation", "n_clicks"),
                Input('bool_groups', 'on'),
                Input('ember_detail_feature', 'n_clicks'),
            ],
            [
                State('index_id', 'value'),
                State('dataset', 'data'),
                State('detail_feature', 'figure')
            ]
        )
        def update_detail_feature(threshold,
                                  max_contrib,
                                  positive,
                                  negative,
                                  masked,
                                  label,
                                  cell,
                                  click_data,
                                  prediction_picking,
                                  validation_click,
                                  bool_group,
                                  click_zoom,
                                  index,
                                  data,
                                  figure):
            """
            update local explanation plot according to app changes.
            -------------------------------------------------------
            threshold: threshold
            max_contrib: max contribution
            positive: boolean
            negative: boolean
            masked: feature(s) to mask
            label: label
            cell: selected cell
            click_data: click on feature selector graph
            prediction_picking: click on prediction picking graph
            validation_click: click on validation
            bool_group: boolean
            click_zoom: click on zoom button
            index: selected index
            data: the dataset
            --------------------------------------------------------
            return
            detail feature graph
            """
            # Zoom is False by Default. It becomes True if we click on it
            click = 2 if click_zoom is None else click_zoom
            if click % 2 == 0:
                zoom_active = False
            else:
                zoom_active = True
            ctx = dash.callback_context
            selected = None
            if ctx.triggered[0]['prop_id'] == 'feature_selector.clickData':
                selected = click_data['points'][0]['customdata'][1]
            elif ctx.triggered[0]['prop_id'] == 'prediction_picking.clickData':
                selected = prediction_picking['points'][0]['customdata']
            elif ctx.triggered[0]['prop_id'] in ['threshold_id.value', 'validation.n_clicks']:
                selected = index
            elif ctx.triggered[0]['prop_id'] == 'dataset.active_cell':
                if cell:
                    selected = data[cell['row']]['_index_']
                else:
                    zoom_active = zoom_active
                    # raise PreventUpdate
            else:
                selected = index
            threshold = threshold if threshold != 0 else None
            if positive == [1]:
                sign = (None if negative == [1] else True)
            else:
                sign = (False if negative == [1] else None)
            self.explainer.filter(threshold=threshold,
                                  features_to_hide=masked,
                                  positive=sign,
                                  max_contrib=max_contrib,
                                  display_groups=bool_group)
            if np.issubdtype(type(self.explainer.x_init.index[0]), np.dtype(int).type):
                selected = int(selected)
            #graph_df = self.layout.components['graph']['detail_feature']
            # self.layout.components['graph']['detail_feature'].figure = self.explainer.plot.local_plot(
            figure = self.explainer.plot.local_plot(
                index=selected,
                label=label,
                show_masked=True,
                yaxis_max_label=8,
                display_groups=bool_group,
                zoom=zoom_active
            )
            # Adjust graph with adding x axis titles
            # self.layout.components['graph']['detail_feature'].adjust_graph(x_ax='Contribution')
            figure = ShapashGraph.adjust_graph_static(figure, x_ax='Contribution')
            # font size can be adapted to screen size
            # list_yaxis = [self.layout.components['graph']['detail_feature'].figure.data[i].y[0] for i in
            #               range(len(self.layout.components['graph']['detail_feature'].figure.data))]
            list_yaxis = [figure.data[i].y[0] for i in
                          range(len(figure.data))]
            # exclude new line with labels of y axis
            list_yaxis = [x.split('<br />')[0] for x in list_yaxis]
            nb_car = max([len(x) for x in list_yaxis])
            # self.layout.components['graph']['detail_feature'].figure.update_layout(
            #     yaxis=dict(tickfont={'size': min(round(500 / nb_car), 12)})
            # )
            figure.update_layout(
                yaxis=dict(tickfont={'size': min(round(500 / nb_car), 12)})
            )
            # return self.layout.components['graph']['detail_feature'].figure
            return figure

        @app.callback(
            Output("validation", "n_clicks"),
            [
                Input("index_id", "n_submit")
            ],
        )
        def click_validation(n_submit):
            """
            submit index selection
            """
            if n_submit:
                return 1
            else:
                raise PreventUpdate

        @app.callback(
            [
                Output('dataset', 'style_data_conditional'),
                Output('dataset', 'style_filter_conditional'),
                Output('dataset', 'style_header_conditional'),
                Output('dataset', 'style_cell_conditional'),
            ],
            [
                Input("validation", "n_clicks")
            ],
            [
                State('dataset', 'data'),
                State('index_id', 'value')
            ]
        )
        def datatable_layout(validation,
                             data,
                             index):
            ctx = dash.callback_context
            if ctx.triggered[0]['prop_id'] == 'validation.n_clicks' and validation is not None:
                pass
            else:
                raise PreventUpdate

            style_data_conditional = [
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                }
            ]
            style_filter_conditional = []
            style_header_conditional = [
                {'if': {'column_id': c}, 'fontWeight': 'bold'}
                for c in ['_index_', '_predict_']
            ]
            style_cell_conditional = [
                {'if': {'column_id': c},
                 'width': '70px', 'fontWeight': 'bold'} for c in ['_index_', '_predict_']
            ]

            selected = check_row(data, index)
            if selected is not None:
                style_data_conditional += [{"if": {"row_index": selected}, "backgroundColor": self.layout.color[0]}]

            return style_data_conditional, style_filter_conditional, style_header_conditional, style_cell_conditional

        @app.callback(
            Output(component_id='prediction_picking', component_property='figure'),
            [
                Input('global_feature_importance', 'clickData'),
                Input('dataset', 'data'),
                Input('apply_filter', 'n_clicks'),
                Input('reset_dropdown_button', 'n_clicks'),
                Input({'type': 'del_dropdown_button', 'index': ALL}, 'n_clicks'),
                Input('select_label', 'value'),
                Input('modal', 'is_open'),
                Input('ember_prediction_picking', 'n_clicks')
            ],
            [
                State('points', 'value'),
                State('violin', 'value'),
                State('prediction_picking','figure')
            ]
        )
        def update_prediction_picking(feature,
                                      data,
                                      apply_filters,
                                      reset_filter,
                                      nclicks_del,
                                      label,
                                      is_open,
                                      click_zoom,
                                      points,
                                      violin,
                                      pp_figure):
            """
            Update feature plot according to label, data,
            selected feature and settings modifications
            ------------------------------------------------
            feature: click on features importance graph
            data: the dataset
            apply_filters: click on apply filter button
            reset_filter: click on reset filter button
            nclicks_del: click on del button
            label: selected label
            is_open: modal
            click_zoom: click on zoom button
            points: number of points
            violin: number of violin plot
            -------------------------------------------------
            return
            prediction picking graph
            """
            ctx = dash.callback_context
            # Filter subset
            subset = None
            if not ctx.triggered:
                raise dash.exceptions.PreventUpdate
            if ctx.triggered[0]['prop_id'] == 'modal.is_open':
                if is_open:
                    raise PreventUpdate
                #else:
                #    self.settings['points'] = points
                #    self.settings_ini['points'] = self.settings['points']
                #    self.settings['violin'] = violin
                #    self.settings_ini['violin'] = self.settings['violin']
            elif ctx.triggered[0]['prop_id'] == 'select_label.value':
                #self.layout.label = label
                # self.layout.subset = None
                subset = None
            # If we have clicked on reset button
            elif ctx.triggered[0]['prop_id'] == 'reset_dropdown_button.n_clicks':
                # self.layout.subset = None
                subset = None
            # If we have clicked on Apply filter button
            elif ctx.triggered[0]['prop_id'] == 'apply_filter.n_clicks':
                # self.layout.subset = [d['_index_'] for d in data]
                subset = [d['_index_'] for d in data]
            # If we have clicked on the last delete button (X)
            elif (('del_dropdown_button' in ctx.triggered[0]['prop_id']) &
                  (None not in nclicks_del)):
                # self.layout.subset = None
                subset = None
            else:
                raise PreventUpdate

            #graph = self.layout.components['graph']['prediction_picking']
            if self.explainer.y_target is not None:
                # self.layout.components['graph']['prediction_picking'].figure = self.explainer.plot.scatter_plot_prediction(
                pp_figure = self.explainer.plot.scatter_plot_prediction(
                    # selection=self.layout.subset,
                    selection=subset,
                    max_points=points,
                    label=label
                )

                # self.layout.components['graph']['prediction_picking'].figure['layout'].clickmode = 'event+select'
                pp_figure['layout'].clickmode = 'event+select'
                # Adjust graph with adding x and y axis titles
                # self.layout.components['graph']['prediction_picking'].adjust_graph(
                pp_figure = ShapashGraph.adjust_graph_static(pp_figure,
                    x_ax="True Values",
                    y_ax="Predicted Values")
            else:
                figure = go.Figure()
                figure.update_layout(
                xaxis =  { "visible": False },
                yaxis = { "visible": False },
                annotations = [
                    {
                        "text": "Provide the y_target argument in the compile() method to display this plot.",
                        "xref": "paper",
                        "yref": "paper",
                        "showarrow": False,
                        "font": {
                            "size": 14
                        }
                    }
                ]
            )
                # self.layout.components['graph']['prediction_picking'].figure = fig
                pp_figure = figure

            # return self.layout.components['graph']['prediction_picking'].figure
            return pp_figure

        @app.callback(
            Output("modal_feature_importance", "is_open"),
            [Input("open_feature_importance", "n_clicks"),
             Input("close_feature_importance", "n_clicks")],
            [State("modal_feature_importance", "is_open")],
        )
        def toggle_modal_feature_importancet(n1,
                                             n2,
                                             is_open):
            """
            Function used to open and close modal explication when we click
            on "?" button on feature_importance graph
            ---------------------------------------------------------------
            n1: click on "?" button
            n2: click on close button in modal
            ---------------------------------------------------------------
            return modal
            """
            if n1 or n2:
                return not is_open
            return is_open

        @app.callback(
            Output("modal_feature_selector", "is_open"),
            [Input("open_feature_selector", "n_clicks"),
             Input("close_feature_selector", "n_clicks")],
            [State("modal_feature_selector", "is_open")],
        )
        def toggle_modal_feature_selector(n1,
                                          n2,
                                          is_open):
            """
            Function used to open and close modal explication when we click
            on "?" button on feature_selector graph
            ---------------------------------------------------------------
            n1: click on "?" button
            n2: click on close button in modal
            ---------------------------------------------------------------
            return modal
            """
            if n1 or n2:
                return not is_open
            return is_open

        @app.callback(
            Output("modal_detail_feature", "is_open"),
            [Input("open_detail_feature", "n_clicks"),
             Input("close_detail_feature", "n_clicks")],
            [State("modal_detail_feature", "is_open")],
        )
        def toggle_modal_detail_feature(n1,
                                        n2,
                                        is_open):
            """
            Function used to open and close modal explication when we click
            on "?" button on detail_feature graph
            ---------------------------------------------------------------
            n1: click on "?" button
            n2: click on close button in modal
            ---------------------------------------------------------------
            return modal
            """
            if n1 or n2:
                return not is_open
            return is_open

        @app.callback(
            Output("modal_prediction_picking", "is_open"),
            [Input("open_prediction_picking", "n_clicks"),
             Input("close_prediction_picking", "n_clicks")],
            [State("modal_prediction_picking", "is_open")],
        )
        def toggle_modal_prediction_picking(n1,
                                            n2,
                                            is_open):
            """
            Function used to open and close modal explication when we click
            on "?" button on prediction_picking graph
            ---------------------------------------------------------------
            n1: click on "?" button
            n2: click on close button in modal
            ---------------------------------------------------------------
            return modal
            """
            if n1 or n2:
                return not is_open
            return is_open

        @app.callback(
            Output("modal_filter", "is_open"),
            [Input("open_filter", "n_clicks"),
             Input("close_filter", "n_clicks")],
            [State("modal_filter", "is_open")],
        )
        def toggle_modal_filters(n1,
                                 n2,
                                 is_open):
            """
            Function used to open and close modal explication when we click
            on "?" button on Dataset Filters Tab
            ---------------------------------------------------------------
            n1: click on "?" button
            n2: click on close button in modal
            ---------------------------------------------------------------
            return modal
            """
            if n1 or n2:
                return not is_open
            return is_open

        # Add or remove plot blocs in the 'dropdowns_container'
        @app.callback(
            Output('dropdowns_container', 'children'),
            [
                Input('add_dropdown_button', 'n_clicks'),
                Input('reset_dropdown_button', 'n_clicks'),
                Input({'type': 'del_dropdown_button', 'index': ALL}, 'n_clicks')
            ],
            [
                 State('dropdowns_container', 'children'),
                 State('name', 'value')
            ]
        )
        def layout_filter(n_clicks_add,
                          n_clicks_rm,
                          n_clicks_reset,
                          currents_filters,
                          name
                          ):
            """
            Function used to create filter blocs in the dropdowns_container.
            Each bloc will contains:
                -label
                -dropdown button to select feature to filter
                -div which will contains modalities
                -delete button
            ---------------------------------------------------------------
            n_clicks_add: click on add filter
            n_clicks_reset: click on reset filter button
            n_click_del: click on delete button
            children: information on dropdown container
            name: name for feature name
            ---------------------------------------------------------------
            return
                filter blocs
            """
            # Context and init handling (no action)
            ctx = dash.callback_context
            if not ctx.triggered:
                raise dash.exceptions.PreventUpdate
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]

            # We use domain name for feature name
            dict_name = [self.explainer.features_dict[i]
                         for i in self.layout.dataframe.drop(['_index_', '_predict_'], axis=1).columns]
            dict_id = [i for i in self.layout.dataframe.drop(['_index_', '_predict_'], axis=1).columns]
            # Create dataframe to sort it by feature_name
            df_feature_name = pd.DataFrame({'feature_name': dict_name,
                                            'feature_id': dict_id})
            df_feature_name = df_feature_name.sort_values(
                by='feature_name').reset_index(drop=True)
            # Options are sorted by feature_name
            options = [
                {"label": '_index_', "value": '_index_'},
                {"label": '_predict_', "value": '_predict_'}] + \
                [{"label": df_feature_name.loc[i, 'feature_name'],
                  "value": df_feature_name.loc[i, 'feature_id']}
                 for i in range(len(df_feature_name))]

            # Creation of a new graph
            if button_id == 'add_dropdown_button':
                # ID index definition
                if n_clicks_add is None:
                    index_id = 0
                else:
                    index_id = n_clicks_add
                # Appending a dropdown block to 'dropdowns_container'children
                subset_filter = html.Div(
                    id={'type': 'bloc_div',
                        'index': index_id},
                    children=[
                        html.Div([
                            html.Br(),
                            # div which will contains label
                            html.Div(
                                    id={'type': 'dynamic-output-label',
                                        'index': index_id},
                                    )
                            ]),
                        html.Div([
                            # div with dopdown button to select feature to filter
                            html.Div(dcc.Dropdown(
                                id={'type': 'var_dropdown',
                                    'index': index_id},
                                options=options,
                                placeholder="Variable"
                            ), style={"width": "30%"}),
                            # div which will contains modalities
                            html.Div(
                                 id={'type': 'dynamic-output',
                                     'index': index_id},
                                 style={"width": "50%"}
                                ),
                            # Button to delete bloc
                            dbc.Button(
                                id={'type': 'del_dropdown_button',
                                    'index': index_id},
                                children='X',
                                color='warning',
                                size='sm'
                            )
                        ], style={'display': 'flex'})
                    ]
                )
                return currents_filters + [subset_filter]
            # Removal of all existing filters
            elif button_id == 'reset_dropdown_button':
                return [html.Div(
                    id={'type': 'bloc_div',
                        'index': 0},
                    children=[])]
            # Removal of an existing filter
            else:
                filter_id_to_remove = eval(button_id)['index']
                return [gr for gr in currents_filters
                        if gr['props']['id']['index'] != filter_id_to_remove]

        @app.callback(
            Output('reset_dropdown_button', 'disabled'),
            [Input('add_dropdown_button', 'n_clicks'),
             Input('reset_dropdown_button', 'n_clicks'),
             Input({'type': 'del_dropdown_button', 'index': ALL}, 'n_clicks')]
        )
        def update_disabled_reset_button(n_click_add,
                                         n_click_reset,
                                         n_click_del):
            """
            Function used to disabled or not the reset filter button.
            This button is disabled if there is no filter added.
            ---------------------------------------------------------------
            n_click_add: click on add filter button
            n_click_reset: click on reset filter button
            n_click_del: click on delete button
            ---------------------------------------------------------------
            return disabled style
            """
            ctx = dash.callback_context
            if ctx.triggered[0]['prop_id'] == 'add_dropdown_button.n_clicks':
                disabled = False
            elif ctx.triggered[0]['prop_id'] == 'reset_dropdown_button.n_clicks':
                disabled = True
            elif None not in n_click_del:
                disabled = True
            else:
                disabled = False
            return disabled

        @app.callback(
            Output('apply_filter', 'style'),
            [Input('add_dropdown_button', 'n_clicks'),
             Input('reset_dropdown_button', 'n_clicks'),
             Input({'type': 'del_dropdown_button', 'index': ALL}, 'n_clicks')]
        )
        def update_style_apply_filter_button(n_click_add,
                                             n_click_reset,
                                             n_click_del):
            """
            Function used to display or not the apply filter button.
            This button is only display if almost one filter was added.
            ---------------------------------------------------------------
            n_click_add: click on add filter button
            n_click_reset: click on reset filter button
            n_click_del: click on delete button
            ---------------------------------------------------------------
            return style of apply filter button
            """
            ctx = dash.callback_context
            if ctx.triggered[0]['prop_id'] == 'add_dropdown_button.n_clicks':
                return {'display': 'block'}
            elif ctx.triggered[0]['prop_id'] == 'reset_dropdown_button.n_clicks':
                return {'display': 'none'}
            elif None not in n_click_del:
                return {'display': 'none'}
            else:
                return {'display': 'block'}

        @app.callback(
            Output({'type': 'dynamic-output-label', 'index': MATCH}, 'children'),
            Input({'type': 'var_dropdown', 'index': MATCH}, 'value'),
        )
        def update_label_filter(value):
            """
            Function used to add label to the filters. Label is updated
            when value is not None
            ---------------------------------------------------------------
            value: value selected on the var dropdown button
            ---------------------------------------------------------------
            return label
            """
            if value is not None:
                return html.Label("Variable {} is filtered".format(value))
            else:
                return html.Label('Select variable to filter')

        @app.callback(
            Output({'type': 'dynamic-output', 'index': MATCH}, 'children'),
            [Input({'type': 'var_dropdown', 'index': MATCH}, 'value'),
             Input({'type': 'var_dropdown', 'index': MATCH}, 'id'),
             Input('add_dropdown_button', 'n_clicks')],
        )
        def display_output(value,
                           id,
                           add_click):
            """
            Function used to create modalities choices. Componenents are different
            according to the type of the selected variable.
            For string variable: component is a dropdown button
            For boolean variable: component is a RadioItems button
            For Integer variable that have less than 20 modalities: component
            is a dropdown button.
            For date variable: component is a DatePickerRange
            Else: components are lower and upper values
            ---------------------------------------------------------------
            value: value selected on the var dropdown button
            id: id of the var dropdown button
            add_click: click on add_dropdown_button
            ---------------------------------------------------------------
            return modalities components. If the component is new, value
            is empty by default.
            """
            # Context and init handling (no action)
            ctx = dash.callback_context
            if not ctx.triggered:
                raise dash.exceptions.PreventUpdate
            # No update last modalities values if we click on add button
            if ctx.triggered[0]['prop_id'] == 'add_dropdown_button.n_clicks':
                raise dash.exceptions.PreventUpdate
            # Creation on modalities dropdown button
            else:
                if value is not None:
                    if type(self.layout.round_dataframe[value].iloc[0]) == bool:
                        new_element = html.Div(dcc.RadioItems(
                            [{'label': val, 'value': val} for
                             val in self.layout.round_dataframe[value].unique()],
                            id={'type': 'dynamic-bool',
                                'index': id['index']},
                            value=self.layout.round_dataframe[value].iloc[0],
                            inline=False
                            ), style={"width": "65%", 'margin-left': '20px'})
                    elif (type(self.layout.round_dataframe[value].iloc[0]) == str) | \
                         ((type(self.layout.round_dataframe[value].iloc[0]) == np.int64) &
                          (len(self.layout.round_dataframe[value].unique()) <= 20)):
                        new_element = html.Div(dcc.Dropdown(
                            id={
                               'type': 'dynamic-str',
                               'index': id['index']
                            },
                            options=[{'label': i, 'value': i} for
                                    i in np.sort(self.layout.round_dataframe[value].unique())],
                            multi=True,
                            ), style={"width": "65%", 'margin-left': '20px'})
                    elif ((type(self.layout.round_dataframe[value].iloc[0]) is pd.Timestamp) |
                          (type(self.layout.round_dataframe[value].iloc[0]) is datetime.datetime)):
                        new_element = html.Div(
                            dcc.DatePickerRange(
                                id={
                                   'type': 'dynamic-date',
                                   'index': id['index']
                                },
                                min_date_allowed=self.layout.round_dataframe[value].min(),
                                max_date_allowed=self.layout.round_dataframe[value].max(),
                                start_date=self.layout.round_dataframe[value].min(),
                                end_date=self.layout.round_dataframe[value].max()
                               ), style={'width': '65%', 'margin-left': '20px'}),
                    else:
                        lower_value = 0
                        upper_value = 0
                        new_element = html.Div([
                                            dcc.Input(
                                                id={
                                                    'type': 'lower',
                                                    'index': id['index']
                                                },
                                                value=lower_value,
                                                type="number",
                                                style={'width': '60px'}),
                                            ' <= {} in [{}, {}]<= '.format(
                                                value,
                                                self.layout.round_dataframe[value].min(),
                                                self.layout.round_dataframe[value].max()),
                                            dcc.Input(
                                                id={
                                                    'type': 'upper',
                                                    'index': id['index']
                                                },
                                                value=upper_value,
                                                type="number",
                                                style={'width': '60px'}
                                            )
                        ], style={'margin-left': '20px'})
                else:
                    new_element = html.Div()
                return new_element
