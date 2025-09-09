import dash
from dash import Dash, html, dcc, dash_table, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import json

# ------  cria o app  ------
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# ------  obrigatório pro Railway  ------
server = app.server 
# ==============================================================================
# 1. CARREGAMENTO E PREPARAÇÃO DOS DADOS
# ==============================================================================
# Nota: Certifique-se de que o arquivo 'base_mario_agrupado_30julho_manha_fim.csv' 
# está na mesma pasta que este script.
filepath = 'base_mario_agrupado_30julho_manha_fim.csv'
try:
    df = pd.read_csv(filepath, sep=',', decimal='.')
except FileNotFoundError:
    print(f"ERRO CRÍTICO: O arquivo '{filepath}' não foi encontrado.")
    print("Por favor, verifique se o nome do arquivo está correto e se ele está na mesma pasta que o script 'app.py'.")
    exit()

df.columns = df.columns.str.strip().str.upper()

EXPECTED_COLUMNS = ['ANO', 'MES', 'NOME_RECEITA', 'ARRECADADO']
if not all(col in df.columns for col in EXPECTED_COLUMNS):
    print("ERRO CRÍTICO: O arquivo CSV não contém todas as colunas esperadas.")
    print(f"Colunas esperadas: {EXPECTED_COLUMNS}")
    print(f"Colunas encontradas após a limpeza: {df.columns.tolist()}")
    exit()

df['ARRECADADO'] = pd.to_numeric(df['ARRECADADO'], errors='coerce')
df.dropna(subset=['ARRECADADO'], inplace=True)
df['DATA'] = pd.to_datetime(df['ANO'].astype(str) + '-' + df['MES'].astype(str) + '-01')

# ==============================================================================
# 2. CÁLCULOS GLOBAIS
# ==============================================================================
total_2022 = df[df['ANO'] == 2022]['ARRECADADO'].sum()
total_2023 = df[df['ANO'] == 2023]['ARRECADADO'].sum()
total_2024 = df[df['ANO'] == 2024]['ARRECADADO'].sum()
# Corrigido para corresponder ao ano 2024, assumindo que os dados de 2025 não estão presentes
# Se houver dados de 2025, ajuste o ano conforme necessário.
total_2025_jun = df[(df['ANO'] == 2025) & (df['MES'] <= 6)]['ARRECADADO'].sum() 

def format_currency(value):
    return f"R$ {value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

# ==============================================================================
# 3. INICIALIZAÇÃO DO APP DASH
# ==============================================================================
# A pasta 'assets' é reconhecida automaticamente pelo Dash
app = dash.Dash(__name__, 
                external_stylesheets=[dbc.themes.CYBORG, dbc.icons.FONT_AWESOME],
                suppress_callback_exceptions=True) # Adicionado para evitar erros em apps multi-página

# ==============================================================================
# 4. DEFINIÇÃO DO LAYOUT (SIDEBAR E CONTEÚDO)
# ==============================================================================
sidebar = html.Div(
    [
        html.Img(src=app.get_asset_url('logo_nvt.jpg'), style={'width': '100%', 'margin-bottom': '25px'}),
        
        html.H2("Análise de Receitas", className="display-6"),
        html.Hr(),
        html.P("Um dashboard para análise da arrecadação.", className="lead"),
        dbc.Nav(
            [
                dbc.NavLink([html.I(className="fas fa-home me-2"), "Visão Geral"], href="/", active="exact"),
                dbc.NavLink([html.I(className="fas fa-mouse-pointer me-2"), "Análise Interativa"], href="/analises", active="exact"),
                dbc.NavLink([html.I(className="fas fa-calculator me-2"), "Estatísticas & Previsão"], href="/estatisticas", active="exact"),
                dbc.NavLink([html.I(className="fas fa-table me-2"), "Tabela Detalhada"], href="/tabela", active="exact"),
            ],
            vertical=True, pills=True,
        ),
    ],
    className="p-4",
    style={"position": "fixed", "top": 0, "left": 0, "bottom": 0, "width": "20rem", "background-color": "#222", "color": "white"}
)

content = html.Div(id="page-content", style={"margin-left": "22rem", "margin-right": "2rem", "padding": "2rem 1rem"})
app.layout = html.Div([dcc.Location(id="url"), sidebar, content])

# ==============================================================================
# 5. DEFINIÇÃO DAS PÁGINAS E SEUS CONTEÚDOS
# ==============================================================================
# --- Página "Visão Geral" ---
visao_geral_layout = dbc.Container([
    dbc.Row([
        dbc.Col(dbc.Card([dbc.CardHeader("Arrecadado em 2022"), dbc.CardBody(html.H4(format_currency(total_2022), className="card-title"))], color="primary", inverse=True), width=3),
        dbc.Col(dbc.Card([dbc.CardHeader("Arrecadado em 2023"), dbc.CardBody(html.H4(format_currency(total_2023), className="card-title"))], color="success", inverse=True), width=3),
        dbc.Col(dbc.Card([dbc.CardHeader("Arrecadado em 2024"), dbc.CardBody(html.H4(format_currency(total_2024), className="card-title"))], color="info", inverse=True), width=3),
        dbc.Col(dbc.Card([dbc.CardHeader("Arrecadado em 2025 (até Jun)"), dbc.CardBody(html.H4(format_currency(total_2025_jun), className="card-title"))], color="warning", inverse=True), width=3),
    ], className="mb-4 text-center"),
    dbc.Row([dbc.Col(html.H3("Visão Geral da Arrecadação"), width=12, className="mb-4 mt-4")]),
    dbc.Row([dbc.Col([html.H5("Selecione o Ano:"), dcc.Dropdown(id='filtro-ano', options=[{'label': ano, 'value': ano} for ano in sorted(df['ANO'].unique())], value=df['ANO'].max(), clearable=False)], width=12)], className="mb-4"),
    dbc.Row([dbc.Col(dcc.Graph(id='grafico-top10-barras-geral'), width=12)], className="mb-4"),
    dbc.Row([dbc.Col(dcc.Graph(id='grafico-evolucao-total-linha'), width=12)])
], fluid=True)

# --- Página "Análise Interativa" ---
analise_interativa_layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H5("1. Selecione o Ano de Referência:"),
            dcc.Dropdown(id='filtro-ano-analises', options=[{'label': ano, 'value': ano} for ano in sorted(df['ANO'].unique())], value=df['ANO'].max(), clearable=False)
        ], width=6),
        dbc.Col([
            html.H5("2. Selecione o Mês para Comparar:"),
            dcc.Dropdown(id='filtro-mes-analises', options=[{'label': mes, 'value': mes} for mes in sorted(df['MES'].unique())], value=1, clearable=False)
        ], width=6),
    ], className="mb-4"),
    html.Hr(),
    dbc.Row([
        dbc.Col([
            html.H5("3. Clique na Receita Desejada:"),
            html.Div(id='legenda-interativa-container', className="mt-3", style={'max-height': '500px', 'overflowY': 'auto'})
        ], width=5),
        dbc.Col([
            html.H5("Comparativo Anual da Receita Escolhida:"),
            html.Div(
                dcc.Graph(id='grafico-comparativo-receita-mes'),
                style={'height': '500px', 'overflow': 'hidden'}
            )
        ], width=7),
    ]),
    
    html.Hr(className="my-4"),
    dbc.Row([
        dbc.Col(dcc.Graph(id='treemap-top10'), width=6),
        dbc.Col(dcc.Graph(id='treemap-top10-vs-others'), width=6),
    ], className="mt-4"),

], fluid=True)

# --- Página "Estatísticas e Previsão" ---
estatisticas_previsao_layout = dbc.Container([
    dbc.Row([dbc.Col(html.H3("Estatísticas Descritivas e Previsão de Arrecadação"), width=12, className="mb-4")]),
    dbc.Row([dbc.Col([html.H5("Estatísticas de Arrecadação por Ano e Mês"), dash_table.DataTable(id='tabela-estatisticas', page_size=15, sort_action="native", style_table={'overflowX': 'auto'}, style_header={'backgroundColor': 'rgb(30, 30, 30)', 'color': 'white'}, style_cell={'backgroundColor': 'rgb(50, 50, 50)', 'color': 'white', 'textAlign': 'left'},)], width=12)], className="mb-5"),
    dbc.Row([dbc.Col([html.H5("Previsão de Arrecadação Total (Modelo SARIMA)"), html.P("Use o slider para selecionar o número de meses a prever."), dcc.Slider(id='slider-previsao', min=3, max=24, step=1, value=12, marks={i: str(i) for i in range(3, 25, 3)}),], width=12)], className="mt-4"),
    dbc.Row([dbc.Col(dcc.Loading(children=[dcc.Graph(id='grafico-previsao')]), width=12)])
], fluid=True)

# --- Página "Tabela Detalhada" ---
tabela_detalhada_layout = dbc.Container([
    dbc.Row([dbc.Col(html.H3("Tabela Detalhada de Receitas"), width=12, className="mb-4")]),
    dbc.Row([dbc.Col([html.H5("Selecione o Ano para visualizar na tabela:"), dcc.Dropdown(id='filtro-ano-tabela', options=[{'label': ano, 'value': ano} for ano in sorted(df['ANO'].unique())], value=df['ANO'].max(), clearable=False)], width=12)], className="mb-4"),
    dbc.Row([dbc.Col([dash_table.DataTable(id='tabela-dados-filtrada', columns=[{"name": i, "id": i} for i in df.columns if i != 'DATA'], page_size=20, sort_action="native", filter_action="native", style_table={'overflowX': 'auto'}, style_header={'backgroundColor': 'rgb(30, 30, 30)', 'color': 'white'}, style_cell={'backgroundColor': 'rgb(50, 50, 50)', 'color': 'white', 'textAlign': 'left', 'whiteSpace': 'normal', 'height': 'auto'},)], width=12)])
], fluid=True)


# ==============================================================================
# 6. CALLBACKS
# ==============================================================================

# Roteador de Páginas
@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/": return visao_geral_layout
    elif pathname == "/analises": return analise_interativa_layout
    elif pathname == "/estatisticas": return estatisticas_previsao_layout
    elif pathname == "/tabela": return tabela_detalhada_layout
    return html.Div([html.H1("404: Not found")], className="p-3 bg-light rounded-3")

# Callbacks "Visão Geral"
@app.callback(Output('grafico-top10-barras-geral', 'figure'), Input('filtro-ano', 'value'))
def update_bar_chart_geral(ano_selecionado):
    if ano_selecionado is None: return {}
    df_filtrado = df[df['ANO'] == ano_selecionado]
    df_top10 = df_filtrado.groupby('NOME_RECEITA')['ARRECADADO'].sum().nlargest(10).sort_values(ascending=True).reset_index()
    fig = px.bar(df_top10, x='ARRECADADO', y='NOME_RECEITA', orientation='h', title=f'Top 10 Receitas em {ano_selecionado}', template='plotly_dark', labels={'ARRECADADO': 'Valor Arrecadado (R$)', 'NOME_RECEITA': 'Nome da Receita'})
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    return fig

# Callbacks da Página "Análise Interativa"
@app.callback(Output('legenda-interativa-container', 'children'), Input('filtro-ano-analises', 'value'))
def generate_custom_legend(ano_selecionado):
    if ano_selecionado is None: return []
    df_top10 = df[df['ANO'] == ano_selecionado].groupby('NOME_RECEITA')['ARRECADADO'].sum().nlargest(10).reset_index()
    legend_items = []
    for i, row in df_top10.iterrows():
        color = px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
        item = html.Div([dbc.Button([html.Span(className="fa fa-circle me-2", style={'color': color}), html.Span(row['NOME_RECEITA'], style={'color': 'white', 'font-size': '14px'})], id={'type': 'legenda-btn', 'index': i}, className="text-start w-100 mb-2", color="dark")])
        legend_items.append(item)
    return legend_items

@app.callback(
    Output('grafico-comparativo-receita-mes', 'figure'),
    [Input({'type': 'legenda-btn', 'index': dash.ALL}, 'n_clicks'), Input('filtro-mes-analises', 'value')],
    [State('filtro-ano-analises', 'value')]
)
def update_chart_from_legend_click(n_clicks, mes_selecionado, ano_selecionado_ref):
    ctx = callback_context
    if not ctx.triggered or all(c is None for c in n_clicks):
        clicked_index = 0
    else:
        clicked_id_str = ctx.triggered[0]['prop_id'].split('.')[0]
        clicked_index = json.loads(clicked_id_str)['index']
    
    df_top10 = df[df['ANO'] == ano_selecionado_ref].groupby('NOME_RECEITA')['ARRECADADO'].sum().nlargest(10).reset_index()
    
    if clicked_index >= len(df_top10): clicked_index = 0
        
    receita_selecionada = df_top10['NOME_RECEITA'].iloc[clicked_index]
    cor_selecionada = px.colors.qualitative.Plotly[clicked_index % len(px.colors.qualitative.Plotly)]
    
    df_chart_data = df[(df['NOME_RECEITA'] == receita_selecionada) & (df['MES'] == mes_selecionado)]
    df_grouped = df_chart_data.groupby('ANO')['ARRECADADO'].sum().reset_index()

    if df_grouped.empty:
        max_valor_no_grafico = 1
    else:
        max_valor_no_grafico = df_grouped['ARRECADADO'].max()
    eixo_y_final_max = max_valor_no_grafico * 1.20
        
    titulo = f'Comparativo para "{receita_selecionada}" no Mês {mes_selecionado}'
    if len(receita_selecionada) > 40:
        titulo = f'Comparativo para "{receita_selecionada[:40]}..." no Mês {mes_selecionado}'
        
    fig = px.bar(df_grouped, x='ANO', y='ARRECADADO', title=titulo, template='plotly_dark', text='ARRECADADO')
    fig.update_traces(marker_color=cor_selecionada, texttemplate='R$ %{text:,.2f}', textposition='outside')
    fig.update_layout(xaxis_type='category', yaxis_range=[0, eixo_y_final_max])
    return fig

# <<< --- CALLBACK DOS TREEMAPS ATUALIZADO --- <<<
@app.callback(
    [Output('treemap-top10', 'figure'),
     Output('treemap-top10-vs-others', 'figure')],
    [Input('filtro-ano-analises', 'value')]
)
def update_treemaps(ano_selecionado):
    if ano_selecionado is None:
        return go.Figure(), go.Figure()

    df_ano = df[df['ANO'] == ano_selecionado]

    # --- Treemap 1: Comparativo das Top 10 Receitas ---
    df_top10 = df_ano.groupby('NOME_RECEITA')['ARRECADADO'].sum().nlargest(10).reset_index()
    
    # Adiciona uma coluna com o texto formatado para o hover
    df_top10['ARRECADADO_FORMATADO'] = df_top10['ARRECADADO'].apply(format_currency)

    fig_treemap_top10 = px.treemap(
        df_top10,
        path=[px.Constant(f"Top 10 Receitas em {ano_selecionado}"), 'NOME_RECEITA'],
        values='ARRECADADO',
        title=f'<b>Comparativo das Top 10 Receitas em {ano_selecionado}</b>',
        template='plotly_dark',
        color='ARRECADADO',
        color_continuous_scale=px.colors.sequential.Viridis,
        custom_data=['ARRECADADO_FORMATADO'] # Passa os dados formatados para o hover
    )
    
    # Melhorias visuais
    fig_treemap_top10.update_layout(
        margin=dict(t=50, l=25, r=25, b=25),
        font=dict(size=14) # Aumenta o tamanho da fonte geral
    )
    fig_treemap_top10.update_traces(
        # Quebra de linha automática nos rótulos e formatação do hover
        texttemplate="<b>%{label}</b><br>%{percentRoot:.2%}",
        hovertemplate='<b>%{label}</b><br>Valor: %{customdata[0]}<extra></extra>',
        textposition='middle center',
        insidetextfont=dict(size=14)
    )


    # --- Treemap 2: Proporção Top 10 vs. Demais ---
    total_arrecadado_ano = df_ano['ARRECADADO'].sum()
    total_top10 = df_top10['ARRECADADO'].sum()
    total_outros = total_arrecadado_ano - total_top10

    df_comparativo = pd.DataFrame({
        'Categoria': ['Top 10 Receitas', 'Demais Receitas'],
        'Valor': [total_top10, total_outros]
    })
    
    # Adiciona uma coluna com o texto formatado para o hover
    df_comparativo['VALOR_FORMATADO'] = df_comparativo['Valor'].apply(format_currency)

    fig_treemap_vs_others = px.treemap(
        df_comparativo,
        path=[px.Constant(f"Total Arrecadado em {ano_selecionado}"), 'Categoria'],
        values='Valor',
        title=f'<b>Proporção: Top 10 vs Demais em {ano_selecionado}</b>',
        template='plotly_dark',
        color='Categoria',
        color_discrete_map={
            'Top 10 Receitas': '#2ca02c', # Verde
            'Demais Receitas': '#7f7f7f'  # Cinza
        },
        custom_data=['VALOR_FORMATADO'] # Passa os dados formatados para o hover
    )
    
    # Melhorias visuais
    fig_treemap_vs_others.update_layout(
        margin=dict(t=50, l=25, r=25, b=25),
        font=dict(size=14)
    )
    fig_treemap_vs_others.update_traces(
        texttemplate="<b>%{label}</b><br>%{percentRoot:.2%}",
        hovertemplate='<b>%{label}</b><br>Valor: %{customdata[0]}<extra></extra>',
        textposition='middle center',
        insidetextfont=dict(size=20) # Fonte maior para as duas categorias principais
    )

    return fig_treemap_top10, fig_treemap_vs_others


# Callbacks das Outras Páginas
@app.callback(Output('grafico-evolucao-total-linha', 'figure'), Input('filtro-ano', 'value'))
def update_total_line_chart(_):
    df_total = df.groupby('DATA')['ARRECADADO'].sum().reset_index()
    fig = px.line(df_total, x='DATA', y='ARRECADADO', title='Evolução da Arrecadação Total (Todos os Anos)', template='plotly_dark', markers=True, labels={'DATA': 'Data', 'ARRECADADO': 'Valor Arrecadado (R$)'})
    return fig

@app.callback(Output('tabela-dados-filtrada', 'data'), Input('filtro-ano-tabela', 'value'))
def update_table_data(ano_selecionado):
    if ano_selecionado is None: return df.to_dict('records')
    df_filtrado = df[df['ANO'] == ano_selecionado]
    return df_filtrado.to_dict('records')

@app.callback([Output('tabela-estatisticas', 'data'), Output('tabela-estatisticas', 'columns')], Input('url', 'pathname'))
def update_stats_table(pathname):
    if pathname == '/estatisticas':
        df_stats = df.groupby(['ANO', 'MES'])['ARRECADADO'].agg(['sum', 'mean', 'median', 'std', 'min', 'max']).reset_index()
        df_stats.columns = ['Ano', 'Mês', 'Total Arrecadado', 'Média', 'Mediana', 'Desvio Padrão', 'Mínimo', 'Máximo']
        for col in df_stats.columns[2:]:
            df_stats[col] = df_stats[col].apply(lambda x: f'R$ {x:,.2f}'.replace(",", "X").replace(".", ",").replace("X", "."))
        columns = [{"name": i, "id": i} for i in df_stats.columns]
        data = df_stats.to_dict('records')
        return data, columns
    return [], []

@app.callback(Output('grafico-previsao', 'figure'), Input('slider-previsao', 'value'))
def update_forecast_chart(n_meses_previsao):
    ts_data = df.groupby('DATA')['ARRECADADO'].sum()
    try:
        model = SARIMAX(ts_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), enforce_stationarity=False, enforce_invertibility=False)
        results = model.fit(disp=False)
        forecast = results.get_forecast(steps=n_meses_previsao)
        forecast_df = forecast.summary_frame(alpha=0.05)
    except Exception as e:
        print(f"Erro no modelo de previsão: {e}")
        return go.Figure(layout={'template': 'plotly_dark', 'title': 'Não foi possível gerar a previsão. Poucos dados.'})
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts_data.index, y=ts_data, mode='lines', name='Histórico'))
    fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['mean'], mode='lines', name='Previsão', line={'dash': 'dot'}))
    fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['mean_ci_upper'], fill=None, mode='lines', line_color='rgba(255,255,255,0.1)', showlegend=False))
    fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['mean_ci_lower'], fill='tonexty', mode='lines', line_color='rgba(255,255,255,0.1)', name='Intervalo de Confiança'))
    fig.update_layout(template='plotly_dark', title=f'Previsão de Arrecadação para os Próximos {n_meses_previsao} Meses')
    return fig

# ==============================================================================
# 7. EXECUÇÃO DO SERVIDOR
# ==============================================================================
if __name__ == '__main__':
    # Para desenvolvimento, use debug=True. Para "produção" ou para ocultar o menu de erros, use debug=False.
    # app.run(debug=False)
    from waitress import serve
    serve(app.server, host="0.0.0.0", port=8050)



