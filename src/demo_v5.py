from __future__ import print_function, division
from dash.dependencies import Input, Output, State, Event
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
#import dash_table_experiments as dt
import pandas as pd
import statsmodels.api as sm
import io, csv, plotly, datetime, flask, base64, dash
import numpy as np
import Stats as stat
from io import StringIO
import plotly.figure_factory as ff
from collections import OrderedDict
import dash_table_experiments as dt
from json_tricks import dumps, loads
import cvxpy as cvx
from scipy.optimize import linprog
from scipy.io import loadmat


def ComputeMoments(X, p):
    """
    Compute Moments for a given data matrix, X, and a given probability vector.
    :param X: Matrix
    :param p: probability vector
    :return: means, standard deviation, and correlation matrix
    """
    p = p.reshape(-1,1)
    J, N = X.shape
    m = X.T.dot(p)
    Sm = X.T.dot((X * np.kron(np.ones((1, N), dtype=float), p)))
    S = Sm - m.dot(m.T)
    C, s = sm.stats.moment_helpers.cov2corr(S, return_std=True)
    return m,s,C
def cvxObjectiveFun(p, A, b, Aeq = None, beq = None, disp = False, solve = 'CVXOPT', maxiter = 10, sumprob = True):
    p = p.ravel()
    if Aeq == None:
        m,n = A.shape
        q = cvx.Variable(A.shape[1])
        rel_ent = cvx.sum(cvx.kl_div(q, p) + q - p)
        objective = cvx.Minimize(rel_ent)
        if sumprob:
            constraints = [
                cvx.sum(q) == 1,
                q >= 0
            ]
        else: constraints = [q >= 0]
        for i in range(A.shape[0]):
            constraints.append(A[i,:]*q <= b[i])
        option = {'kkt_solver' : 'ROBUST_KKTSOLVER', 'max_iters' : maxiter}
        if solve == 'CVXOPT':
            pstar = cvx.Problem(objective, constraints).solve(solver = solve, verbose = disp,
                                                              kkt_solver = 'ROBUST_KKTSOLVER', max_iters = maxiter)
        else: pstar = cvx.Problem(objective, constraints).solve(solver = solve, verbose = disp)
    elif A == None:
        m,n = Aeq.shape
        q = cvx.Variable(Aeq.shape[1])
        objective = cvx.Minimize(
            cvx.sum(cvx.kl_div(q, p) + q - p)
        )
        if sumprob:
            constraints = [cvx.sum(q) == 1, q >= 0]
        else:
            constraints = [q >= 0]
        for i in range(Aeq.shape[0]):
            constraints.append(Aeq[i,:]*q == beq[i])
        option = {'kkt_solver' : 'ROBUST_KKTSOLVER', 'max_iters' : maxiter}
        if solve == 'CVXOPT': pstar = cvx.Problem(objective, constraints).solve(solver = solve, verbose = disp, **option)
        else: pstar = cvx.Problem(objective, constraints).solve(solver = solve, verbose = disp)
    else:
        m,n = A.shape
        q = cvx.Variable(A.shape[1])
        objective = cvx.Minimize(
            cvx.sum(cvx.kl_div(q, p) + q - p  )
        )
        if sumprob: constraints = [cvx.sum(q) == 1,q >= 0]
        else: constraints = [q >= 0]
        for i in range(A.shape[0]):
            constraints.append(A[i,:]*q <= b[i])
        for i in range(Aeq.shape[0]):
            constraints.append(Aeq[i,:]*q == beq[i])
        option = {'kkt_solver' : 'ROBUST_KKTSOLVER', 'max_iters' : maxiter}
        if solve == 'CVXOPT': pstar = cvx.Problem(objective, constraints).solve(solver = solve, verbose = disp, **option)
        else: pstar = cvx.Problem(objective, constraints).solve(solver = solve, verbose = disp)
    return q.value
def getPriorMoments(df, p):
    m,s,C = ComputeMoments(df.values, p)
    m = m.ravel()
    var = list(df.columns)
    table = [['Risk Factor' , 'Prior Mean Returns', 'Prior Variance']]
    for i in range(len(var)):
        table.append([var[i] , m[i], s[i] ** 2])
    fig = ff.create_table(table)
    return fig
def getKDE(X, var, p, prior, n = 10000):
    kde_est = {}
    x_min = np.min(X) - 3
    x_max = np.max(X) + 3
    kde_est['x_grid'] = np.linspace(x_min, x_max, n)
    for i, v in enumerate(var):
        x_min = np.min(X[:, i]) - 3
        x_max = np.max(X[:, i]) + 3
        kde_est[v + '_x_grid'] = np.linspace(x_min, x_max, n)
        kde_est[v + '_posterior'] = stat.kde(X[:, i], kde_est[v + '_x_grid'], weights=p.ravel()).ravel()
        kde_est[v + '_prior'] = stat.kde(X[:, i], kde_est[v + '_x_grid'], weights=prior.ravel()).ravel()
    kde_est = pd.DataFrame(kde_est, dtype=float)
    return kde_est
def getPosteriorMoments(df, pos):
    m,s,C = ComputeMoments(df.values, pos)
    m = m.ravel()
    var = list(df.columns)
    table = [['Risk Factor' , 'Posterior Mean Returns', 'Posterior Variance']]
    for i in range(len(var)):
        table.append([var[i] , m[i], s[i]**2])
    fig = ff.create_table(table)
    return fig
def checkConstraints(b, pos, ind, isvarcons):
    n = b.shape[0]
    m, s, C = ComputeMoments(df.values, pos.reshape(-1, 1))
    table = [['Views', 'Inequality Constraints']]
    sat_cons = True
    for i in range(n):
        if isvarcons[i]:
            sat_cons = (s[ind[i]] ** 2 >= (abs(b[i, 0]) - m[ind[i], 0] ** 2))
        else:
            if b[i, 0] >= 0:
                sat_cons = (m[ind[i], 0] >= b[i, 0])
            else:
                sat_cons = (m[ind[i], 0] <= b[i, 0])
    return sat_cons
def getIneqFig(b, m, ind, isvarcons):
    n = b.shape[0]
    table = [['Views', 'Inequality Constraints']]
    for i in range(n):
        if isvarcons[i]:
            table.append(['Variance View on ' + vars[ind[i]], abs(b[i,0]) - m[ind[i],0] ** 2])
        else:
            table.append(['Mean View on ' + vars[ind[i]] , b[i,0]])
    ineq_fig = ff.create_table(table)
    return ineq_fig
def getEqFig(beq, m, ind, isvarcons):
    table = [['Views', 'Equality Constraint']]
    for i in range(len(beq)):
        table.append(['Mean View on ' + vars[ind[i]], beq[i,0]])
    eq_fig = ff.create_table(table)
    return eq_fig
def getViewsIntoDataFrame():
    views = OrderedDict()
    views['Index'] = [];
    views['Risk Factor'] = [];
    views['Mean View'] = [];
    views['Volatility Views'] = []
    for i, v in enumerate(vars):
        views['Index'].append(i)
        views['Risk Factor'].append(v)
        views['Mean View'].append(0)
        views['Volatility Views'].append(1)
    return pd.DataFrame(views, dtype = float)
def KernelBootstrap(data, epsilon, J):
    obs, m = data.shape
    C = data.cov()
    numsamples = int(np.floor(J/obs))
    samples = np.zeros((numsamples*obs,m) , dtype = float)
    for i in range(obs-1):
        samples[i*numsamples : (i+1)*numsamples, : ] = np.random.multivariate_normal(data.iloc[i,:], epsilon*C, numsamples)
    return samples
def getBootstrapData(data, J = (10**4)):
    eps = 0.15;
    data_boot = KernelBootstrap(data, eps, J)
    data_boot = pd.DataFrame(data_boot, columns=list(data.columns), dtype = float)
    N = data_boot.shape[0]
    return data_boot, N
def getParams():
    kernel_boot = {'epsilon' : [0.0] , 'Number of Observations' : [0.0]}
    return pd.DataFrame(kernel_boot, dtype= float)
def CondProbViewsThreshold(View, X):
    """
    A function to convert a dictionary of views to a given matrix to be used for EntropyProg
    :param View: A dictionary which contains the keys of 'Who', 'Equal', 'Cond_Who', 'Cond_Equal', 'sgn', 'v', 'below', 'Cond_Below'
    :param X: X is the vector of data
    :return: Returns the matrix A, b, and gamma
    """
    n, m = X.shape[0], len(View)
    A = np.zeros((m, n), dtype=float)
    b = []
    g = []
    for k in range(len(View)):
        I_mrg = np.isfinite(X[:, 0])

        for s in range(len(View[k]['Who'])):
            Who = View[k]['Who'][s]
            Or_Targets = View[k]['Equal'][s]
            I_mrg_or = X[:,Who] > np.inf
            cond = View[k]['below'][s]
            if cond[0]:
                for i in range(len(Or_Targets)):
                    I_mrg_or = np.logical_or(I_mrg_or, X[:, Who] <= np.array(Or_Targets[i]))
            else:
                for i in range(len(Or_Targets)):
                    I_mrg_or = np.logical_or(I_mrg_or, X[:, Who] >= np.array(Or_Targets[i]))
            I_mrg = np.logical_and(I_mrg, I_mrg_or)
        I_cnd = np.isfinite(X[:, 0])

        for s in range(len(View[k]['Cond_Who'])):
            Who = View[k]['Cond_Who'][s]
            Or_Targets = View[k]['Cond_Equal'][s]
            cond = View[k]['Cond_below'][s]
            I_cnd_or = X[:,Who] > np.inf
            if cond[0]:
                for i in range(len(Or_Targets)):
                    I_cnd_or = np.logical_or(I_cnd_or, X[:, Who] <= np.array(Or_Targets[i]))
            else:
                for i in range(len(Or_Targets)):
                    I_cnd_or = np.logical_or(I_cnd_or, X[:, Who] >= np.array(Or_Targets[i]))
            I_cnd = np.logical_and(I_cnd, I_cnd_or)
        I_jnt = np.logical_and(I_mrg, I_cnd)

        if len(View[k]['Cond_Who']) != 0:
            sgn = np.array(View[k]['sgn'], dtype=float).reshape(-1, 1)
            I_jnt = np.array(I_jnt)
            temp1 = (I_jnt - np.array(View[k]['v'], dtype=float) * I_cnd).reshape(-1, 1).T
            new_A = sgn.dot(temp1).squeeze()
            new_B = 0
        else:
            new_A = np.dot(np.array(View[k]['sgn']).reshape(-1, 1), I_mrg.reshape(-1, 1).T)
            new_B = np.array(View[k]['sgn'], dtype=float).dot(View[k]['v']).squeeze()
        A[k, :] = new_A
        b.append(new_B)
        g.append(-np.log(1 - np.array(View[k]['c'], dtype=float)))
    return A, np.array(b, dtype = float), np.array(g, dtype = float)
def tweak(A, b, g, options='scipy'):
    """
    Consistency check as described in A. Meucci - "Fully Flexible Views: Theory and Practice" . Ensures that EntropyProg
    returns a consistent probability vector
    :param A: The matrix of constraints
    :param b: The inequality vector
    :param g: array
    :param options: Choice of 'Scipy' or 'CVX' for optimization with using scipy or cvx. The default is 'scipy'.
    :return: db
    """
    if options == 'cvx':
        K, J = A.shape
        #        p = p.reshape(-1,1)
        x = cvx.Variable(K)
        zero = np.zeros((J, 1), dtype=float)
        p = cvx.Variable(J)
        objective = cvx.Minimize(g.T * x)
        constraints = [
            A * p <= (b + x),
            cvx.sum_entries(p) == 1,
            p >= 0,
            p <= 1,
            x >= 0
        ]
        pstar = cvx.Problem(objective, constraints).solve(verbose = True)
        x_sol = x.value
        p_sol = p.value
    else:
        K, J = A.shape
        g_ = np.concatenate((g, np.zeros((J, 1), dtype=float)), axis=0)
        A_ = np.concatenate((-np.identity(K), A), axis=1)
        Aeq = np.concatenate((np.zeros((1, K), dtype=float), np.ones((1, J), dtype=float)), axis=1)
        lb = np.zeros((K + J, 1), dtype=float)
        ub = np.ones((K + J, 1), dtype=float)
        ub[:K, 0] = np.inf
        beq = 1
        bounds = np.concatenate((lb, ub), axis=1)
        res = linprog(g_, A_, b, Aeq, beq, bounds)
        x_sol = res.x[:K]
    return x_sol
def getIndices(View):
    l, indices = len(View.keys()), []
    for i in range(l):
        who = View[i]['Who']
        for w in who:
            indices.append(w)
    return indices
def getThreshold(bin):
    if bin =='quartile':
        quartile = np.percentile(df.values, [(1/4)*100, (1/2)*100, (3/4)*100], axis = 0)
        return quartile
    elif bin == 'decile':
        decile = np.percentile(df.values, [(i/10)*100 for i in range(1,10)], axis = 0)
        return decile
    else:
        tercile = np.percentile(df.values, [(1 / 3) * 100, (2 / 3) * 100], axis=0)
        return tercile
def updateViews(view):
    for ii in range(len(view.keys())):
        try:
            who, equal, cond_who, cond_equal, threshold = view[ii]['Who'], view[ii]['Equal'], view[ii]['Cond_Who'], view[ii]['Cond_Equal'], view[ii]['Threshold']
        except KeyError:
            print('Dictionary Invalid')
        bin = getThreshold(threshold)
        Equal = [bin[equal[i],who[i]].tolist() for i in range(len(who))]
        Cond_Equal = [bin[cond_equal[j],cond_who[j]].tolist() for j in range(len(cond_who))]
        view[ii]['Equal'] = Equal
        view[ii]['Cond_Equal'] = Cond_Equal
    return view, bin
def getProbConstraints(views):
    J, N = df.values.shape
#    tercile = np.percentile(df.values, [(1 / 3) * 100, (2 / 3) * 100], axis=0)
    View = {}
    exec (views)
    View, tercile = updateViews(View)
    indices = getIndices(View)
    isvarcons = [False] * len(View.keys())
    A, b, g = CondProbViewsThreshold(View, df.values)
    b = np.array(b).reshape(-1, 1)
    g = np.array(g).reshape(-1, 1)
    db = tweak(A, b, g, options='cvx')
    b += db
    tmp = {'b': b, 'varcons': isvarcons, 'index': indices}
    return tmp, A, b
def getProbView(views):
    J, N = df.values.shape
#    tercile = np.percentile(df.values, [(1 / 3)*100 , (2 / 3)*100 ], axis=0)
    View = {}
    exec (views)
    View, tercile = updateViews(View)
    indices = getIndices(View)
    isvarcons = [False] * len(View.keys())
    A, b, g = CondProbViewsThreshold(View, df.values)
    b = np.array(b).reshape(-1, 1)
    g = np.array(g).reshape(-1, 1)
    db = tweak(A, b, g, options='cvx')
    b += db
    p_ = cvxObjectiveFun(p, A, b, disp=True, maxiter=1000, solve='SCS', sumprob=True)
    p_ = np.asarray(p_, dtype=float)
    tmp = {'posterior': p_, 'b': b, 'varcons': isvarcons, 'index': indices, 'A' : A}
    return tmp, A, b
def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            uploaded_data = True
            # Assume that the user uploaded a CSV file
            cont = decoded.decode('utf-8')
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), index_col= 0, header= 0, dtype = float)
            J, N = df.shape
            try:
                p = df['p'].values
                df = df.drop(['p'], axis = 1)
            except KeyError:
                print('Own prior not uploaded, using default prior of uniform probabilities')
                p = np.ones((J, 1), dtype=float) / J
            vars = list(df.columns)
            X = df.values
            figure = getPriorMoments(df, p)
            df_views = getViewsIntoDataFrame()
            m, s, C = ComputeMoments(X, p)
            global df, m, s, C, p, X, vars, figure, df_views, uploaded_data
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded),index_col= 0, header= 0, dtype = float)
            vars = list(df.columns)
            J, N = df.shape
            X = df.values
            p = np.ones((J, 1), dtype=float) / J
            figure = getPriorMoments(df, p)
            m, s, C = ComputeMoments(X, p)
            df_views = getViewsIntoDataFrame()
            global df, m, s, C, p, vars, figure, df_views
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])
    return html.Div([
        html.H5(filename)
    ])

#df = pd.read_csv('../../Data/meucci_return_data.csv', index_col= 0, header= 0, dtype = float)
df = pd.DataFrame({'test':np.ones(10)}, dtype = float)
J, N = df.shape
p = np.ones((J, 1), dtype=float) / J
vars = list(df.columns)
vertical = True
states = [-1, 0, 1, 'none']
views = ['Moment Views', 'Probabilistic Views']
df_views = getViewsIntoDataFrame()
last_optimize, last_download, download_data = 0, 0, 0
#begin dash app
app = dash.Dash()
app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})

def serve_layout():
    return html.Div([
        html.H4(children='Moments'),
        html.Div(
            id = 'moments',
            className = 'row'),
        html.Div([
            html.Hr(style={'margin': '0', 'margin-bottom': '5'}),
            html.H4(children = 'Input Views')
            ], className = 'row'),
        html.Div([
            html.Div([
                html.Label('Use Kernel Bootstrap ?'),
                dcc.RadioItems(
                    id = 'use-bootstrap',
                    options = [{'label': 'True' , 'value' : True}, {'label' : 'False' , 'value' : False}],
                    value = False, labelStyle = {'display' : 'inline-block'}
                ),
                html.Div(id='bootstrap-data' , style = {'display' : 'none'}),
                html.Label('Select Views'),
                dcc.Dropdown(
                            id = 'select_views',
                            options = [{'label' : view, 'value' : view} for view in views],
                            value = views[0],
                            multi = True
                        ),
                html.Div(id='enter-views', className='rows'),
            ])
            ], className = 'two rows'),
        html.Div([
                html.H4('Editable DataTable'),
                dt.DataTable(
                    rows = df_views.to_dict('records'),
                    editable = True,
                    id = 'editable-table'
                ),
                dcc.Textarea(id='probabilistic-views',
                          placeholder='Enter Probabilistic Views',
                          value="View[0] = {'Who': [0], 'Equal': [[0, 0]], 'below': [[True]], 'Cond_Who': [],'Cond_Equal': [[]], 'v': [0.99], 'sgn': [-1],'c': [0.99], 'Cond_below': [[]], 'Threshold' : 'tercile' }",
                          style={'width': '100%', 'max-height': '350', 'max-width': '100%'}),
                html.Button('Optimize', id='optimize', style={'display': 'inline-block'})
                ], className = 'six columns', style = {'display' : 'none'}),
        html.Div(id='intermediate-value' , style = {'display' : 'none'}),
        html.Div(id='intermediate-kde-value', style = {'display' : 'none'}),
        html.Div([
            html.Hr(style={'margin': '0', 'margin-bottom': '5'}),
            html.Div([
                html.Div(id='check-constraints'),
                html.A(
                    html.Button('Download Data', id='my-dropdown', style={'display': 'inline-block'}),
                    id='download-link',
                    download="rawdata.csv",
                    href="",
                    target="_blank")
                ], className='rows'),
            html.Hr(style={'margin': '0', 'margin-bottom': '5'}),
            html.H4('Posterior vs Prior Distribution')
            ] , className = 'row'),
        html.Div(
            dcc.Tabs(
                tabs=[
                    {'label': var, 'value': var} for var in vars
                ],
                value=vars[0],
                id='tabs',
                vertical=vertical,
                style={
                    'height': '100vh',
                    'borderRight': 'thin lightgrey solid',
                    'textAlign': 'left'
                }
            ),
            style={'width': '20%', 'float': 'left'}
        ),
        html.Div(
            html.Div(id='tab-output'),
            style={'width': '80%', 'float': 'right'}
        )
    ], style={
            'fontFamily': 'Sans-Serif',
            'margin-left': 'auto',
            'margin-right': 'auto',
        }
    )

app.scripts.config.serve_locally = True
app.layout = html.Div([
    html.Div([
        html.H1('Stress Testing with Causal Views', style = {'text-align' : 'center'}),
        html.Hr(style={'margin': '0', 'margin-bottom': '5'}),
        html.Div([
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Files')
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                # Allow multiple files to be uploaded
                multiple=True
            ),
            html.Div(id='output-data-upload'),
            html.Div(dt.DataTable(rows=[{}]), style={'display': 'none'})
        ]),
        html.Div(id = 'upload_data'),
        html.Hr(style={'margin': '0', 'margin-bottom': '5'}),
    ]),
    html.Div(id = 'output'),
    html.Div([
        html.H4(children='Moments'),
        html.Div(
            id = 'moments',
            className = 'row'),
        html.Div([
            html.Hr(style={'margin': '0', 'margin-bottom': '5'}),
            html.H4(children = 'Input Views')
            ], className = 'row'),
        html.Div([
            html.Div([
                html.Label('Use Kernel Bootstrap ?'),
                dcc.RadioItems(
                    id = 'use-bootstrap',
                    options = [{'label': 'True' , 'value' : True}, {'label' : 'False' , 'value' : False}],
                    value = False, labelStyle = {'display' : 'inline-block'}
                ),
                html.Div(id='bootstrap-data' , style = {'display' : 'none'}),
                html.Label('Select Views'),
                dcc.Dropdown(
                            id = 'select_views',
                            options = [{'label' : view, 'value' : view} for view in views],
                            value = views[0],
                            multi = True
                        ),
                html.Div(id='enter-views', className='rows'),
            ])
            ], className = 'two rows'),
        html.Div([
                html.H4('Editable DataTable'),
                dt.DataTable(
                    rows = df_views.to_dict('records'),
                    editable = True,
                    id = 'editable-table'
                ),
                dcc.Textarea(id='probabilistic-views',
                          placeholder='Enter Probabilistic Views',
                          value="View[0] = {'Who': [0], 'Equal': [[0, 0]], 'below': [[True]], 'Cond_Who': [],'Cond_Equal': [[]], 'v': [0.99], 'sgn': [-1],'c': [0.99], 'Cond_below': [[]], 'Threshold' : 'tercile' }",
                          style={'width': '100%', 'max-height': '350', 'max-width': '100%'}),
                html.Button('Optimize', id='optimize', style={'display': 'inline-block'})
                ], className = 'six columns', style = {'display' : 'none'}),
        html.Div(id='intermediate-value' , style = {'display' : 'none'}),
        html.Div(id='intermediate-kde-value', style = {'display' : 'none'}),
        html.Div([
            html.Hr(style={'margin': '0', 'margin-bottom': '5'}),
            html.Div([
                html.Div(id='check-constraints'),
                html.A(
                    html.Button('Download Data', id='my-dropdown', style={'display': 'inline-block'}),
                    id='download-link',
                    download="rawdata.csv",
                    href="",
                    target="_blank")
            ], className='rows'),
            html.Hr(style={'margin': '0', 'margin-bottom': '5'}),
            html.H4('Posterior vs Prior Distribution')
            ] , className = 'row'),
            html.Div(
                dcc.Tabs(
                    tabs=[
                        {'label': var, 'value': var} for var in vars
                    ],
                    value=vars[0],
                    id='tabs',
                    vertical=vertical,
                    style={
                        'height': '100vh',
                        'borderRight': 'thin lightgrey solid',
                        'textAlign': 'left'
                    }
                ),
                style={'width': '20%', 'float': 'left'}
            ),
            html.Div(
                html.Div(id='tab-output'),
                style={'width': '80%', 'float': 'right'}
            )
    ], style={
            'fontFamily': 'Sans-Serif',
            'margin-left': 'auto',
            'margin-right': 'auto',
            'display': 'none'
        }
    )])

@app.callback(Output('download-link', 'href'),
              [Input('my-dropdown', 'n_clicks'),
               Input('intermediate-value', 'children')])
def update_link(download, json_data):
    if json_data is not None:
        if download > last_download:
            last_download = download
            global last_download
            tmp = loads(json_data)
            if 'b' in tmp.keys():
                link = '/dash/urlToDownload?value={}'.format(json_data)
            else:
                link = '/dash/urlToDownload?value={}'.format(json_data)
            return link
        else:
            link = '/dash/urlToDownload?value={}'.format(json_data)
            return link

@app.server.route('/dash/urlToDownload')
def download_csv():
    value = flask.request.args.get('value')
    pos = loads(value)['posterior']
    temp_val = np.concatenate((pos, df.values), axis = 1)
    col = ['posterior_prob'] + list(df.columns)
    strIO = StringIO.StringIO()
    writer = csv.writer(strIO)
    writer.writerow(col)
    for i in range(temp_val.shape[0]):
        writer.writerow(temp_val[i, :].tolist())
    strIO.seek(0)
    return flask.send_file(strIO,
                     mimetype='text/csv',
                     attachment_filename='posterior.csv',
                     as_attachment=True)

@app.callback(Output('tabs', 'tabs'),
              [Input('upload-data' , 'contents')])
def upload_tab(list_of_contents):
    if list_of_contents is not None:
        return [
                {'label': var, 'value': var} for var in vars
            ]
    else:
        return [
                {'label': var, 'value': var} for var in vars
            ]

@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents'),
               Input('upload-data', 'filename'),
               Input('upload-data', 'last_modified')])
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children

@app.callback(Output('output','children'),
              [Input('upload-data', 'contents')])
def update_app(list_of_contents):
    if list_of_contents is not None:
        return serve_layout()
    else:
        return html.Div([])

@app.callback(Output('enter-views', 'children'),
              [Input('select_views', 'value')])
def enterViews(views):
    if ('Probabilistic Views' in views) and ('Moment Views' not in views):
        return html.Div([
                            html.H4('Input Probabilistic Views'),
                            dcc.Textarea(id='probabilistic-views',
                                      placeholder='Enter Probabilistic Views',
                                         value="View[0] = {'Who': [0], 'Equal': [[0, 0]], 'below': [[True]], 'Cond_Who': [],'Cond_Equal': [[]], 'v': [0.99], 'sgn': [-1],'c': [0.99], 'Cond_below': [[]], 'Threshold' : 'tercile' }",
                                         style={'width': '100%', 'max-height': '350', 'max-width': '100%'}),
                            html.Button('Optimize', id='optimize', style={'display': 'inline-block'})
                        ], className = 'rows')
    elif ('Probabilistic Views' in views) and ('Moment Views' in views):
        return html.Div([html.Div([
                            html.H4('Editable DataTable'),
                            dt.DataTable(
                                rows = df_views.to_dict('records'),
                                editable = True,
                                id = 'editable-table'
                            )
                        ], className = 'six columns'),
                html.Div([
                            html.H4('Input Probabilistic Views'),
                            dcc.Textarea(id='probabilistic-views',
                                      placeholder='Enter Probabilistic Views',
                                         value="View[0] = {'Who': [0], 'Equal': [[0, 0]], 'below': [[True]], 'Cond_Who': [],'Cond_Equal': [[]], 'v': [0.99], 'sgn': [-1],'c': [0.99], 'Cond_below': [[]], 'Threshold' : 'tercile' }",
                                         style={'width': '100%', 'max-height': '350', 'max-width': '100%'})], className = 'six columns'),
                html.Button('Optimize', id='optimize', style={'display': 'inline-block'})
                ], className = 'row')
    else:
        return html.Div([
                            html.H4('Editable DataTable For Moment Views'),
                            dt.DataTable(
                                rows = df_views.to_dict('records'),
                                editable = True,
                                id = 'editable-table'
                            ),
                            html.Button('Optimize', id='optimize', style={'display': 'inline-block'})
                        ], className = 'rows')

@app.callback(Output('bootstrap-data', 'children'),
              [Input('use-bootstrap', 'value')])
def update_data(bootstrap):
    if bootstrap:
        data, N = getBootstrapData(df)
        df = data.copy()
        X = df.values
        J, N = df.shape
        p = np.ones((J, 1), dtype=float) / J
        global df, p
    return None

@app.callback(Output('moments','children'),
              [Input('intermediate-value', 'children')])
def update_fig(json_data):
    if json_data is not None:
        tmp = loads(json_data)
        if 'b' in tmp.keys():
            pos = tmp['posterior']
            figure_prior = getPriorMoments(df, p)
            posterior_table = getPosteriorMoments(df, pos.reshape(-1, 1))
            return [html.Div([
                dcc.Graph(id='Prior Moments', figure=figure_prior),
            ], className='six columns'),
                html.Div([
                    dcc.Graph(id='Posterior Moments', figure=posterior_table)
                ], className='six columns')]
        else:
            figure = getPriorMoments(df, p)
            return html.Div([
                dcc.Graph(
                    id='Moments Under Prior',
                    figure=figure
                )
            ])

@app.callback(
    Output('intermediate-value', 'children'),
    [Input('editable-table', 'rows'),
     Input('select_views', 'value'),
     Input('probabilistic-views' , 'value'),
     Input('optimize', 'n_clicks')]
)
def update_output(rows, prob_views, views, optimize):
    if optimize > last_optimize:
        last_optimize = optimize
        global last_optimize
        if ('Probabilistic Views' in prob_views) and ('Moment Views' not in prob_views):
            tmp, A, b = getProbView(views)
            return dumps(tmp)
        elif ('Probabilistic Views' in prob_views) and ('Moment Views' in prob_views):
            df_tmp = pd.DataFrame(rows)
            mean_idx = df_tmp['Mean View'] != 0
            vol_idx = df_tmp['Volatility Views'] != 1
            temp, A, b_prob = getProbConstraints(views)
            A_tmp = OrderedDict()
            isvarcons_prob, index = temp['varcons'], temp['index']
            for j in range(A.shape[0]):
                A_tmp[str(j) + '_prob'] = A[j, :].tolist()
            if np.any(mean_idx) and np.any(vol_idx):  # mean and vol ineq constraint
                mean_value = df_tmp.loc[mean_idx | vol_idx, 'Mean View'].values.astype(float)
                vol_value = df_tmp.loc[vol_idx | mean_idx, 'Volatility Views'].values.astype(float)
                var = list(df_tmp.loc[mean_idx | vol_idx, 'Risk Factor'])
                temp1, temp2, indices1, indices2, isvarcons1, isvarcons2 = [], [], [], [], [], []
                for i, v in enumerate(var):
                    ind = vars.index(v)
                    if mean_value[i] > 0:
                        indices1.append(ind)
                        isvarcons1.append(False)
                        A_tmp[str(i) + '_mean'] = -df[v].values
                        temp1.append(-(m[ind] + mean_value[i] * s[ind]))
                    elif mean_value[i] < 0:
                        indices1.append(ind)
                        isvarcons1.append(False)
                        A_tmp[str(i) + '_mean'] = df[v].values
                        temp1.append((m[ind] + mean_value[i] * s[ind]))
                    if vol_value[i] > 1:
                        A_tmp[str(i) + '_vol'] = -np.power(df[v].values, 2)
                        indices2.append(ind)
                        temp2.append(
                            -(m[ind] ** 2 + vol_value[i] * s[
                                ind] ** 2))  # may need to correct this, currently assuming sample mean
                        isvarcons2.append(True)
                    elif vol_value[i] < 1:
                        A_tmp[str(i) + '_vol'] = np.power(df[v].values, 2)
                        indices2.append(ind)
                        temp2.append(m[ind] ** 2 + vol_value[i] * s[ind] ** 2)
                        isvarcons2.append(True)
                isvarcons = isvarcons_prob + isvarcons1 + isvarcons2
                b = temp1 + temp2
                indices = index + indices1 + indices2
                b = np.array(b, dtype=float).reshape(-1, 1)
                b = np.append(b_prob,b, axis=0)
                A = pd.DataFrame(A_tmp, dtype=float).values.T
                p_ = cvxObjectiveFun(p, A, b, solve='SCS', disp=True, maxiter=100)
                p_ = np.asarray(p_, dtype=float)
                df_prob = pd.DataFrame({'posterior': p_.ravel()}, dtype=float)
                tmp = {'posterior': p_, 'b': b, 'varcons': isvarcons, 'index': indices, 'A' : A}
                return dumps(tmp)
            elif np.any(mean_idx):
                mean_value = df_tmp.loc[mean_idx | vol_idx, 'Mean View'].values.astype(float)
                var = list(df_tmp.loc[mean_idx | vol_idx, 'Risk Factor'])
                b, isvarcons, indices = [], [], []
                for i, v in enumerate(var):
                    ind = vars.index(v)
                    if mean_value[i] > 0:
                        indices.append(ind)
                        isvarcons.append(False)
                        A_tmp[str(i) + '_mean'] = -df[v].values
                        b.append(-(m[ind] + mean_value[i] * s[ind]))
                    elif mean_value[i] < 0:
                        indices.append(ind)
                        isvarcons.append(False)
                        A_tmp[str(i) + '_mean'] = df[v].values
                        b.append((m[ind] + mean_value[i] * s[ind]))
                b = np.array(b, dtype=float).reshape(-1, 1)
                b = np.append(b_prob,b, axis=0)
                A = pd.DataFrame(A_tmp, dtype=float).values.T
                p_ = cvxObjectiveFun(p, A, b, solve='SCS', disp=True, maxiter=100)
                p_ = np.array(p_, dtype=float)
                df_prob = pd.DataFrame({'posterior': p_.ravel()}, dtype=float)
                tmp = {'posterior': p_, 'b': b, 'varcons': isvarcons_prob+isvarcons, 'index': index + indices, 'A' : A}
                return dumps(tmp)
            elif np.any(vol_idx):
                vol_value = df_tmp.loc[vol_idx | mean_idx, 'Volatility Views'].values.astype(float)
                var = list(df_tmp.loc[mean_idx | vol_idx, 'Risk Factor'])
                b, isvarcons, indices = [], [], []
                for i, v in enumerate(var):
                    ind = vars.index(v)
                    if vol_value[i] > 1:
                        indices.append(ind)
                        A_tmp[str(i) + '_vol'] = -np.power(df[v].values, 2)
                        b.append(-(m[ind] ** 2 + vol_value * s[ind] ** 2))
                        isvarcons.append(True)
                    elif vol_value[i] < 1:
                        indices.append(ind)
                        A_tmp[str(i) + '_vol'] = np.power(df[v].values, 2)
                        b.append((m[ind] ** 2 + vol_value * s[ind] ** 2))
                        isvarcons.append(True)
                b = np.array(b, dtype=float).reshape(-1,1)
                b = np.append(b_prob, b, axis = 0)
                A = pd.DataFrame(A_tmp, dtype=float).values.T
                p_ = cvxObjectiveFun(p, A, b, solve='SCS', disp=True, maxiter=100)
                p_ = np.array(p_, dtype=float)
                tmp = {'posterior': p_, 'b': b, 'varcons': isvarcons_prob+isvarcons, 'index': index+indices , 'A' : A}
                return dumps(tmp)
            else:
                b = temp['b']
                A_prob = pd.DataFrame(A_tmp, dtype=float).values.T
                p_ = cvxObjectiveFun(p, A_prob, b, disp=True, maxiter=1000, solve='SCS', sumprob=True)
                p_ = np.asarray(p_, dtype=float)
                tmp = {'posterior': p_, 'b': b, 'varcons': isvarcons_prob, 'index': index, 'A' : A_prob}
                return dumps(tmp)
        else:
            df_tmp = pd.DataFrame(rows)
            mean_idx = df_tmp['Mean View'] != 0
            vol_idx = df_tmp['Volatility Views'] != 1
            A_tmp = {}
            if np.any(mean_idx) and np.any(vol_idx): #mean and vol ineq constraint
                mean_value = df_tmp.loc[mean_idx | vol_idx, 'Mean View'].values.astype(float)
                vol_value = df_tmp.loc[vol_idx | mean_idx, 'Volatility Views'].values.astype(float)
                var = list(df_tmp.loc[mean_idx | vol_idx, 'Risk Factor'])
                temp1, temp2, indices1, indices2, isvarcons1, isvarcons2 = [], [], [], [], [], []
                for i, v in enumerate(var):
                    ind = vars.index(v)
                    if mean_value[i] > 0:
                        indices1.append(ind)
                        isvarcons1.append(False)
                        A_tmp[str(i) + '_mean'] = -df[v].values
                        temp1.append( -(m[ind] + mean_value[i] * s[ind]) )
                    elif mean_value[i] < 0:
                        indices1.append(ind)
                        isvarcons1.append(False)
                        A_tmp[str(i) + '_mean'] = df[v].values
                        temp1.append( (m[ind] + mean_value[i] * s[ind]) )
                    if vol_value[i] > 1:
                        A_tmp[str(i) + '_vol'] = -np.power(df[v].values, 2)
                        indices2.append(ind)
                        temp2.append(
                            -(m[ind] ** 2 + vol_value[i] * s[ind] ** 2))  # may need to correct this, currently assuming sample mean
                        isvarcons2.append(True)
                    elif vol_value[i] < 1:
                        A_tmp[str(i) + '_vol'] = np.power(df[v].values,2)
                        indices2.append(ind)
                        temp2.append(m[ind]**2 + vol_value[i] * s[ind] ** 2)
                        isvarcons2.append(True)
                isvarcons = isvarcons1 + isvarcons2
                b = temp1 + temp2
                indices = indices1 + indices2
                b = np.array(b, dtype=float)
                A = pd.DataFrame(A_tmp, dtype = float).values.T
                p_ = cvxObjectiveFun(p, A, b, solve='SCS', disp=True, maxiter=100)
                p_ = np.asarray(p_, dtype=float)
                df_prob = pd.DataFrame({'posterior': p_.ravel()}, dtype=float)
                tmp = {'posterior': p_, 'b': b, 'varcons': isvarcons, 'index': indices, 'A' : A}
                return dumps(tmp)
            elif np.any(mean_idx):
                mean_value = df_tmp.loc[mean_idx | vol_idx, 'Mean View'].values.astype(float)
                var = list(df_tmp.loc[mean_idx | vol_idx, 'Risk Factor'])
                b, isvarcons, indices = [] , [] , []
                for i,v in enumerate(var):
                    ind = vars.index(v)
                    if mean_value[i] > 0:
                        indices.append(ind)
                        isvarcons.append(False)
                        A_tmp[str(i) + '_mean'] = -df[v].values
                        b.append(-(m[ind] + mean_value[i] * s[ind]))
                    elif mean_value[i] < 0:
                        indices.append(ind)
                        isvarcons.append(False)
                        A_tmp[str(i) + '_mean'] = df[v].values
                        b.append((m[ind] + mean_value[i] * s[ind]))
                b = np.array(b, dtype=float)
                A = pd.DataFrame(A_tmp, dtype=float).values.T
                p_ = cvxObjectiveFun(p, A, b, solve='SCS', disp=True, maxiter=100)
                p_ = np.array(p_, dtype=float)
                df_prob = pd.DataFrame({'posterior': p_.ravel()}, dtype=float)
                tmp = {'posterior': p_, 'b': b, 'varcons': isvarcons, 'index': indices, 'A' : A}
                return dumps(tmp)
            elif np.any(vol_idx):
                vol_value = df_tmp.loc[vol_idx | mean_idx, 'Volatility Views'].values.astype(float)
                var = list(df_tmp.loc[mean_idx | vol_idx, 'Risk Factor'])
                b, isvarcons, indices = [], [], []
                for i, v in enumerate(var):
                    ind = vars.index(v)
                    if vol_value[i] > 1:
                        indices.append(ind)
                        A_tmp[str(i) + '_vol'] = -np.power(df[v].values, 2)
                        b.append(-(m[ind] ** 2 + vol_value * s[ind] ** 2))
                        isvarcons.append(True)
                    elif vol_value[i] < 1:
                        indices.append(ind)
                        A_tmp[str(i) + '_vol'] = np.power(df[v].values,2)
                        b.append((m[ind] ** 2 + vol_value * s[ind] ** 2))
                        isvarcons.append(True)
                b = np.array(b, dtype=float)
                A = pd.DataFrame(A_tmp, dtype=float).values.T
                p_ = cvxObjectiveFun(p, A, b, solve='SCS', disp=True, maxiter=100)
                p_ = np.array(p_, dtype=float)
                tmp = {'posterior': p_, 'b': b, 'varcons': isvarcons, 'index': indices, 'A' : A}
                return dumps(tmp)
            else:
                tmp = {'posterior': p}
                return dumps(tmp)
    else:
        tmp = {'posterior': p}
        return dumps(tmp)

@app.callback(
    Output('intermediate-kde-value', 'children'),
    [Input('intermediate-value', 'children')])
def getkde(json_posterior):
    if (json_posterior is not None) and ('test' not in df.keys()):
        tmp = loads(json_posterior)
        pos = tmp['posterior']
        kde = getKDE(df.values, vars, pos, p)
        return kde.to_json(orient = 'split', double_precision=13)
    else:
        return None

@app.callback(
    Output('check-constraints', 'children'),
    [Input('intermediate-value' , 'children')]
)
def checkConstraints(json_data):
    if json_data is not None:
        tmp = loads(json_data)
        if 'b' in tmp.keys():
            b = tmp['b']
            pos = tmp['posterior']
            left_val = tmp['A'].dot(pos)
            diff = np.absolute(left_val - b)
            if np.all(diff <= 1e-3):
                return 'Constraints Satisfied'
            elif np.all(left_val <= b):
                return 'Constraints Satisfied'
            else:
                return 'Constraints Not Satisfied'
        else:
            return 'No Constraints Specified'


@app.callback(Output('tab-output', 'children'),
              [Input('tabs', 'value'),
               Input('intermediate-kde-value', 'children')])
def display_content(value, kde_json):
    if (kde_json is not None) and (value != 'test'):
        kde_df = pd.read_json(kde_json, dtype=float, orient='split')
        data = [
            go.Scatter(
                x=kde_df[value + '_x_grid'].values.tolist(),
                y=kde_df[value + '_posterior'].values.tolist(),
                name = 'Posterior',
                connectgaps = True,
                fill='tozeroy',
                mode='markers',
                marker = dict(size = 2)
            ),
            go.Scatter(
                x=kde_df[value + '_x_grid'],
                y=kde_df[value + '_prior'],
                name = 'Prior',
                fill='none',
                connectgaps=True,
                mode='None',
                marker=dict(size=2, color = 'r')
            )
        ]
        return html.Div([
            dcc.Graph(
                id = 'graph1',
                figure = {
                    'data' : data,
                    'layout' : {'margin' : {'l': 30,'r': 0,'b': 30,'t': 0},
                                'legend' : {'x' : -.1, 'y' : 1.2},
                                'font' : {'size' : 14}}
                }
            )
        ])
    else:
        return None

if __name__ == '__main__':
    app.run_server()