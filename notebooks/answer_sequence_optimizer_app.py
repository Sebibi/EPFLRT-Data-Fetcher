import numpy as np
import pandas as pd
from itertools import combinations
from qpsolvers import solve_qp
from scipy.optimize import linprog
from scipy.sparse import csc_matrix
import streamlit as st


st.set_page_config(layout="wide")
st.title("Answer sequence optimizer")

# Decision variables: [A1, B1, C1, D1, A2, B2, C2, D2, A3, B3, C3, D3, A4, B4, C4, D4, A5, B5, C5, D5, A6, B6, C6, D6]
# Objective function coefficients: minimize pairwise distances of points with same line number


n_letters = st.number_input(label="Number of letters", value=4, min_value=0, max_value=100)
n_questions = st.number_input(label="Number of questions", value=10, min_value=0, max_value=100)
letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'][:n_letters]


n_letter_comb = len(list(combinations(letters, 2)))
n_differences = n_questions * n_letter_comb
n_variables = n_differences + n_questions * n_letters


# Objective: minimize sum of all differences
c = np.array([0] * (n_questions * n_letters) + [1] * n_differences)

# Inequality constraints (Ax <= b)
A_ub = np.array([])
for i in range(n_questions):
    indexes = [i * n_letters + j for j in range(n_letters)]
    indexes_comb = list(combinations(indexes, 2))
    for j, (index1, index2) in enumerate(indexes_comb):
        cons = np.zeros(n_variables)
        cons[index1] = 1
        cons[index2] = -1
        cons[n_letters * n_questions + i * n_letter_comb + j] = -1
        A_ub = np.vstack((A_ub, cons)) if A_ub.size else cons

        cons = np.zeros(n_variables)
        cons[index1] = -1
        cons[index2] = 1
        cons[n_letters * n_questions + i * n_letter_comb + j] = -1
        A_ub = np.vstack((A_ub, cons))

b_ub = np.zeros(len(A_ub))
# Equality constraints (Ax = b)
A_eq = np.array([])
for i in range(n_questions):
    cons = np.zeros(n_variables)
    indexes = [i * n_letters + j for j in range(n_letters)]
    cons[indexes] = 1
    A_eq = np.vstack((A_eq, cons)) if A_eq.size else cons

b_eq = np.ones(len(A_eq))

# Bounds
bounds = [(0, 1) for _ in range(n_letters * n_questions)] + [(-1, 1) for _ in range(n_differences)]


# Add new constraints
def add_answer(A_eq, b_eq, answer, answer_score):
    answer_indexes = [i * n_letters + letters.index(a) for i, a in enumerate(answer)]
    cons = np.zeros(n_variables)
    cons[answer_indexes] = 1
    A_eq = np.vstack((A_eq, cons))
    b_eq = np.append(b_eq, answer_score)
    return A_eq, b_eq


n_answers = st.number_input(label="Number of answers", value=3, min_value=0, max_value=100)

dataA = np.zeros((n_questions, n_letters)).astype(bool)
dataA[:, 0] = True

dataB = np.zeros((n_questions, n_letters)).astype(bool)
dataB[:, 1] = True

dataC = np.zeros((n_questions, n_letters)).astype(bool)
dataC[:, 2] = True

for i in range(n_answers):
    with st.expander(f"Answer {i + 1}"):
        cols = st.columns([2, 1])
        if i == 0:
            data = dataA
        elif i == 1:
            data = dataB
        elif i == 2:
            data = dataC
        else:
            data = np.zeros((n_questions, n_letters)).astype(bool)
        df_data = pd.DataFrame(data, columns=letters)
        df_data.index += 1
        df_data.index.name = 'Question'

        df_data = cols[0].data_editor(df_data.T, key=f"answer {i}")
        answer = df_data.T.idxmax(axis=1).tolist()
        answer_score = cols[1].number_input(label="Answer score", value=n_questions // n_letters, min_value=0, max_value=100, key=f"answer_score {i}")
        A_eq, b_eq = add_answer(A_eq, b_eq, answer, answer_score)


# Solve

solvers = ['osqp', 'scipy']
selected_solver = st.radio("Choose the solver", options=solvers, index=0)
if selected_solver == 'osqp':
    P = csc_matrix(np.zeros((n_variables, n_variables)))
    q = c
    G = csc_matrix(A_ub)
    h = b_ub
    A = csc_matrix(A_eq)
    b = b_eq
    lb = np.array([elem[0] for elem in bounds])
    ub = np.array([elem[1] for elem in bounds])
    x = solve_qp(P, q, G, h, A, b, lb=lb, ub=ub, solver='osqp')

else:
    var_type = st.radio("Variable type", options=['Continuous', 'Integer'], index=0)
    integrality = 1 if var_type == 'Integer' else 0
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, integrality=integrality)
    x = res.x

# Show results
data = x[:(n_letters * n_questions)].reshape(n_questions, n_letters)
df = pd.DataFrame(data, columns=letters)
df.index += 1
df.index.name = 'Question'

cols = st.columns(2)
with cols[0]:
    st.subheader('Optimization result')
    st.dataframe(df.round(2))

with cols[1]:
    st.subheader('Answers')
    answer_df = df.idxmax(axis=1).rename('Answer')
    likelihood_df = df.max(axis=1).rename('Likeliness')
    df_answer = pd.concat((answer_df, likelihood_df), axis=1)
    st.dataframe(df_answer.round(2))
    st.info(f"Answers: {df.idxmax(axis=1).tolist()}")