from typing import TypedDict

import numpy as np
import pandas as pd
import streamlit as st
# from qpsolvers import solve_qp
from scipy.optimize import linprog
from scipy.sparse import csc_matrix


class Likelihood(TypedDict):
    A: float
    B: float
    C: float
    D: float


def normalize_likelihood(likelihood: pd.Series) -> Likelihood:
    total = sum(likelihood.values)
    return likelihood.apply(lambda x: x / total)


def sum_to_one(likelihood: pd.Series) -> Likelihood:
    total = sum(likelihood.values)
    complement = 1 - total
    if complement < 0:
        return normalize_likelihood(likelihood)
    if complement > 0:
        zero_value_count = sum([1 for v in likelihood.values if v == 0])
        fill_value = complement / zero_value_count
        likelihood = likelihood.apply(lambda x: x if x != 0 else fill_value)
    return likelihood


st.set_page_config(layout="wide")
st.title("Answer sequence optimizer")

# Decision variables: [A1, B1, C1, D1, A2, B2, C2, D2, A3, B3, C3, D3, A4, B4, C4, D4, A5, B5, C5, D5, A6, B6, C6, D6]
# Objective function coefficients: minimize pairwise distances of points with same line number

cols = st.columns(4)
n_letters = cols[0].number_input(label="Number of letters", value=4, min_value=0, max_value=100, on_change=st.session_state.pop, kwargs={'key': 'default_answer'})
n_questions = cols[1].number_input(label="Number of questions", value=6, min_value=0, max_value=100, on_change=st.session_state.pop, kwargs={'key': 'default_answer'})
letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'][:n_letters]

n_differences = n_questions * n_letters
n_variables = n_differences + n_questions * n_letters

score_type = cols[2].radio("Choose answer score type", options=['Manual', 'Random'], index=1)
if score_type == 'Random':
    if 'answer_seed' not in st.session_state:
        import time
        st.session_state.answer_seed = int(time.perf_counter_ns() % 256)
    np.random.seed(st.session_state.answer_seed)
    answer = np.random.randint(low=0, high=n_letters, size=n_questions).tolist()
    st.session_state.answer = [letters[i] for i in answer]
    with cols[3].expander(f"Show real answer for seed {st.session_state.answer_seed}"):
        st.warning(" ".join(st.session_state.answer))
else:
    st.session_state.answer = ['%'] * n_questions
    st.session_state.pop('answer_seed', None)


# Set the targets 
st.subheader("Set targets function")
np.random.seed(42)
data = np.random.random((n_questions, n_letters))
data = np.zeros((n_questions, n_letters))
df = pd.DataFrame(data, columns=letters)
df.index += 1
cols = st.columns(2)
df_target = cols[0].data_editor(df)

df_target = df_target.apply(lambda x: sum_to_one(x), axis=1)
cols[1].dataframe(df_target)
# df_target = df_target.apply(lambda x: normalize_likelihood(x.to_dict()))
targets = df_target.values

# Optimization Problem
########################################################################
# Objective: minimize sum of all differences
c = np.array([0] * (n_questions * n_letters) + [1] * n_differences)

# Inequality constraints (Ax <= b)
# Inequality constraints (Ax <= b)
A_ub = np.array([])
b_ub = np.array([])
for i in range(n_questions):
    for j in range(n_letters):
        cons = np.zeros(n_variables)
        cons[i * n_letters + j] = 1
        cons[n_letters * n_questions + i * n_letters + j] = -1
        A_ub = np.vstack((A_ub, cons)) if A_ub.size else cons
        b_ub = np.append(b_ub, targets[i, j])

        cons = np.zeros(n_variables)
        cons[i * n_letters + j] = -1
        cons[n_letters * n_questions + i * n_letters + j] = -1
        A_ub = np.vstack((A_ub, cons))
        b_ub = np.append(b_ub, -targets[i, j])

# Equality constraints (Ax = b)
A_eq = np.array([])
for i in range(n_questions):
    cons = np.zeros(n_variables)
    indexes = [i * n_letters + j for j in range(n_letters)]
    cons[indexes] = 1
    A_eq = np.vstack((A_eq, cons)) if A_eq.size else cons

b_eq = np.ones(len(A_eq))
bounds = [(0, 1) for _ in range(n_letters * n_questions)] + [(-1, 1) for _ in range(n_differences)]
########################################################################


# Add new constraints
def add_answer(A_eq, b_eq, answer, answer_score):
    answer_indexes = [i * n_letters + letters.index(a) for i, a in enumerate(answer)]
    cons = np.zeros(n_variables)
    cons[answer_indexes] = 1
    A_eq = np.vstack((A_eq, cons))
    b_eq = np.append(b_eq, answer_score)
    return A_eq, b_eq


### Create the user answers ###
st.divider()

dataA = ['A'] * n_questions
dataB = ['B'] * n_questions
dataC = ['C'] * n_questions

if 'default_answer' not in st.session_state:
    st.session_state.default_answer = [dataA, dataB, dataC]

cols = st.columns([2, 1, 1, 4])
if cols[2].button("Add answer"):
    st.session_state.default_answer.append(dataA)
if cols[2].button("Remove answer") and len(st.session_state.default_answer) > 1:
    st.session_state.default_answer.pop()
if cols[2].button("Use Opt answers"):
    st.session_state.default_answer.append(st.session_state.opt_answers)

cols[0].subheader("Answers")
cols[1].subheader("Scores")
for i in range(len(st.session_state.default_answer)):
    data = st.session_state.default_answer[i]
    default_answer = " ".join(data)
    answer = cols[0].text_input(f"Answer {i+1}", value=default_answer, key=f"answer {i}", label_visibility="collapsed")
    answer = answer.split(' ')
    assert len(answer) == n_questions and all([a in letters + [" "] for a in answer])

    if score_type == 'Random':
        answer_match = [a == b for a, b in zip(answer, st.session_state.answer)]
        answer_score = sum(answer_match)
        # # if cols[1].button("Show score", key=f"show_score {i}"):
        cols[1].text_input(f"Score {i+1}", answer_score, disabled=True, label_visibility="collapsed", key=f"answer_score {i}")
    else:
        answer_score = cols[1].number_input(
            label=f"Score {i+1}", 
            value=n_questions // n_letters, 
            min_value=0,
            max_value=n_questions, 
            key=f"answer_score {i}",
            label_visibility="collapsed"
        )

    A_eq, b_eq = add_answer(A_eq, b_eq, answer, answer_score)
    if answer_score == n_questions:
        st.balloons()

        

### Show the Optimization results ###
with cols[3]:
    # Solve
    with st.expander("Solver settings", expanded=True):
        # cols = st.columns(2)
        # solvers = ['osqp', 'scipy']
        # selected_solver = cols[0].radio("Choose the solver", options=solvers, index=1)
        # if selected_solver == 'osqp':
        #     P = csc_matrix(np.zeros((n_variables, n_variables)))
        #     q = c
        #     G = csc_matrix(A_ub)
        #     h = b_ub
        #     A = csc_matrix(A_eq)
        #     b = b_eq
        #     lb = np.array([elem[0] for elem in bounds])
        #     ub = np.array([elem[1] for elem in bounds])
        #     x = solve_qp(P, q, G, h, A, b, lb=lb, ub=ub, solver='osqp')
        #     if x is None:
        #         st.error("OSQP solver failed")
        #     else:
        #         st.info("OSQP solver finished")
 
        if True:
            var_type = st.radio("Variable type", options=['Continuous', 'Integer'], index=1)
            integrality = 1 if var_type == 'Integer' else 0
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, integrality=integrality)
            x = res.x
            if res.success:
                st.info(res.message)
            else:
                st.error(res.message)


    # Sow results
    if x is not None:
        data = x[:(n_letters * n_questions)].reshape(n_questions, n_letters)
        df = pd.DataFrame(data, columns=letters)
        df.index += 1
        st.subheader('Answers')
        answer_df = df.idxmax(axis=1).rename('Answer')
        likelihood_df = df.max(axis=1).rename('Likeliness')
        df_answer = pd.concat((answer_df, likelihood_df), axis=1)
        st.session_state.opt_answers = df.idxmax(axis=1).tolist()
        opt_answers_string = " ".join(st.session_state.opt_answers)
        if score_type == 'Random' and st.session_state.opt_answers == st.session_state.answer:
            st.success(f"{' '.join(opt_answers_string)}")
        else:
            st.warning(f"{" ".join(opt_answers_string)}")
        st.dataframe(df_answer.round(2).T, use_container_width=True)
        cols = st.columns(2)
        with cols[0]:
            st.subheader('Optimization result')
            st.dataframe(df.round(2))
        with cols[1]:
            st.subheader('Targets')
            st.dataframe(df_target.round(2))