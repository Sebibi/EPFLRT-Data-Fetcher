import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import linprog
import time
from stqdm import stqdm


def normalize_likelihood(likelihood: pd.Series) -> pd.Series:
    total = sum(likelihood.values)
    return likelihood.apply(lambda x: x / total)

def sum_to_one2(likelihood: pd.Series) -> pd.Series:
    total = sum(likelihood.values)
    complement = 1 - total
    if complement < 0:
        return normalize_likelihood(likelihood)
    if complement > 0:
        zero_value_count = sum([1 for v in likelihood.values if v == 0])
        fill_value = complement / zero_value_count
        likelihood = likelihood.apply(lambda x: x if x != 0 else fill_value)
    return likelihood  

def sum_to_one(likelihood: pd.Series) -> pd.Series:
    total = sum(likelihood.values)
    if total == 0:
        return likelihood.apply(lambda x: 1 / len(likelihood))
    return likelihood.apply(lambda x: x / total)


def create_problem(targets: np.ndarray, user_answers: list[list[str]], user_answer_scores: list[int]) -> tuple:
    """
    Create the optimization problem
    # Decision variables: [A1, B1, C1, D1, A2, B2, C2, D2, A3, B3, C3, D3, A4, B4, C4, D4, A5, B5, C5, D5, A6, B6, C6, D6]
    # Objective function coefficients: minimize pairwise distances of points with same line number

    """
    # Variables [#answers, #differences]
    n_letters = st.session_state.n_letters
    n_questions = st.session_state.n_questions
    n_differences = n_questions * n_letters
    n_variables = 2 * n_questions * n_letters

    # Objective: minimize sum of all differences
    c = np.array([0] * (n_questions * n_letters) + [1] * n_differences)

    # Inequality constraints (Ax <= b)
    # 
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
    # Each question must have exactly one answer
    A_eq = np.array([])
    for i in range(n_questions):
        cons = np.zeros(n_variables)
        indexes = [i * n_letters + j for j in range(n_letters)]
        cons[indexes] = 1
        A_eq = np.vstack((A_eq, cons)) if A_eq.size else cons

    b_eq = np.ones(len(A_eq))
    bounds = [(0, 1) for _ in range(n_letters * n_questions)] + [(-1, 1) for _ in range(n_differences)]
    ########################################################################

    for answer, answer_score in zip(user_answers, user_answer_scores):
        A_eq, b_eq = add_answer(A_eq, b_eq, answer, answer_score)
    return c, A_ub, b_ub, A_eq, b_eq, bounds


# Add new constraints related to the answer and its score
def add_answer(A_eq, b_eq, answer, answer_score):
    n_letters = st.session_state.n_letters
    n_questions = st.session_state.n_questions
    n_variables = 2 * n_questions * n_letters 
    letters = st.session_state.letters


    answer_indexes = [i * n_letters + letters.index(a) for i, a in enumerate(answer)]
    cons = np.zeros(n_variables)
    cons[answer_indexes] = 1
    A_eq = np.vstack((A_eq, cons))
    b_eq = np.append(b_eq, answer_score)
    return A_eq, b_eq


#### Streamlit App ####

st.set_page_config(layout="wide")
st.title("Answer sequence optimizer")

with st.sidebar:
    ### Quizz Settings ###
    cols = st.columns(2)
    st.session_state.n_letters = cols[0].number_input(label="Letters", value=4, min_value=0, max_value=100, on_change=st.session_state.pop, kwargs={'key': 'default_answer'})
    st.session_state.n_questions = cols[1].number_input(label="Questions", value=6, min_value=0, max_value=100, on_change=st.session_state.pop, kwargs={'key': 'default_answer'})
    st.session_state.letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'][:st.session_state.n_letters]
    if 'default_answer' not in st.session_state:
        st.session_state.default_answer = [[l] * st.session_state.n_questions for l in ['A', 'B', 'C']]
    
    ### Create the Real Answer ###
    score_type = st.radio("Choose answer score type", options=['Manual', 'Random'], index=1)
    if score_type == 'Random':
        if 'answer_seed' not in st.session_state:
            st.session_state.answer_seed = int(time.perf_counter_ns() % 256)
        np.random.seed(st.session_state.answer_seed)
        answer = np.random.randint(
            low=0, 
            high=st.session_state.n_letters, 
            size=st.session_state.n_questions
        ).tolist()
        st.session_state.answer = [st.session_state.letters[i] for i in answer]
        with st.expander(f"Seed {st.session_state.answer_seed} Answers"):
            st.warning(" ".join(st.session_state.answer))
    else:
        st.session_state.answer = ['%'] * st.session_state.n_questions
        st.session_state.pop('answer_seed', None)


### Create the targets  ###
with st.expander("Set targets", expanded=True):
    st.subheader("Set targets function")
    data = np.zeros((st.session_state.n_questions, st.session_state.n_letters))
    df = pd.DataFrame(data, columns=st.session_state.letters)
    df.index += 1
    cols = st.columns(2)
    df_target = cols[0].data_editor(df)
    df_target = df_target.apply(lambda x: sum_to_one(x), axis=1)
    cols[1].dataframe(df_target)
    targets = df_target.values


with st.sidebar:
    st.divider()
    if st.button("Add answer"):
        st.session_state.default_answer.append(['A'] * st.session_state.n_questions)
    if st.button("Remove answer") and len(st.session_state.default_answer) > 1:
        st.session_state.default_answer.pop()
    if st.button("Use Opt answers"):
        st.session_state.default_answer.append(st.session_state.opt_answers)


### Create the User Answers ###
cols = st.columns([2, 1, 1, 8])

cols[0].markdown("Answer")
cols[1].markdown("Score")
user_answers = [None for _ in range(len(st.session_state.default_answer))]
user_answers_score = [None for _ in range(len(st.session_state.default_answer))]
for i in range(len(st.session_state.default_answer)):
    default_answer = " ".join(st.session_state.default_answer[i])
    answer = cols[0].text_input(f"Answer {i+1}", value=default_answer, key=f"answer {i}", label_visibility="collapsed")
    answer = answer.split(' ')
    assert len(answer) == st.session_state.n_questions and all([a in st.session_state.letters + [" "] for a in answer])

    if score_type == 'Random':
        answer_score = sum([a == b for a, b in zip(answer, st.session_state.answer)])
        cols[1].text_input(f"Score {i+1}", answer_score, disabled=True, label_visibility="collapsed", key=f"answer_score {i}")
    else:
        answer_score = cols[1].number_input(
            label=f"Score {i+1}", value=1, 
            min_value=0, max_value=st.session_state.n_questions, 
            key=f"answer_score {i}", label_visibility="collapsed"
        )
    user_answers[i] = answer
    user_answers_score[i] = answer_score

    if answer_score == st.session_state.n_questions:
        st.balloons()


### Solve the Optimization problem ###
with cols[3]:
    with st.expander("Solver settings", expanded=False):
        var_type = st.radio("Variable type", options=['Continuous', 'Integer'], index=1)
        integrality = 1 if var_type == 'Integer' else 0
        n_iter = st.number_input("Other solutions iter", value=100, min_value=0, max_value=1000, key='iterations')

        c, A_ub, b_ub, A_eq, b_eq, bounds = create_problem(targets, user_answers, user_answers_score)
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, integrality=integrality)
        x = res.x
        if res.success:
            st.info(res.message)
        else:
            st.error(res.message)


    ### Sow the Optimization results ###
    if x is not None:
        n_letters = st.session_state.n_letters
        n_questions = st.session_state.n_questions
        letters = st.session_state.letters

        data = x[:(n_letters * n_questions)].reshape(n_questions, n_letters)
        df = pd.DataFrame(data, columns=letters)
        df.index += 1
        st.subheader('Answers')
        answer_df = df.idxmax(axis=1).rename('Answer')
        likelihood_df = df.max(axis=1).rename('Likeliness')
        df_answer = pd.concat((answer_df, likelihood_df), axis=1)
        st.session_state.opt_answers = df.idxmax(axis=1).tolist()
        opt_answers_string = " ".join(st.session_state.opt_answers)
        
        cols = st.columns(2)
        if score_type == 'Random' and st.session_state.opt_answers == st.session_state.answer:
            cols[0].success(f"{' '.join(opt_answers_string)}")
        else:
            cols[0].warning(f"{' '.join(opt_answers_string)}")
        other_res = cols[1].expander("Other feasible results", expanded=True)

        cols = st.columns(2)
        st.dataframe(df_answer.round(2).T, use_container_width=True)
        with cols[0]:
            st.subheader('Results')
            st.dataframe(df.round(2))
        with cols[1]:
            st.subheader('Targets')
            st.dataframe(df_target.round(2))

if var_type == 'Integer' and res.success:
    targets2 = targets.copy()
    # cols = st.columns(2)
    all_res = [None for _ in range(n_iter)]
    with other_res:
        for i in stqdm(range(n_iter)):
            c, A_ub, b_ub, A_eq, b_eq, bounds = create_problem(targets2, user_answers, user_answers_score)
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, integrality=integrality)

            data = res.x[:(n_letters * n_questions)].reshape(n_questions, n_letters)
            df_data = pd.DataFrame(data, columns=letters)
            df_targets = pd.DataFrame(targets2, columns=letters)
            df_data.index += 1
            df_targets.index += 1
            df_new_targets = (df_targets - df_data * 0.1).copy()
            df_new_targets[df_new_targets < 0] = 0
            targets2 = df_new_targets.apply(lambda x: sum_to_one(x), axis=1).values
            opt_answers = df_data.idxmax(axis=1).tolist()
            opt_answers_string = " ".join(opt_answers)
            all_res[i] = opt_answers_string

        for r in set(all_res) - set([" ".join(st.session_state.opt_answers)]):
            st.warning(r)

