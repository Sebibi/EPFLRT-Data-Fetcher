import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Quizz Answer App")

from typing import TypedDict


class Likelihood(TypedDict):
    A: float
    B: float
    C: float
    D: float


def create_likelihood(a: float, b: float, c: float, d: float) -> Likelihood:
    return {'A': a, 'B': b, 'C': c, 'D': d}


def random_likelihood() -> Likelihood:
    rand_nums = np.random.random(4)
    a, b, c, d = rand_nums / sum(rand_nums)
    return create_likelihood(a, b, c, d)


def normalize_likelihood(likelihood: Likelihood) -> Likelihood:
    total = sum(likelihood.values())
    return {k: v / total for k, v in likelihood.items()}


def sum_to_one(likelihood: Likelihood) -> Likelihood:
    total = sum(likelihood.values())
    complement = 1 - total
    if complement < 0:
        raise ValueError(f"Total is greater than 1: {total}")
    if complement > 0:
        zero_value_count = sum([1 for v in likelihood.values() if v == 0])
        fill_value = complement / zero_value_count
        return {k: v if v != 0 else fill_value for k, v in likelihood.items()}
    return likelihood

# Input the node count
node_types = ['A', 'B', 'C', 'D']

n_A = st.number_input(label="Number of A nodes", value=8, min_value=0, max_value=100)
n_B = st.number_input(label="Number of B nodes", value=4, min_value=0, max_value=100)
n_C = st.number_input(label="Number of C nodes", value=4, min_value=0, max_value=100)
n_D = st.number_input(label="Number of D nodes", value=3, min_value=0, max_value=100)

node_count = [n_A, n_B, n_C, n_D]
node_count_dict = dict(zip(node_types, node_count))
n_questions = sum(node_count)

# Load the data from load button
data = st.file_uploader("Upload a CSV file", type="csv")

if data is not None:
    node_types = ['A', 'B', 'C', 'D']
    init_answers_df = pd.read_csv(data, index_col=0)[:n_questions]
    st.subheader('Initial answers')
    st.dataframe(init_answers_df.T)

    init_answers = init_answers_df.to_dict('records')
    init_answers = [sum_to_one(x) for x in init_answers]
    init_answers = [normalize_likelihood(x) for x in init_answers]

    node_answers_type = [f"{t}{i + 1}" for t in node_types for i in range(node_count_dict[t])]
    assert len(
        node_answers_type) == n_questions, f"Number of answers: {len(node_answers_type)}, expected: {n_questions}"
    node_questions = [f"Q{i + 1}" for i in range(n_questions)]

    G = nx.Graph()
    G.add_nodes_from(node_answers_type, bipartite=0)
    G.add_nodes_from(node_questions, bipartite=1)

    for i, init_answer in enumerate(init_answers):
        question_node = node_questions[i]
        for j, (node_type, node_count) in enumerate(node_count_dict.items()):
            for k in range(node_count):
                answer_node = f"{node_type}{k + 1}"
                G.add_edge(question_node, answer_node, weight=init_answer[node_type])


    # Perform the matching
    matching = nx.max_weight_matching(G, maxcardinality=True)
    matching = [(v, k) if v in node_questions else (k, v) for k, v in dict(matching).items()]
    matching = {k: v for k, v in dict(matching).items() if k in node_questions}
    matching_df = pd.Series(matching)

    assert len(matching_df) == n_questions, f"Number of questions: {len(matching_df)}, expected: {n_questions}"

    # Process matching for nice display
    matching_df.index = matching_df.index.str.replace('Q', '')
    matching_df.index = matching_df.index.astype(int)
    matching_df = matching_df.sort_index()

    matching_df = matching_df.apply(lambda x: x[0])
    matching_df.index.name = 'Question'
    matching_df.name = 'Answer'

    matching_df_int = matching_df.apply(lambda x: node_types.index(x))
    matching_df_int.name = 'Answer_int'
    df = pd.concat([matching_df, matching_df_int], axis=1)

    df['Likeliness'] = df.apply(lambda x: init_answers[x.name - 1][x['Answer']], axis=1)
    init_answers_df = pd.DataFrame(init_answers)
    init_answers_df.index = init_answers_df.index + 1

    df = pd.concat([df, init_answers_df], axis=1)
    df.drop('Answer_int', axis=1, inplace=True)
    df = df.round(2)
    st.subheader('Matching')
    st.dataframe(df.T)
    st.info(f"Answers: {df['Answer'].tolist()}")

    cols = st.columns(2)
    cols[0].subheader('Matching view')
    cols[0].dataframe(df)

    cols[1].subheader('Matching description')
    cols[1].dataframe(df['Likeliness'].describe())