import pandas as pd
from random import sample
import numpy as np
from statistics import geometric_mean
import itertools


def random_draft(brawlers):
    draft = sample(brawlers, 6)
    return (draft[0:3], draft[3:6])


def evaluate_draft(draft):
    blue = draft[0]
    red = draft[1]
    subframe = matchups.loc[blue, red]
    values = subframe.values

    # evaluation_metric = vectorwise_norms
    # evaluation_metric = selective_geometric_mean
    evaluation_metric = combinatorial_max

    score = evaluation_metric(values)
    return score


# Evaluation metrics
def vectorwise_norms(matrix):
    column_norms = np.linalg.norm(matrix, axis=0) / np.sqrt(3)
    row_norms = np.linalg.norm(matrix, axis=1) / np.sqrt(3)
    score = geometric_mean(list(column_norms) + list(row_norms))
    return score


def selective_geometric_mean(matrix):
    columnwise_score = np.sqrt(np.dot(np.max(matrix, 0), np.median(matrix, 0)))
    rowwise_score = np.sqrt(np.dot(np.max(matrix, 1), np.median(matrix, 1)))
    score = np.sqrt(columnwise_score * rowwise_score)
    return score


def combinatorial_max(matrix):
    candidate_scores = []
    l = [0, 1, 2]
    P = itertools.permutations(l)
    for p in P:
        candidate_score = matrix[0, p[0]] * matrix[1, p[1]] * matrix[2, p[2]]
        candidate_scores += [candidate_score]
    score = np.linalg.norm(candidate_scores, 3)
    return score


#

class Node:
    def __init__(self, val=[], children=[]):
        self.val = val
        self.children = children
        #self.optimalDraft = optimalDraft


def buildTree(brawlers):
    node = Node()
    addChildren(node, brawlers)
    return node


def addChildren(node, brawlers):
    if len(node.val) == 5:
        node.children = [Node(node.val + [pick]) for pick in set(brawlers) - set(node.val)]
    elif len(node.val) == 3 or len(node.val) == 1:
        couples = list(itertools.combinations(set(brawlers) - set(node.val), 2))
        node.children = [Node(node.val + list(couple)) for couple in couples]
        for child in node.children:
            addChildren(child, brawlers)
    elif len(node.val) == 0:
        node.children = [Node(node.val + [pick]) for pick in brawlers]
        for child in node.children:
            addChildren(child, brawlers)


def minimax(node, maximizingPlayer):
    if len(node.val) == 6:
        blue = [node.val[0]] + node.val[3:5]
        red = node.val[1:3] + [node.val[5]]
        draft = (blue, red)
        return evaluate_draft(draft), draft

    optimalDraft = ()
    if maximizingPlayer:
        maxEva = -np.inf

        for child in node.children:
            eva, draft = minimax(child, False)
            if eva > maxEva:
                maxEva = eva
                optimalDraft = draft
        return (maxEva, optimalDraft)

    else:
        minEva = +np.inf

        for child in node.children:
            eva, draft = minimax(child, True)
            if eva < minEva:
                minEva = eva
                optimalDraft = draft
        return (minEva, optimalDraft)


if __name__ == '__main__':
    matchups_file = 'C:/Users/Alessandro/Documents/Ennio/matchups.xlsx'
    matchups = pd.read_excel(matchups_file, index_col=0)
    # matchups = pd.DataFrame.transpose(matchups)
    brawlers = list(matchups.columns)
    print(brawlers)
    draft = random_draft(brawlers)
    # draft = (['Spike', 'Crow', 'Tara'], ['Leon', 'Stu', 'Carl'])
    # draft = (['Carl', 'Amber', 'Tara'], ['Crow', 'Spike', 'Stu'])
    # draft = (draft[1], draft[0])
    # print(evaluate_draft(draft))
    tree = buildTree(brawlers)
    maxEva, optimalDraft = minimax(tree, True)
    print(maxEva)
    print(optimalDraft)
    pass
