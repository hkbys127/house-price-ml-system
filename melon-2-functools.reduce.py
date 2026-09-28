import pandas as pd

from src.clean import (
    fill_semantic_absence,
    normalize_text_columns,
    replace_infinite_values,
)

dataframe = pd.read_csv("your_data.csv")
dataframe = normalize_text_columns(dataframe)
dataframe = replace_infinite_values(dataframe)
dataframe = fill_semantic_absence(dataframe)

"""
use pipe()
"""
dataframe = (
    dataframe.pipe(normalize_text_columns)
    .pipe(replace_infinite_values)
    .pipe(fill_semantic_absence)
)
"""
if need more arguments
"""
dataframe = (
    dataframe.pipe(normalize_text_columns)
    .pipe(replace_infinite_values, value=0)  # type: ignore
    .pipe(fill_semantic_absence)
)

"""
use functools.reduce
"""
from functools import reduce

pipeline = [normalize_text_columns, replace_infinite_values, fill_semantic_absence]
dataframe = reduce(lambda df, func: func(df), pipeline, dataframe)


"""
why use lambda df(if not use)
"""

# 先单独定义一个函数
def apply_func(data, action):
    return action(data)


# 然后传给 reduce
dataframe = reduce(apply_func, pipeline, dataframe)


