# 或者结合 .loc 拿过滤后的结果判断 empty
if not dataframe.loc[lambda df: df[TARGET_COLUMN] <= 0].empty:
    print("存在小于等于0的值")


# 方式 A：提取该列切片后判断 .any()
if dataframe.loc[dataframe[TARGET_COLUMN] <= 0, TARGET_COLUMN].any():
    ...  # 注意：如果全为0，False/0的bool可能会有细微差别，更稳妥的是用下面的 empty

# 方式 B：直接判断筛选后的结果是否为空（最稳妥）
if not dataframe.loc[dataframe[TARGET_COLUMN] <= 0].empty:
    print("存在小于等于0的值")

    dataframe[TARGET_COLUMN] <= 0


"""
.query() can pick rows and .filter() or .select_dtypes() or boolean mask can pick columns
如果你写：dataframe['price'] <= 0
➔ 这才会返回一个只有 True/False 的 Boolean 掩码（Series）：
[False, True, False]

但如果你写：dataframe.query("price <= 0")
➔ 它在后台已经把掩码应用到原表上了！它直接把你想要的那些数据行提取了出来，返回给你一个全新的 DataFrame：
"""

"""
result = (
    raw_df
    .dropna(subset=['price'])
    .assign(double_price = lambda x: x['price'] * 2)
    
    # ✅ 对比这里！换成了 lambda df: df['double_price'] > 100
    .loc[lambda df: df['double_price'] > 100] 
)
"""

"""
IIFE（Immediately Invoked Function Expression，立即调用函数表达式）

# 第一步：先定义函数（起个名字叫 check）
def check(df):
    return df['price'] <= 0

# 第二步：在需要的地方“调用”它
result = check(dataframe)

# 定义的瞬间（用括号包起来），后面紧接着跟一个 (dataframe) 把它运行掉！
result = (lambda df: df['price'] <= 0)(dataframe)
场景：你只想做一次临时计算，连函数名都懒得起
假设你有一项非常奇葩、只用一次的计算逻辑，你不想专门写个 def my_weird_function() 扔在代码顶端占地方，又不想拆成多行写一堆临时变量：
# 直接用 IIFE 搞定一个临时的计算结界：
is_valid = (lambda df: (df['a'] > 0) & (df['b'] < 100))(dataframe)
"""

if (sales_data[STOCK_COL] < 10).any():
    print("库存里存在小于10的记录")

if sales_data.query(f"{STOCK_COL}< 10").empty:
    print("库存里不存在小于10的记录")


def clean_and_over100(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    df["price"] = pd.to_numeric(
        df["price"],
        errors="coerce",
    )

    result_df = df.loc[df["price"] > 100, ["price"]]
    return result_df

    orders = orders.loc[(orders["amount"] > 0)&(orders["amount"] <= 1000,), :]

    orders = (orders
        .pipe(lambda df: df.loc[(df["amount"] > 0),:])
        .pipe(lambda df: df.loc[(df["amount"] <= 1000),:])
    )
           