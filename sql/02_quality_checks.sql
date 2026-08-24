-- 1. 检查训练集和测试集行数
SELECT
    'house_train_raw' AS table_name,
    COUNT(*) AS row_count
FROM house_train_raw

UNION ALL

SELECT
    'house_test_raw' AS table_name,
    COUNT(*) AS row_count
FROM house_test_raw;


-- 2. 检查训练集 ID 是否重复
SELECT
    COUNT(*) AS total_rows,
    COUNT(DISTINCT id) AS unique_ids,
    COUNT(*) - COUNT(DISTINCT id) AS duplicated_ids
FROM house_train_raw;


-- 3. 检查目标变量是否缺失
SELECT
    COUNT(*) AS total_rows,
    COUNT(sale_price) AS non_null_sale_price,
    COUNT(*) - COUNT(sale_price) AS null_sale_price
FROM house_train_raw;


-- 4. 查看房价基本统计
SELECT
    MIN(sale_price) AS minimum_price,
    MAX(sale_price) AS maximum_price,
    AVG(sale_price) AS average_price,
    PERCENTILE_CONT(0.5)
        WITHIN GROUP (ORDER BY sale_price) AS median_price
FROM house_train_raw;


-- 5. 查看几个关键特征的缺失数量
SELECT
    COUNT(*) FILTER (
        WHERE lot_frontage IS NULL
    ) AS lot_frontage_nulls,

    COUNT(*) FILTER (
        WHERE garage_yr_blt IS NULL
    ) AS garage_yr_blt_nulls,

    COUNT(*) FILTER (
        WHERE electrical IS NULL
    ) AS electrical_nulls,

    COUNT(*) FILTER (
        WHERE mas_vnr_area IS NULL
    ) AS mas_vnr_area_nulls
FROM house_train_raw;


-- 6. 按社区查看数量和平均售价
SELECT
    neighborhood,
    COUNT(*) AS house_count,
    ROUND(AVG(sale_price), 2) AS average_sale_price
FROM house_train_raw
GROUP BY neighborhood
ORDER BY average_sale_price DESC;


-- 7. 查看整体质量和售价之间的关系
SELECT
    overall_qual,
    COUNT(*) AS house_count,
    ROUND(AVG(sale_price), 2) AS average_sale_price
FROM house_train_raw
GROUP BY overall_qual
ORDER BY overall_qual;


-- 8. 检查不合理数值
SELECT
    COUNT(*) FILTER (
        WHERE sale_price <= 0
    ) AS invalid_sale_price,

    COUNT(*) FILTER (
        WHERE gr_liv_area <= 0
    ) AS invalid_living_area,

    COUNT(*) FILTER (
        WHERE overall_qual NOT BETWEEN 1 AND 10
    ) AS invalid_overall_quality
FROM house_train_raw;