-- 1. 检查清洗表是否存在，以及行数是否保持不变
SELECT
    'house_train_raw' AS table_name,
    COUNT(*) AS row_count
FROM house_train_raw

UNION ALL

SELECT
    'house_train_clean',
    COUNT(*)
FROM house_train_clean

UNION ALL

SELECT
    'house_test_raw',
    COUNT(*)
FROM house_test_raw

UNION ALL

SELECT
    'house_test_clean',
    COUNT(*)
FROM house_test_clean;


-- 2. 清洗后 ID 是否仍然唯一
SELECT
    COUNT(*) AS total_rows,
    COUNT(DISTINCT id) AS unique_ids,
    COUNT(*) - COUNT(DISTINCT id) AS duplicated_ids
FROM house_train_clean;


-- 3. 目标变量是否合法
SELECT
    COUNT(*) FILTER (
        WHERE sale_price IS NULL
    ) AS missing_sale_price,

    COUNT(*) FILTER (
        WHERE sale_price <= 0
    ) AS invalid_sale_price,

    MIN(sale_price) AS minimum_sale_price,
    MAX(sale_price) AS maximum_sale_price
FROM house_train_clean;


-- 4. 检查“没有设施”类别是否已经被显式保存
SELECT
    COUNT(*) FILTER (
        WHERE alley = 'None'
    ) AS no_alley,

    COUNT(*) FILTER (
        WHERE bsmt_qual = 'None'
    ) AS no_basement,

    COUNT(*) FILTER (
        WHERE fireplace_qu = 'None'
    ) AS no_fireplace,

    COUNT(*) FILTER (
        WHERE garage_type = 'None'
    ) AS no_garage,

    COUNT(*) FILTER (
        WHERE pool_q = 'None'
    ) AS no_pool
FROM house_train_clean;


-- 5. 查看仍然存在的真正未知值
SELECT
    COUNT(*) FILTER (
        WHERE lot_frontage IS NULL
    ) AS unknown_lot_frontage,

    COUNT(*) FILTER (
        WHERE electrical IS NULL
    ) AS unknown_electrical,

    COUNT(*) FILTER (
        WHERE garage_yr_blt IS NULL
    ) AS unknown_or_absent_garage_year,

    COUNT(*) FILTER (
        WHERE mas_vnr_area IS NULL
    ) AS unknown_masonry_area
FROM house_train_clean;


-- 6. 检查几个主要字段的范围
SELECT
    MIN(overall_qual) AS minimum_quality,
    MAX(overall_qual) AS maximum_quality,

    MIN(gr_liv_area) AS minimum_living_area,
    MAX(gr_liv_area) AS maximum_living_area,

    MIN(year_built) AS oldest_year_built,
    MAX(year_built) AS newest_year_built
FROM house_train_clean;


-- 7. 检查异常大的居住面积记录
SELECT
    id,
    gr_liv_area,
    overall_qual,
    sale_price
FROM house_train_clean
WHERE gr_liv_area > 4000
ORDER BY gr_liv_area DESC;