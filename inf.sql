WITH my_cte AS (
    SELECT base_sku, base_source_store, comp_source_store, lower(comp_sku) as comp_sku 
    FROM ml_temp_db.ctc_inf_1507
),
crawl_attempt AS (
    SELECT
        LOWER(CAST(a.base_sku AS varchar)) AS base_sku,
        CASE WHEN b.product_title is null THEN FALSE ELSE TRUE end as base_sku_in_catalog,
        b.upc as upc,
        b.product_title as base_title,
        comp_sku,
        comp_source_store,
        base_source_store,
        b.segment,
        b.manufacturer_part_number as base_mpn
    FROM my_cte as a
    LEFT JOIN (
        Select sku, product_title, upc, segment, manufacturer_part_number
        FROM bungee_customercatalog.athena_auroradb_catalog
        WHERE source = 'ctc' 
        and capture_date = (
            select max(capture_date) 
            from bungee_customercatalog.athena_auroradb_catalog
            where source = 'ctc'
        )
        and (is_active = 'True' AND is_discontinued = 'False')
    ) as b
    ON(lower(cast(a.base_sku as varchar)) = lower(b.sku))
),
-- select * from crawl_attempt
deep_crawl AS (
    SELECT DISTINCT LOWER(sku) AS dc_sku, LOWER(upc) AS dc_upc, LOWER(CONCAT_WS('_', source_name, store_name)) AS dc_source_store,MIN(capture_date) AS comp_sku_first_date
    FROM "bungeedatalake"."bungee_competitiveintelligence_datalake"
    WHERE year IN ('2026','2025' ,'2024')
    AND source_name IN ('amazonca','homedepotca','homehardwareca','ronaca','walmartca')
    GROUP BY 1,2,3
),
pdp AS (
    SELECT 
        DISTINCT LOWER(sku) AS pdp_sku, 
        product_title AS comp_title, 
        product_url AS comp_url, 
        LOWER(upc) AS pdp_upc, 
        manufacturer_part_number AS pdp_mpn,
        REPLACE(source_store, '<>', '_') AS pdp_source_store,
        year||month||day as pdp_date
    FROM pdp_newdev.product_warehouse
    WHERE source_store IN ('amazonca<>amazonca', 'homedepotca<>homedepotca', 'homehardwareca<>homehardwareca','ronaca<>ronaca', 'walmartca<>walmartca') 
    AND product_segment = 'gm'
),
fl_dump AS (
    Select DISTINCT 
        lower(base_sku) as base_sku, 
        replace(base_source_store,'<>','_') as base_source_store,
        lower(comp_sku) as comp_sku, 
        replace(comp_source_store,'<>','_') as comp_source_store, 
        score, match_date, inserted_date, answer, array_distinct(array_agg(queue_name)) as fastlane_queue_name
    FROM ml_internal_uat.fastlane_dump
    WHERE base_source_store = 'ctc_ctc'
    AND comp_source_store IN ('amazonca_amazonca', 'homedepotca_homedepotca', 'homehardwareca_homehardwareca', '_', 'ronaca_ronaca', 'walmartca_walmartca')
    AND segment = 'gm'
    GROUP BY 1,2,3,4,5,6,7,8
),
csf_upc as(
    Select DISTINCT 
        split_part(uuid_a,'<>',1) as base_upc, 
        lower(split_part(uuid_a,'<>',2)) as base_sku,
        split_part(uuid_b,'<>',1) as comp_upc, 
        lower(split_part(uuid_b,'<>',2)) as comp_sku, 
        score, 
        replace(split(uuid_a,'<>',3)[3],'<>','_') as base_source_store,
        replace(split(uuid_b,'<>',3)[3],'<>','_') as comp_source_store, 
        MIN(year||month||day) as upc_sugg_date
    FROM product_seam_prod.type_upc_matches
    WHERE split(uuid_a,'<>',3)[3] = 'ctc<>ctc'
    AND split(uuid_b,'<>',3)[3] IN ('amazonca<>amazonca', 'homedepotca<>homedepotca', 'homehardwareca<>homehardwareca','ronaca<>ronaca', 'walmartca<>walmartca') 
    AND segment = 'gm'
    GROUP BY 1,2,3,4,5,6,7

),
csf_mpn as(
    Select DISTINCT 
        lower(split_part(uuid_a,'<>',1)) as base_sku,
        lower(split_part(uuid_b,'<>',2)) as comp_sku, 
        score, 
        replace(base_source_store,'<>','_') as base_source_store,
        replace(comp_source_store,'<>','_') as comp_source_store, 
        MIN(year||month||day) as mpn_sugg_date
    FROM product_seam_prod.type_mpn_matches
    WHERE base_source_store = 'ctc<>ctc'
    AND comp_source_store IN ('amazonca<>amazonca', 'homedepotca<>homedepotca', 'homehardwareca<>homehardwareca','ronaca<>ronaca', 'walmartca<>walmartca') 
    AND segment = 'gm'
    GROUP BY 1,2,3,4,5
),
match_lib AS (
    SELECT
        LOWER(base_sku) AS matchlib_sku, 
        base_upc AS matchlib_upc, 
        comp_sku AS matchlib_comp, 
        comp_upc AS matchlib_comp_upc,
        base_source_store, comp_source_store, match_date, matcher_comments,
        MAX(CASE WHEN deleted_date IS NULL THEN true ELSE false END) AS is_present_in_match_library,
        MAX(CASE WHEN deleted_date IS NOT NULL THEN true ELSE false END) AS is_deleted_in_match_library,
        MAX(deleted_date) AS last_deleted_date,
        MIN(load_date) AS match_library_inserted_date
    FROM match_library.match_library_snapshot 
    WHERE active = true and deleted_date is null 
    AND base_source_store = 'ctc_ctc' 
    and comp_source_store IN ('amazonca_amazonca', 'homedepotca_homedepotca', 'homehardwareca_homehardwareca', '_', 'ronaca_ronaca', 'walmartca_walmartca')
    and load_date = '2026-07-15'
    GROUP BY 1,2,3,4,5,6,7,8
),
csf_ml_input as(
    SELECT LOWER(sku) as inp_sku, product_title as inp_title, upc, source_store as inp_source_store, capture_date as inp_capture_date
    FROM product_seam_prod.checkpoint_precision_model_type_ml_input  
    WHERE year||month||day IN (select max(year||month||day) from product_seam_prod.checkpoint_precision_model_type_ml_input  WHERE source_store IN ('amazonca_amazonca', 'homedepotca_homedepotca', 'homehardwareca_homehardwareca', 'ronaca_ronaca', 'walmartca_walmartca', 'ctc_ctc')) 
    AND source_store IN ('amazonca_amazonca', 'homedepotca_homedepotca', 'homehardwareca_homehardwareca', 'ronaca_ronaca', 'walmartca_walmartca', 'ctc_ctc') AND segment = 'gm'
    UNION 
    SELECT LOWER(sku) as inp_sku, product_title as inp_title, upc, source_store as inp_source_store, capture_date as inp_capture_date
    FROM product_seam_prod.checkpoint_generic_model_type_ml_input  
    WHERE year||month||day IN (select max(year||month||day) from product_seam_prod.checkpoint_generic_model_type_ml_input  WHERE source_store IN ('amazonca_amazonca', 'homedepotca_homedepotca', 'homehardwareca_homehardwareca', 'ronaca_ronaca', 'walmartca_walmartca', 'ctc_ctc')) 
    AND source_store IN ('amazonca_amazonca', 'homedepotca_homedepotca', 'homehardwareca_homehardwareca', 'ronaca_ronaca', 'walmartca_walmartca', 'ctc_ctc') AND segment = 'gm'
),
csf_precision as(
    Select DISTINCT 
        lower(split_part(base_sku_uuid,'<>',1)) as base_sku,
        split_part(comp_sku_uuid,'<>',1) as comp_sku, 
        score, 
        replace(base_source_store,'<>','_') as base_source_store,
        replace(comp_source_store,'<>','_') as comp_source_store, 
        MIN(year||month||day) as first_sugg_date, 
        max(score) as model_score 
    from product_seam_prod.checkpoint_precision_model_type_directed_pairs  
    where base_source_store = 'ctc<>ctc'
    AND comp_source_store IN ('amazonca<>amazonca', 'homedepotca<>homedepotca', 'homehardwareca<>homehardwareca','ronaca<>ronaca', 'walmartca<>walmartca') 
    AND segment = 'gm'
    GROUP BY 1,2,3,4,5
),
csf_generic as (
    Select DISTINCT 
        split_part(base_sku_uuid,'<>',1) as base_sku,
        split_part(comp_sku_uuid,'<>',1) as comp_sku, 
        score, 
        replace(base_source_store,'<>','_') as base_source_store,
        replace(comp_source_store,'<>','_') as comp_source_store, 
        MIN(year||month||day) as first_sugg_date, 
        max(score) as model_score
    from product_seam_prod.checkpoint_generic_model_type_directed_pairs  
    where base_source_store = 'ctc<>ctc'
    AND comp_source_store IN ('amazonca<>amazonca', 'homedepotca<>homedepotca', 'homehardwareca<>homehardwareca','ronaca<>ronaca', 'walmartca<>walmartca') 
    AND segment = 'gm'
    GROUP BY 1,2,3,4,5
),
csf_semantic as (
    Select DISTINCT 
        split_part(base_sku_uuid,'<>',1) as base_sku,
        split_part(comp_sku_uuid,'<>',1) as comp_sku, 
        score, 
        replace(base_source_store,'<>','_') as base_source_store,
        replace(comp_source_store,'<>','_') as comp_source_store, 
        MIN(year||month||day) as first_sugg_date, 
        max(score) as model_score
    from product_seam_prod.checkpoint_semantic_minilm_l6_v2_type_directed_pairs  
    where base_source_store = 'ctc<>ctc'
    AND comp_source_store IN ('amazonca<>amazonca', 'homedepotca<>homedepotca', 'homehardwareca<>homehardwareca','ronaca<>ronaca', 'walmartca<>walmartca') 
    AND segment = 'gm'
    GROUP BY 1,2,3,4,5
),
csf_queue as (
    Select DISTINCT 
        lower(base_sku) as base_sku, 
        base_source_store, 
        lower(comp_sku) as comp_sku, 
        comp_source_store, 
        MAX(aggregated_score) as queue_score, 
        MIN(year||month||day) as queue_date, 
        array_distinct(array_agg(queue_name)) as queue_name
    FROM product_seam_prod.type_queue
    WHERE tenant = 'ctc'
    AND segment = 'gm'
    GROUP BY 1,2,3,4
),
afm AS (
    select 
        product_segment AS segment,
        company_code AS tenant,
        base_sku, search_key, search_key_type, search_type, comp_source_store,
        concat(base_sku,'<>',replace(comp_source_store,'_','<>')) as afm_request_key
    from ml_internal.afm_request_data 
    where company_code = 'ctc' 
    and comp_source_store IN ('amazonca_amazonca', 'homedepotca_homedepotca', 'homehardwareca_homehardwareca', 'ronaca_ronaca', 'walmartca_walmartca') 
    AND product_segment = 'gm'
),
result AS (
    SELECT
        a.*,
        CASE WHEN dc_sku IS NOT NULL THEN TRUE ELSE FALSE END AS sku_in_dc,
        comp_sku_first_date,
        dc_upc,
        CASE WHEN dc_upc IS NOT NULL THEN TRUE ELSE FALSE END AS upc_in_dc,
        CASE WHEN pdp_sku IS NOT NULL THEN TRUE ELSE FALSE END AS sku_in_pdp,
        pdp_date,
        pdp_upc,
        pdp_mpn,
        CASE WHEN pdp_upc IS NOT NULL THEN TRUE ELSE FALSE END AS upc_in_pdp,
        CASE WHEN f.inserted_date IS NOT NULL THEN TRUE ELSE FALSE END AS is_fl_attempted,
        f.inserted_date,
        f.answer,
        f.fastlane_queue_name,
        comp_title,
        CASE WHEN h.is_present_in_match_library THEN TRUE ELSE FALSE END AS is_present_in_match_library,
        h.is_deleted_in_match_library,
        h.last_deleted_date,
        h.match_library_inserted_date,
        h.matcher_comments,
        REGEXP_REPLACE(a.upc, '^0+', '') AS normalized_base_upc,
        REGEXP_REPLACE(b.dc_upc, '^0+', '') AS normalized_dc_upc,
        REGEXP_REPLACE(c.pdp_upc, '^0+', '') AS normalized_pdp_upc,
        CASE
            WHEN REGEXP_REPLACE(a.upc, '^0+', '') = REGEXP_REPLACE(b.dc_upc, '^0+', '')
              OR REGEXP_REPLACE(a.upc, '^0+', '') = REGEXP_REPLACE(c.pdp_upc, '^0+', '')
            THEN TRUE ELSE FALSE
        END AS has_upc_overlap,
        
        -- CSF UPC
        CASE WHEN cu.score IS NOT NULL THEN TRUE ELSE FALSE END AS is_suggested_by_csf_upc,
        cu.score AS csf_upc_score,
        cu.upc_sugg_date AS csf_upc_first_sugg_date,
        cu.base_upc AS csf_base_upc,
        cu.comp_upc AS csf_comp_upc,
        
        -- CSF MPN
        CASE WHEN cm.score IS NOT NULL THEN TRUE ELSE FALSE END AS is_suggested_by_csf_mpn,
        cm.score AS csf_mpn_score,
        cm.mpn_sugg_date AS csf_mpn_first_sugg_date,
        -- cm.base_mpn AS csf_base_mpn,
        -- cm.comp_mpn AS csf_comp_mpn,
        
        -- CSF Precision Model
        CASE WHEN cp.model_score IS NOT NULL THEN TRUE ELSE FALSE END AS is_suggested_by_csf_precision,
        cp.model_score AS csf_precision_score,
        cp.first_sugg_date AS csf_precision_first_sugg_date,
        
        -- CSF Generic Model
        CASE WHEN cg.model_score IS NOT NULL THEN TRUE ELSE FALSE END AS is_suggested_by_csf_generic,
        cg.model_score AS csf_generic_score,
        cg.first_sugg_date AS csf_generic_first_sugg_date,
        
        -- CSF Semantic Model
        CASE WHEN cg.model_score IS NOT NULL THEN TRUE ELSE FALSE END AS is_suggested_by_semantic,
        cg.model_score AS csf_semantic_score,
        cg.first_sugg_date AS csf_semantic_first_sugg_date,
        
        -- CSF Queue
        CASE WHEN cq.queue_score IS NOT NULL THEN TRUE ELSE FALSE END AS is_pair_in_csf_queue,
        cq.queue_score AS csf_queue_score,
        cq.queue_date AS csf_queue_date,
        cq.queue_name AS csf_queue_name,
        
        -- CSF ML Input: Base SKU
        CASE WHEN cmi_base.inp_sku IS NOT NULL THEN TRUE ELSE FALSE END AS base_sku_is_present_in_csf_ml_input,
        cmi_base.inp_title AS base_title_csf_ml_input,
        cmi_base.inp_capture_date AS base_capture_date_csf,
        
        -- CSF ML Input: Comp SKU
        CASE WHEN cmi_comp.inp_sku IS NOT NULL THEN TRUE ELSE FALSE END AS comp_sku_is_present_in_csf_ml_input,
        cmi_comp.inp_title AS comp_title_csf_ml_input,
        cmi_comp.inp_capture_date AS comp_capture_date_csf,
        
        -- AFM Suggestions
        CASE WHEN afm.afm_request_key IS NOT NULL THEN TRUE ELSE FALSE END AS has_afm_request,
        afm.search_type AS afm_search_type,
        afm.search_key_type AS afm_search_key_type
    FROM crawl_attempt AS a
    LEFT JOIN deep_crawl AS b
        ON a.comp_source_store = b.dc_source_store
        AND a.comp_sku = b.dc_sku
    LEFT JOIN pdp AS c
        ON a.comp_source_store = c.pdp_source_store
        AND a.comp_sku = c.pdp_sku
    LEFT JOIN fl_dump AS f
        ON a.comp_source_store = f.comp_source_store
        AND a.base_sku = f.base_sku
        AND a.comp_sku = f.comp_sku
    LEFT JOIN match_lib AS h
        ON a.comp_source_store = h.comp_source_store
        AND a.base_sku = h.matchlib_sku
        AND a.comp_sku = h.matchlib_comp
    LEFT JOIN csf_generic as cg 
        ON a.comp_source_store = cg.comp_source_store
        AND a.base_sku = cg.base_sku
        AND a.comp_sku = cg.comp_sku
    LEFT JOIN csf_semantic as cs
        ON a.comp_source_store = cs.comp_source_store
        AND a.base_sku = cs.base_sku
        AND a.comp_sku = cs.comp_sku
    LEFT JOIN csf_precision AS cp
        ON a.comp_source_store = cp.comp_source_store
        AND a.base_sku = cp.base_sku
        AND a.comp_sku = cp.comp_sku    
    LEFT JOIN csf_queue AS cq
        ON a.comp_source_store = cq.comp_source_store
        AND a.base_sku = cq.base_sku
        AND a.comp_sku = cq.comp_sku
    LEFT JOIN csf_ml_input as cmi_base
       ON a.base_sku = cmi_base.inp_sku
       AND a.base_source_store = cmi_base.inp_source_store
    LEFT JOIN csf_ml_input AS cmi_comp
        ON a.comp_sku = cmi_comp.inp_sku
        AND a.comp_source_store = cmi_comp.inp_source_store
    LEFT JOIN afm as afm
        ON a.base_sku = afm.base_sku
       AND a.comp_source_store = afm.comp_source_store
    LEFT JOIN csf_upc as cu
        ON a.base_sku = cu.base_sku
        AND a.comp_source_store = cu.comp_source_store
        AND a.comp_sku = cu.comp_sku
    LEFT JOIN csf_mpn as cm
        ON a.base_sku = cm.base_sku
        AND a.comp_source_store = cm.comp_source_store
        AND a.comp_sku = cm.comp_sku
)
SELECT *
FROM (
    SELECT *,
           ROW_NUMBER() OVER (
               PARTITION BY base_sku, comp_sku, comp_source_store
               ORDER BY base_sku DESC
           ) AS rn
    FROM result
) t
WHERE rn = 1