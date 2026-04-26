#!/usr/bin/env python3
"""
Spatial QA question generator — CLI version.

Usage:
  python generator_cli.py                        # generate all 28 template types
  python generator_cli.py --types range+name knn+name intersects+count
  python generator_cli.py --list                 # list all available template types
  python generator_cli.py --n 500                # override default N per type
  python generator_cli.py --output-dir ./my_questions
"""

import argparse
import json
import os
import random
import time
from io import StringIO
from pathlib import Path

import language_tool_python
import psycopg2
import requests
import shapely
from dateutil.parser import parse
from textblob.blob import Word
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# DB connection
# ─────────────────────────────────────────────────────────────────────────────

DB_PARAMS = dict(
    host="localhost",
    dbname="osm_ca",
    user="postgres",
    password="postgres",
    port=5432,
)


def run_sql_select(sql, return_dict=False):
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    cur.execute(sql)
    colnames = [desc[0] for desc in cur.description]
    records = cur.fetchall()
    cur.close()
    conn.close()
    if return_dict:
        return [
            {colnames[j]: records[i][j] for j in range(len(colnames)) if records[i][j] is not None}
            for i in range(len(records))
        ]
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Configuration: selectors, filters, templates
# ─────────────────────────────────────────────────────────────────────────────

HERE = Path(__file__).parent

pois_selector = {
    "tourism": ["aquarium", "attraction", "viewpoint", "art gallery", "theme park",
                "museum", "bed and breakfast", "gallery", "zoo", "hotel"],
    "amenity": ["restaurant", "hospital", "university", "food", "fast food restaurant",
                "coffee shop", "cafe"],
    "leisure": ["park", "beach resort", "golf course", "nature reserve", "garden", "stadium"],
}

poi_filters = json.loads((HERE / "filter_labels.json").read_text())

park_lake_selector = {
    "parks": ["recreation ground", "common", "nature reserve", "park", "garden",
              "sports centre", "golf course", "marina"],
    "water": ["bay", "harbour", "lake", "reservoir"],
}

road_waterway_selector = {
    "roads":    ["road", "footway", "cycleway", "secondary", "pedestrian", "primary",
                 "residential", "track", "motorway"],
    "waterway": ["canal", "river", "stream"],
}

direction_filters = [
    ["north",     "((degrees(ST_Azimuth(origin.geometry, pois.geometry)) BETWEEN 0.0 AND 22.5) OR (degrees(ST_Azimuth(origin.geometry, pois.geometry)) BETWEEN 337.5 AND 360))"],
    ["northeast", "degrees(ST_Azimuth(origin.geometry, pois.geometry)) BETWEEN 22.5 AND 67.5"],
    ["east",      "degrees(ST_Azimuth(origin.geometry, pois.geometry)) BETWEEN 67.5 AND 112.5"],
    ["southeast", "degrees(ST_Azimuth(origin.geometry, pois.geometry)) BETWEEN 112.5 AND 157.5"],
    ["south",     "degrees(ST_Azimuth(origin.geometry, pois.geometry)) BETWEEN 157.5 AND 202.5"],
    ["southwest", "degrees(ST_Azimuth(origin.geometry, pois.geometry)) BETWEEN 202.5 AND 247.5"],
    ["west",      "degrees(ST_Azimuth(origin.geometry, pois.geometry)) BETWEEN 247.5 AND 292.5"],
    ["northwest", "degrees(ST_Azimuth(origin.geometry, pois.geometry)) BETWEEN 292.5 AND 337.5"],
]

regions_selector = ["city", "town", "village", "island", "municipality", "county",
                    "neighbourhood", "suburb", "state"]

SQL_NO_REF   = "SELECT * FROM TABLE2  TABLESAMPLE SYSTEM(5) WHERE PREDICATE MUST_HAVE IS NOT NULL LIMIT 1"
SQL_WITH_REF = ("SELECT * FROM TABLE2 TABLESAMPLE SYSTEM(5) WHERE PREDICATE MUST_HAVE IS NOT NULL "
                "AND ST_DWithin(TABLE2.geometry, ST_GeomFromText('WKT',4326)::geography, 1E5) LIMIT 1")

language_tool = language_tool_python.LanguageTool("en-US")
is_bad_rule = lambda rule: "spelling" in rule.message.lower() and len(rule.replacements) and rule.replacements[0][0].isupper()


def poi_category_to_sql_name(value):
    if " " in value:
        if value == "coffee shop":
            value = "coffee"
        elif value == "fast food restaurant":
            value = "fast_food"
        else:
            value = value.replace(" ", "_")
    return value


# ─────────────────────────────────────────────────────────────────────────────
# Geometry verification
# ─────────────────────────────────────────────────────────────────────────────

def verify_geo_wkts(template_tokens, selected_entities, sql):
    for tt in template_tokens:
        token_key = tt[0]
        if not token_key.endswith("_wkt]"):
            continue
        entity_key = token_key.replace("_wkt]", "]")
        if entity_key not in selected_entities:
            continue
        entity = selected_entities[entity_key]
        if "geo_wkt" not in entity:
            continue
        expected_wkt = entity["geo_wkt"]
        assert expected_wkt in sql, \
            f"geo_wkt for {entity_key} not found in SQL"
        for subkey, table in [("poi", "pois"), ("region", "regions")]:
            if subkey in entity:
                osm_id = entity[subkey].get("osm_id")
                if osm_id:
                    db_result = run_sql_select(f"SELECT geometry FROM {table} WHERE osm_id = {osm_id} LIMIT 1")
                    assert db_result, f"No DB record in {table} for osm_id={osm_id}"
                    db_wkt = shapely.to_wkt(shapely.from_wkb(db_result[0][0]))
                    assert db_wkt == expected_wkt, \
                        f"geo_wkt mismatch for {entity_key} (osm_id={osm_id}): stored='{expected_wkt}' vs db='{db_wkt}'"
                break


# ─────────────────────────────────────────────────────────────────────────────
# Core generator
# ─────────────────────────────────────────────────────────────────────────────

def question_generator(text_templates, variable_types, template_tokens, sql_template,
                       answer_type, verifier, n=100, disable_progress=False):
    generated_questions = []
    progress = tqdm(total=n) if not disable_progress else None

    while len(generated_questions) < n:
        selected_entities = {}
        ref_geo_wkt = None
        skip_iteration = False
        geo_wkt = None
        poi_mbr = [1e100, 1e100, -1e100, -1e100]

        for k, v in variable_types:
            if "poi_filter" in v:
                sub_category = random.choice(list(poi_filters.keys()))
                main_category = None
                for kk in pois_selector:
                    if sub_category in pois_selector[kk]:
                        main_category = kk
                        break
                selected_filter = random.choice(poi_filters[sub_category])
                selected_entities[k] = {
                    "main_category": main_category, "sub_category": sub_category,
                    "poi_filter_desc": selected_filter[0], "poi_filter_sql": selected_filter[1],
                }
            elif "poi" in v:
                main_category = random.choice(list(pois_selector.keys()))
                sub_category  = random.choice(pois_selector[main_category])
                if "distance_limited" in v and ref_geo_wkt is not None:
                    predicate = "%s ILIKE '%s' AND " % (main_category, poi_category_to_sql_name(sub_category))
                    if sub_category in ("restaurant", "food"):
                        predicate = " (amenity ILIKE 'fast_food' OR amenity ILIKE 'restaurant' OR amenity ILIKE 'food') AND "
                    elif sub_category in ("cafe", "coffee_shop"):
                        predicate = " (amenity ILIKE 'cafe' OR amenity ILIKE 'coffee_shop') AND "
                    _sql = SQL_WITH_REF.replace("TABLE2", "pois").replace("MUST_HAVE", "addr_state").replace("WKT", ref_geo_wkt).replace("PREDICATE", predicate)
                    output = run_sql_select(_sql, return_dict=True)
                    if not output:
                        skip_iteration = True; break
                    random_poi = output[0]
                elif "mbr_limited" in v and poi_mbr is not None:
                    mbr_wkt = shapely.box(*poi_mbr).wkt
                    output = run_sql_select(
                        SQL_NO_REF.replace("TABLE2", "pois").replace("MUST_HAVE", "addr_state")
                        .replace("PREDICATE", "ST_Intersects(geometry, ST_GeomFromText('%s',4326)::geography) AND " % mbr_wkt),
                        return_dict=True,
                    )
                    if not output:
                        skip_iteration = True; break
                    random_poi = output[0]
                    main_category = sub_category = None
                    for kk in pois_selector:
                        if kk in random_poi:
                            main_category = kk
                            sub_category  = random_poi[kk]
                            break
                elif "multi_source" in v:
                    select_sql = f"""
                    SELECT p.* FROM pois p TABLESAMPLE SYSTEM (5)
                    JOIN LATERAL (
                        SELECT n.* FROM pois n
                        WHERE n.id <> p.id AND n.addr_state IS NOT NULL
                        AND n.{selected_entities['[1]']['main_category']} ILIKE '{selected_entities['[1]']['sub_category']}'
                        AND n.wikidata IS NOT NULL
                        ORDER BY ST_Distance(p.geometry, n.geometry) LIMIT 1
                    ) nn ON TRUE
                    WHERE p.{main_category} ILIKE '{sub_category}'
                    AND p.addr_state IS NOT NULL LIMIT 1;
                    """
                    output = run_sql_select(select_sql, return_dict=True)
                    if not output:
                        skip_iteration = True; break
                    random_poi = output[0]
                    ref_geo_wkt = shapely.to_wkt(shapely.from_wkb(random_poi["geometry"]))
                else:
                    select_sql = SQL_NO_REF.replace("TABLE2", "pois").replace("MUST_HAVE", "addr_state").replace("PREDICATE", "%s ILIKE '%s' AND " % (main_category, poi_category_to_sql_name(sub_category)))
                    output = run_sql_select(select_sql, return_dict=True)
                    if not output:
                        skip_iteration = True; break
                    random_poi = output[0]
                    ref_geo_wkt = shapely.to_wkt(shapely.from_wkb(random_poi["geometry"]))

                display_name = random_poi["poi_name"]
                if "addr_city"  in random_poi: display_name += ", " + random_poi["addr_city"]
                if "addr_state" in random_poi: display_name += ", " + random_poi["addr_state"]
                geo_wkt = shapely.to_wkt(shapely.from_wkb(random_poi["geometry"]))
                p = shapely.from_wkb(random_poi["geometry"])
                poi_mbr = [min(poi_mbr[0], p.x), min(poi_mbr[1], p.y), max(poi_mbr[2], p.x), max(poi_mbr[3], p.y)]
                random_poi["geometry"] = geo_wkt
                selected_entities[k] = {
                    "main_category": main_category, "sub_category": sub_category,
                    "display_name": display_name, "geo_wkt": geo_wkt, "poi": random_poi,
                }

            elif "region" in v:
                predicate = "ST_Intersects(geometry, ST_GeomFromText('%s',4326)::geography)" % geo_wkt
                ss = "SELECT * FROM regions WHERE %s AND wikipedia IS NOT NULL LIMIT 1" % predicate
                output = run_sql_select(ss, return_dict=True)
                if not output:
                    skip_iteration = True; break
                random_region = output[0]
                geo_wkt = shapely.to_wkt(shapely.from_wkb(random_region["geometry"]))
                random_region["geometry"] = geo_wkt
                selected_entities[k] = {
                    "region_name": random_region["wikipedia"][3:],
                    "geo_wkt": geo_wkt, "region": random_region,
                }

            elif "park_lake" in v:
                __tmp = random.choice(list(park_lake_selector.keys()))
                table = "parks" if __tmp == "parks" else "lakes"
                main_category = "leisure" if table == "parks" else __tmp
                sub_category  = random.choice(list(park_lake_selector[__tmp]))
                predicate = "leisure IS NOT NULL AND " if main_category == "leisure" else \
                    " (waterway ILIKE '%s' OR water ILIKE '%s') AND " % (sub_category, sub_category)
                predicate += " ST_Area(geometry) > 0 AND "
                output = run_sql_select(
                    SQL_NO_REF.replace("TABLE2", table).replace("MUST_HAVE", "wikipedia").replace("PREDICATE", predicate),
                    return_dict=True,
                )
                if not output:
                    skip_iteration = True; break
                random_obj = output[0]
                geo_wkt = shapely.to_wkt(shapely.from_wkb(random_obj["geometry"]))
                random_obj["geometry"] = geo_wkt
                selected_entities[k] = {
                    "park_name": random_obj["wikipedia"][3:], "main_category": main_category,
                    "sub_category": sub_category, "table": table,
                    Word(table).singularize(): random_obj,
                }

            elif "road_waterway" in v:
                __tmp = random.choice(list(road_waterway_selector.keys()))
                table = "roads" if __tmp == "roads" else "lakes"
                main_category = "highway" if table == "roads" else __tmp
                sub_category  = random.choice(list(road_waterway_selector[__tmp]))
                if main_category == "highway" and not (sub_category[-3:] == "way" or sub_category == "road"):
                    sub_category_label = sub_category + " road"
                else:
                    sub_category_label = sub_category
                predicate = (
                    "(road_name IS NOT NULL OR wikipedia IS NOT NULL) AND highway = '%s' " % sub_category
                    if main_category == "highway"
                    else "(lake_name IS NOT NULL OR wikipedia IS NOT NULL) AND (waterway = '%s' OR water = '%s') " % (sub_category, sub_category)
                )
                ss = "SELECT * FROM %s TABLESAMPLE SYSTEM(5) WHERE %s LIMIT 1;" % (table, predicate)
                output = run_sql_select(ss, return_dict=True)
                if not output:
                    skip_iteration = True; break
                random_obj = output[0]
                geo_wkt = shapely.to_wkt(shapely.from_wkb(random_obj["geometry"]))
                random_obj["geometry"] = geo_wkt
                selected_entities[k] = {
                    "main_category": main_category, "sub_category": sub_category,
                    "sub_category_label": sub_category_label, "table": table,
                    Word(table).singularize(): random_obj,
                }

            elif "distance" in v:
                distance = random.randint(1, 200)
                distance = (distance - (distance % 10)) * 1000
                selected_entities[k] = {"distance": distance, "text": "%d kilometers" % int(distance / 1000.0)}

            elif "direction" in v:
                dir_desc, dir_sql = random.choice(direction_filters)
                selected_entities[k] = {"direction_desc": dir_desc, "direction_predicate": dir_sql}

        if skip_iteration:
            continue

        t   = random.choice(text_templates)
        sql = str(sql_template)

        if "[1_type] = '[1]'" in sql_template and "sub_category" in selected_entities:
            sc = selected_entities["sub_category"]
            if sc in ("restaurant", "food"):
                sql = sql.replace("[1_type] = '[1]'", "(amenity ILIKE 'fast_food' OR amenity ILIKE 'restaurant' OR amenity ILIKE 'food')")
            elif sc in ("cafe", "coffee_shop"):
                sql = sql.replace("[1_type] = '[1]'", "(amenity ILIKE 'cafe' OR amenity ILIKE 'coffee_shop')")

        for tt in template_tokens:
            k2, v2 = tt[0], tt[1]
            for _k in selected_entities:
                if _k in v2:
                    _value = selected_entities[_k][v2[v2.find(" ") + 1:]]
                    if k2 in sql:
                        value = poi_category_to_sql_name(str(_value)) if "sub_category" in v2 else str(_value)
                        if "name" in k2:
                            value = value.replace("'", "''")
                        sql = sql.replace(k2, value)
                    if k2 in t:
                        value = _value.replace("_", " ")
                        if "append_an" in tt:
                            an = ("an " if value[0] in "aeoiu" else "a ")
                            if "any [1]" in t:
                                an = ""
                            value = an + value
                        elif "pluralize" in tt:
                            value = Word(value).pluralize()
                        t = t.replace(k2, value)

        # For road/waterway types, add name-quality predicate
        _sql = sql
        if main_category == "highway":
            predicate = "(road_name IS NOT NULL OR wikipedia IS NOT NULL) AND highway = '%s' " % sub_category
            _sql = sql.replace("WHERE", "WHERE %s AND " % predicate)
        elif main_category in ("waterway", "water"):
            predicate = "(lake_name IS NOT NULL OR wikipedia IS NOT NULL) AND (waterway = '%s' OR water = '%s') " % (sub_category, sub_category)
            _sql = sql.replace("WHERE", "WHERE %s AND " % predicate)

        answers = run_sql_select(_sql, return_dict=True)
        verify_geo_wkts(template_tokens, selected_entities, sql)

        if verifier(template_tokens, selected_entities, answers):
            matches = language_tool.check(t)
            matches = [r for r in matches if not is_bad_rule(r)]
            t = language_tool_python.utils.correct(t, matches)
            for i in range(len(answers)):
                if "geometry" in answers[i]:
                    answers[i]["geometry"] = shapely.to_wkt(shapely.from_wkb(answers[i]["geometry"]))
            generated_questions.append({
                "question": t, "sql": sql, "answers": answers,
                "answer_type": answer_type, "question_entities": selected_entities,
            })
            if progress:
                progress.update(1)

    if progress:
        progress.close()
    return generated_questions


def save(questions, filename, output_dir: Path):
    path = output_dir / filename
    with open(path, "w") as f:
        for q in questions:
            f.write(json.dumps(q) + "\n")
    print(f"  Saved {len(questions)} questions → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Template registry
# Each entry: (type_name, setup_fn)
# setup_fn() returns (text_templates, variable_types, template_tokens,
#                     template_sql, answer_type, verifier_fn)
# ─────────────────────────────────────────────────────────────────────────────

def _load_templates(name):
    path = HERE / "templates" / f"{name}.txt"
    return [l.strip() for l in path.read_text().splitlines() if l.strip()]


def _make_range_name_verifier():
    def verifier(tt, se, answers):
        if poi_category_to_sql_name(se["[1]"]["sub_category"]) not in [
            "viewpoint", "attraction", "museum", "theme park", "hotel", "gallery", "zoo",
            "aquarium", "art_gallery", "cafe", "restaurant", "fast_food", "university",
            "hospital", "coffee", "park", "golf_course", "nature_reserve", "stadium", "garden", "beach_resort"
        ]: return False
        if poi_category_to_sql_name(se["[3]"]["sub_category"]) not in [
            "aquarium", "attraction", "viewpoint", "art_gallery", "theme_park", "museum",
            "gallery", "zoo", "hotel", "university", "park", "nature_reserve", "garden", "stadium", "hospital"
        ]: return False
        if poi_category_to_sql_name(se["[3]"]["sub_category"]) == poi_category_to_sql_name(se["[1]"]["sub_category"]):
            return False
        return bool(answers)
    return verifier


def _make_knn_verifier():
    def verifier(tt, se, answers):
        if poi_category_to_sql_name(se["[1]"]["sub_category"]) not in [
            "viewpoint", "attraction", "museum", "theme_park", "hotel", "gallery", "zoo",
            "aquarium", "art_gallery", "cafe", "restaurant", "fast_food", "university",
            "hospital", "coffee", "park", "golf_course", "nature_reserve", "stadium", "garden", "beach_resort"
        ]: return False
        if poi_category_to_sql_name(se["[2]"]["sub_category"]) not in [
            "aquarium", "attraction", "viewpoint", "art_gallery", "theme_park", "museum",
            "gallery", "zoo", "hotel", "university", "park", "nature_reserve", "garden", "stadium", "hospital"
        ]: return False
        if poi_category_to_sql_name(se["[2]"]["sub_category"]) == poi_category_to_sql_name(se["[1]"]["sub_category"]):
            return False
        return bool(answers)
    return verifier


def _make_towards_verifier(keys=("[1]", "[3]", "[4]")):
    def verifier(tt, se, answers):
        vals = [poi_category_to_sql_name(se[k]["sub_category"]) for k in keys]
        allowed1 = ["viewpoint", "attraction", "museum", "theme park", "hotel", "gallery", "zoo",
                    "aquarium", "art gallery", "cafe", "restaurant", "fast_food", "university",
                    "hospital", "coffee", "park", "golf_course", "nature_reserve", "stadium", "garden", "beach_resort"]
        allowed_rest = ["aquarium", "attraction", "viewpoint", "art_gallery", "theme_park", "museum",
                        "gallery", "zoo", "hotel", "university", "park", "nature_reserve", "garden", "stadium", "hospital"]
        if vals[0] not in allowed1: return False
        for v in vals[1:]:
            if v not in allowed_rest: return False
        if len(set(vals)) < len(vals): return False
        return bool(answers)
    return verifier


def _make_simple_nonempty_verifier():
    def verifier(tt, se, answers): return bool(answers)
    return verifier


def _make_intersects_count_verifier():
    def verifier(tt, se, answers):
        if se["[1]"]["sub_category"] not in [
            "viewpoint", "attraction", "museum", "theme park", "hotel", "gallery", "zoo",
            "aquarium", "art gallery", "cafe", "restaurant", "fast food restaurant", "university",
            "hospital", "coffee shop", "park", "golf course", "nature reserve", "stadium", "garden", "beach resort"
        ]: return False
        if not answers: return False
        return answers[0].get("count", 0) != 0
    return verifier


def _make_park_lake_verifier(require_area=False, require_length=False):
    def verifier(tt, se, answers):
        if se["[1]"]["sub_category"] not in ["bay", "harbour", "lake", "reservoir"] \
                and se["[1]"]["main_category"] != "leisure":
            return False
        if not answers: return False
        if require_area and (not answers[0].get("area")): return False
        return True
    return verifier


def _make_road_waterway_verifier(require_length=False):
    def verifier(tt, se, answers):
        if not answers: return False
        if not ("wikipedia" in answers[0] or "lake_name" in answers[0] or "road_name" in answers[0]):
            return False
        if require_length and not answers[0].get("length"): return False
        return True
    return verifier


RANGE_VARIABLE_TYPES_3POI = [("[1]", "poi"), ("[2]", "distance"), ("[3]", "poi")]
RANGE_TEMPLATE_TOKENS_NAME = [
    ("[1_type]", "[1] main_category"), ("[1]", "[1] sub_category", "append_an"),
    ("[2]", "[2] distance"), ("[2_text]", "[2] text"),
    ("[3]", "[3] display_name"), ("[3_wkt]", "[3] geo_wkt"),
]

KNN_VARIABLE_TYPES = [("[1]", "poi"), ("[2]", "poi")]
KNN_TEMPLATE_TOKENS = [
    ("[1_type]", "[1] main_category"), ("[1]", "[1] sub_category"),
    ("[2]", "[2] display_name"), ("[2_wkt]", "[2] geo_wkt"),
]

REGION_VARIABLE_TYPES = [("[1]", "road_waterway"), ("[2]", "region")]
REGION_TEMPLATE_TOKENS = [
    ("[1_type]", "[1] main_category"), ("[1_table]", "[1] table"),
    ("[1]", "[1] sub_category_label"), ("[2]", "[2] region_name"), ("[2_wkt]", "[2] geo_wkt"),
]


def build_registry():
    """Returns an ordered list of (type_name, text_templates, variable_types, template_tokens,
    template_sql, answer_type, verifier) tuples."""
    reg = []

    # ── Range ──────────────────────────────────────────────────────────────
    range_sql_name = "SELECT * FROM pois\nWHERE ST_DWithin(pois.geometry, ST_GeomFromText('[3_wkt]',4326)::geography, [2])\nAND [1_type] = '[1]';\n"
    range_v = _make_range_name_verifier()

    reg.append(("range+name",
        [l.replace("[2]", "[2_text]") for l in _load_templates("range+name")],
        RANGE_VARIABLE_TYPES_3POI, RANGE_TEMPLATE_TOKENS_NAME, range_sql_name, "name", range_v))

    reg.append(("range+loc",
        [l.replace("[2]", "[2_text]") for l in _load_templates("range+loc")],
        RANGE_VARIABLE_TYPES_3POI, RANGE_TEMPLATE_TOKENS_NAME, range_sql_name, "loc", range_v))

    count_tokens = [
        ("[1_type]", "[1] main_category"), ("[1]", "[1] sub_category", "pluralize"),
        ("[2]", "[2] distance"), ("[2_text]", "[2] text"),
        ("[3]", "[3] display_name"), ("[3_wkt]", "[3] geo_wkt"),
    ]
    range_count_sql = "SELECT COUNT(*) FROM pois\nWHERE ST_DWithin(pois.geometry, ST_GeomFromText('[3_wkt]',4326)::geography, [2])\nAND [1_type] = '[1]';\n"
    reg.append(("range+count",
        [l.replace("[2]", "[2_text]") for l in _load_templates("range+count")],
        RANGE_VARIABLE_TYPES_3POI, count_tokens, range_count_sql, "count", range_v))

    angle_tokens = [
        ("[1_type]", "[1] main_category"), ("[1]", "[1] sub_category", "append_an"),
        ("[2]", "[2] distance"), ("[2_text]", "[2] text"),
        ("[3]", "[3] display_name"), ("[3_wkt]", "[3] geo_wkt"),
    ]
    range_angle_sql = "SELECT *, degrees(ST_Azimuth(ST_GeomFromText('[3_wkt]',4326)::geography, pois.geometry)) AS angle FROM pois\nWHERE ST_DWithin(pois.geometry, ST_GeomFromText('[3_wkt]',4326)::geography, [2])\nAND [1_type] = '[1]';\n"
    reg.append(("range+angle",
        [l.replace("[2]", "[2_text]") for l in _load_templates("range+angle")],
        RANGE_VARIABLE_TYPES_3POI, angle_tokens, range_angle_sql, "angle", range_v))

    range_dist_sql = "SELECT *, ST_Distance(ST_GeomFromText('[3_wkt]',4326)::geography, pois.geometry) AS distance FROM pois\nWHERE ST_DWithin(pois.geometry, ST_GeomFromText('[3_wkt]',4326)::geography, [2])\nAND [1_type] = '[1]';\n"
    reg.append(("range+distance",
        [l.replace("[2]", "[2_text]") for l in _load_templates("range+distance")],
        RANGE_VARIABLE_TYPES_3POI, angle_tokens, range_dist_sql, "distance", range_v))

    # ── Range + non-spatial filter ──────────────────────────────────────────
    nsf_vtypes = [("[1]", "poi_filter"), ("[2]", "distance"), ("[3]", "poi")]
    nsf_tokens = [
        ("[1_type]", "[1] main_category"), ("[1]", "[1] sub_category"),
        ("[1_filter]", "[1] poi_filter_desc", "append_an"), ("[1_predicate]", "[1] poi_filter_sql"),
        ("[2]", "[2] distance"), ("[2_text]", "[2] text"),
        ("[3]", "[3] display_name"), ("[3_wkt]", "[3] geo_wkt"),
    ]
    nsf_sql = "SELECT * FROM pois\nWHERE ST_DWithin(pois.geometry, ST_GeomFromText('[3_wkt]',4326)::geography, [2])\nAND [1_type] = '[1]'\nAND [1_predicate];\n"
    nsf_v = _make_simple_nonempty_verifier()
    reg.append(("range:non_spat_filter+name",
        [l.replace("[1]", "[1_filter]").replace("[2]", "[2_text]") for l in _load_templates("range:non_spat_filter+name")],
        nsf_vtypes, nsf_tokens, nsf_sql, "name", nsf_v))
    reg.append(("range:non_spat_filter+loc",
        [l.replace("[1]", "[1_filter]").replace("[2]", "[2_text]") for l in _load_templates("range:non_spat_filter+loc")],
        nsf_vtypes, nsf_tokens, nsf_sql, "loc", nsf_v))

    # ── Range + direction ───────────────────────────────────────────────────
    dir_vtypes = [("[1]", "poi"), ("[2]", "distance"), ("[3]", "poi"), ("[4]", "direction")]
    dir_tokens = [
        ("[1_type]", "[1] main_category"), ("[1]", "[1] sub_category"),
        ("[2]", "[2] distance"), ("[2_text]", "[2] text"),
        ("[3]", "[3] display_name"), ("[3_wkt]", "[3] geo_wkt"),
        ("[4]", "[4] direction_desc"), ("[4_predicate]", "[4] direction_predicate"),
    ]
    dir_sql = "\nWITH origin AS (SELECT ST_GeomFromText('[3_wkt]',4326)::geography AS geometry)\nSELECT * FROM pois, origin\nWHERE ST_DWithin(pois.geometry, ST_GeomFromText('[3_wkt]',4326)::geography, [2])\nAND [1_type] = '[1]'\nAND [4_predicate];\n"
    reg.append(("range:direction+name",
        [l.replace("[2]", "[2_text]") for l in _load_templates("range:direction+name")],
        dir_vtypes, dir_tokens, dir_sql, "name", _make_simple_nonempty_verifier()))
    reg.append(("range:direction+loc",
        [l.replace("[2]", "[2_text]") for l in _load_templates("range:direction+loc")],
        dir_vtypes,
        [("[1_type]", "[1] main_category"), ("[1]", "[1] sub_category", "append_an"),
         ("[2]", "[2] distance"), ("[2_text]", "[2] text"),
         ("[3]", "[3] display_name"), ("[3_wkt]", "[3] geo_wkt"),
         ("[4]", "[4] direction_desc"), ("[4_predicate]", "[4] direction_predicate")],
        dir_sql, "loc", _make_simple_nonempty_verifier()))

    # ── Range + towards ─────────────────────────────────────────────────────
    towards_vtypes = [("[2]", "distance"), ("[3]", "poi"), ("[4]", "poi distance_limited"), ("[1]", "poi mbr_limited")]
    towards_tokens = [
        ("[1_type]", "[1] main_category"), ("[1]", "[1] sub_category"),
        ("[2]", "[2] distance"), ("[2_text]", "[2] text"),
        ("[3]", "[3] display_name"), ("[3_wkt]", "[3] geo_wkt"),
        ("[4]", "[4] display_name"), ("[4_wkt]", "[4] geo_wkt"),
    ]
    towards_sql = "\nWITH angle AS (SELECT degrees(ST_Azimuth(ST_GeomFromText('[3_wkt]',4326)::geography, ST_GeomFromText('[4_wkt]',4326)::geography)) AS value)\nSELECT * FROM pois, angle\nWHERE [1_type] = '[1]'\nAND ST_DWithin(pois.geometry, ST_GeomFromText('[3_wkt]',4326)::geography, [2])\nAND (degrees(ST_Azimuth(ST_GeomFromText('[3_wkt]',4326)::geography, pois.geometry)) BETWEEN angle.value - 22.5 AND angle.value + 22.5);\n"
    towards_v = _make_towards_verifier(("[1]", "[3]", "[4]"))
    reg.append(("range:towards+name",
        [l.replace("[2]", "[2_text]") for l in _load_templates("range:towards+name")],
        towards_vtypes, towards_tokens, towards_sql, "name", towards_v))
    reg.append(("range:towards+loc",
        [l.replace("[2]", "[2_text]") for l in _load_templates("range:towards+loc")],
        towards_vtypes, towards_tokens, towards_sql, "loc", towards_v))

    # ── KNN ─────────────────────────────────────────────────────────────────
    knn_sql = "SELECT * FROM pois\nWHERE [1_type] = '[1]'\nORDER BY geometry <-> ST_GeomFromText('[2_wkt]',4326)::geography ASC\nLIMIT 1;\n"
    knn_v = _make_knn_verifier()
    reg.append(("knn+name",  _load_templates("knn+name"), KNN_VARIABLE_TYPES, KNN_TEMPLATE_TOKENS, knn_sql, "name", knn_v))
    reg.append(("knn+loc",   _load_templates("knn+loc"),  KNN_VARIABLE_TYPES, KNN_TEMPLATE_TOKENS, knn_sql, "loc",  knn_v))

    knn_dist_sql = "SELECT *, ST_Distance(geometry, ST_GeomFromText('[2_wkt]',4326)::geography) AS distance FROM pois\nWHERE [1_type] = '[1]'\nORDER BY geometry <-> ST_GeomFromText('[2_wkt]',4326)::geography ASC\nLIMIT 1;\n"
    reg.append(("knn+distance", _load_templates("knn+distance"), KNN_VARIABLE_TYPES, KNN_TEMPLATE_TOKENS, knn_dist_sql, "distance", knn_v))

    knn_angle_sql = "SELECT *, degrees(ST_Azimuth(ST_GeomFromText('[2_wkt]',4326)::geography, pois.geometry)) AS angle FROM pois\nWHERE [1_type] = '[1]'\nORDER BY geometry <-> ST_GeomFromText('[2_wkt]',4326)::geography ASC\nLIMIT 1;\n"
    reg.append(("knn+angle", _load_templates("knn+angle"), KNN_VARIABLE_TYPES, KNN_TEMPLATE_TOKENS, knn_angle_sql, "angle", knn_v))

    # ── KNN + non-spatial filter ────────────────────────────────────────────
    knn_nsf_vtypes = [("[1]", "poi_filter"), ("[2]", "poi")]
    knn_nsf_tokens = [
        ("[1_type]", "[1] main_category"), ("[1]", "[1] sub_category"),
        ("[1_filter]", "[1] poi_filter_desc"), ("[1_predicate]", "[1] poi_filter_sql"),
        ("[2]", "[2] display_name"), ("[2_wkt]", "[2] geo_wkt"),
    ]
    knn_nsf_sql = "SELECT * FROM pois\nWHERE [1_type] = '[1]'\nAND [1_predicate]\nORDER BY geometry <-> ST_GeomFromText('[2_wkt]',4326)::geography ASC\nLIMIT 1;\n"
    knn_nsf_v = _make_simple_nonempty_verifier()
    reg.append(("knn:non_spat_filter+name",
        [l.replace("[1]", "[1_filter]") for l in _load_templates("knn:non_spat_filter+name")],
        knn_nsf_vtypes, knn_nsf_tokens, knn_nsf_sql, "name", knn_nsf_v))
    reg.append(("knn:non_spat_filter+loc",
        [l.replace("[1]", "[1_filter]") for l in _load_templates("knn:non_spat_filter+loc")],
        knn_nsf_vtypes, knn_nsf_tokens, knn_nsf_sql, "loc", knn_nsf_v))

    # ── KNN + direction ─────────────────────────────────────────────────────
    knn_dir_vtypes = [("[1]", "poi"), ("[2]", "poi"), ("[3]", "direction")]
    knn_dir_tokens = [
        ("[1_type]", "[1] main_category"), ("[1]", "[1] sub_category"),
        ("[2]", "[2] display_name"), ("[2_wkt]", "[2] geo_wkt"),
        ("[3]", "[3] direction_desc"), ("[3_predicate]", "[3] direction_predicate"),
    ]
    knn_dir_sql = "\nWITH origin AS (SELECT ST_GeomFromText('[2_wkt]',4326)::geography AS geometry)\nSELECT * FROM pois, origin\nWHERE pois.[1_type] = '[1]'\nAND [3_predicate]\nORDER BY pois.geometry <-> origin.geometry ASC\nLIMIT 1;\n"
    knn_dir_v = _make_knn_verifier()  # reuse — checks [1] and [2]
    reg.append(("knn:direction+name", _load_templates("knn:direction+name"), knn_dir_vtypes, knn_dir_tokens, knn_dir_sql, "name", knn_dir_v))
    reg.append(("knn:direction+loc",  _load_templates("knn:direction+loc"),  knn_dir_vtypes, knn_dir_tokens, knn_dir_sql, "loc",  knn_dir_v))

    # ── KNN + towards ───────────────────────────────────────────────────────
    knn_towards_vtypes = [("[2]", "poi"), ("[3]", "poi distance_limited"), ("[1]", "poi mbr_limited")]
    knn_towards_tokens = [
        ("[1_type]", "[1] main_category"), ("[1]", "[1] sub_category"),
        ("[2]", "[2] display_name"), ("[2_wkt]", "[2] geo_wkt"),
        ("[3]", "[3] display_name"), ("[3_wkt]", "[3] geo_wkt"),
    ]
    knn_towards_sql = "\nWITH angle AS (SELECT degrees(ST_Azimuth(ST_GeomFromText('[2_wkt]',4326)::geography, ST_GeomFromText('[3_wkt]',4326)::geography)) AS value)\nSELECT * FROM pois, angle\nWHERE [1_type] = '[1]'\nAND (degrees(ST_Azimuth(ST_GeomFromText('[2_wkt]',4326)::geography, pois.geometry)) BETWEEN angle.value - 22.5 AND angle.value + 22.5)\nORDER BY pois.geometry <-> ST_GeomFromText('[2_wkt]',4326)::geography ASC\nLIMIT 1;\n"
    knn_towards_v = _make_towards_verifier(("[1]", "[2]", "[3]"))
    reg.append(("knn:towards+name", _load_templates("knn:towards+name"), knn_towards_vtypes, knn_towards_tokens, knn_towards_sql, "name", knn_towards_v))
    reg.append(("knn:towards+loc",  _load_templates("knn:towards+loc"),  knn_towards_vtypes, knn_towards_tokens, knn_towards_sql, "loc",  knn_towards_v))

    # ── KNN multi-source (both types share the same SQL, generation is plain knn) ──
    # multi_source1 and multi_source2 questions need Wikipedia lookups which are
    # handled separately.  We include basic stubs here so --list shows them.
    # Actual multi-source generation is done by generate_multi_source() below.
    reg.append(("knn+name+multi_source1", None, None, None, knn_sql, "name", knn_v))
    reg.append(("knn+name+multi_source2", None, None, None, knn_sql, "name", knn_v))

    # ── Intersects + count ──────────────────────────────────────────────────
    intersects_count_vtypes = [("[1]", "poi"), ("[2]", "region")]
    intersects_count_tokens = [
        ("[1_type]", "[1] main_category"), ("[1]", "[1] sub_category", "pluralize"),
        ("[2]", "[2] region_name"), ("[2_wkt]", "[2] geo_wkt"),
    ]
    intersects_count_sql = "SELECT COUNT(*) FROM pois\nWHERE [1_type] = '[1]'\nAND ST_Intersects(pois.geometry, ST_GeomFromText('[2_wkt]',4326))\n"
    reg.append(("intersects+count",
        _load_templates("intersects+count"),
        intersects_count_vtypes, intersects_count_tokens, intersects_count_sql, "count",
        _make_intersects_count_verifier()))

    # ── Intersects + area max ───────────────────────────────────────────────
    pl_vtypes = [("[1]", "park_lake"), ("[2]", "region")]
    pl_tokens = [
        ("[1_type]", "[1] main_category"), ("[1_table]", "[1] table"),
        ("[1]", "[1] sub_category"), ("[2]", "[2] region_name"), ("[2_wkt]", "[2] geo_wkt"),
    ]
    area_max_sql = "SELECT *, ST_Area([1_table].geometry::geography) AS computed_area FROM [1_table]\nWHERE [1_type] = '[1]'\nAND ST_Intersects([1_table].geometry::geography, ST_GeomFromText('[2_wkt]',4326)::geography)\nORDER BY computed_area DESC\nLIMIT 1;\n"
    reg.append(("intersects:area_max+name",
        _load_templates("intersects:area_max+name"),
        pl_vtypes, pl_tokens, area_max_sql, "name",
        _make_park_lake_verifier()))

    area_total_tokens = [
        ("[1_type]", "[1] main_category"), ("[1_table]", "[1] table"),
        ("[1]", "[1] sub_category", "pluralize"), ("[2]", "[2] region_name"), ("[2_wkt]", "[2] geo_wkt"),
    ]
    area_total_sql = "SELECT SUM(ST_Area([1_table].geometry::geography)) AS area FROM [1_table]\nWHERE [1_type] = '[1]'\nAND ST_Intersects([1_table].geometry::geography, ST_GeomFromText('[2_wkt]',4326)::geography)\n"
    reg.append(("intersects:area_total+area",
        _load_templates("intersects:area_total+area"),
        pl_vtypes, area_total_tokens, area_total_sql, "area",
        _make_park_lake_verifier(require_area=True)))

    # ── Intersects + length max / total ────────────────────────────────────
    length_max_sql = "SELECT * FROM [1_table]\nWHERE [1_type] = '[1]'\nAND ST_Intersects([1_table].geometry, ST_GeomFromText('[2_wkt]',4326)::geography)\nORDER BY ST_Length([1_table].geometry) DESC\nLIMIT 1;\n"
    reg.append(("intersects:length_max+name",
        _load_templates("intersects:length_max+name"),
        REGION_VARIABLE_TYPES, REGION_TEMPLATE_TOKENS, length_max_sql, "name",
        _make_road_waterway_verifier()))

    length_total_tokens = [
        ("[1_type]", "[1] main_category"), ("[1_table]", "[1] table"),
        ("[1]", "[1] sub_category_label", "pluralize"), ("[2]", "[2] region_name"), ("[2_wkt]", "[2] geo_wkt"),
    ]
    length_total_sql = "SELECT SUM(ST_Length([1_table].geometry::geography)) AS length FROM [1_table]\nWHERE [1_type] = '[1]'\nAND ST_Intersects([1_table].geometry::geography, ST_GeomFromText('[2_wkt]',4326)::geography);\n"
    reg.append(("intersects:length_total+length",
        _load_templates("intersects:length_total+length"),
        REGION_VARIABLE_TYPES, length_total_tokens, length_total_sql, "length",
        _make_road_waterway_verifier(require_length=True)))

    return reg


# ─────────────────────────────────────────────────────────────────────────────
# Multi-source generation (requires Wikipedia API)
# ─────────────────────────────────────────────────────────────────────────────

SESSION = requests.Session()
SESSION.headers.update({"Accept": "application/json", "User-Agent": "UCRPOIQuestionGenerator_Research/1.0 (contact:@ucr.edu)"})

picked_featuress = {
    "park": ["Architect", "Built", "Created"],
    "museum": ["Established", "Built", "Director", "Architect", "Founder", "Headquarters"],
    "zoo": ["Date opened"],
    "hospital": ["Opened", "Affiliated university", "Emergency department", "Helipad"],
    "aquarium": ["Date opened", "Volume of largest tank"],
    "stadium": ["Capacity", "Opened", "Construction", "Architect", "Former names"],
    "university": ["Established", "Motto", "Mascot", "Nickname", "Former names"],
    "hotel": ["Founded", "Number of locations"],
    "nature_reserve": ["Established", "Nearest\xa0city"],
    "theme_park": ["Founded", "Number of locations"],
    "golf_course": ["Designed by", "Architect"],
    "attraction": ["Opening date", "Built"],
}

picked_features_q_phrases = {
    "Architect": "What is the name of the architect that designed the closest [1] from [2]?",
    "Built": "When was the the nearest [1] from [2] built?",
    "Established": "When was the nearest [1] from [2] established?",
    "Director": "Who is the director of the nearest [1] from [2]?",
    "Founder": "Who founded the closest [1] from [2]?",
    "Headquarters": "Where is the headquarters of the nearest [1] from [2] located?",
    "Opened": "When was the nearest [1] from [2] first opened?",
    "Opening date": "On what date was the closest [1] from [2] opened?",
    "Affiliated university": "What is the name of the university that is affiliated with the closest [1] from [2]?",
    "Emergency department": "What type of emergency department is available at the nearest [1] from [2]?",
    "Helipad": "Does the nearest [1] from [2] have a helpad?",
    "Date opened": "When was the nearest [1] from [2] opened?",
    "Volume of largest tank": "What is the volume of the largest tank at the nearest [1] from [2]?",
    "Capacity": "How many spectators can the nearest [1] from [2] hold?",
    "Construction": "When was the nearest [1] from [2] constructed?",
    "Former names": "Can you tell me about a former name of the nearest [1] from [2]?",
    "Motto": "What is the motto of the nearest [1] from [2]?",
    "Mascot": "What is the mascot of the nearest [1] from [2]?",
    "Nickname": "What is the nickname of the closest [1] from [2]?",
    "Designed by": "Who designed the nearest [1] from [2]?",
    "Nearest\xa0city": "What is the closest city from the nearest [1] from [2]?",
}

picked_features_descriptors = {
    "Architect": "the %s designed by the architect %s",
    "Built": "the %s that was built in %s",
    "Established": "the %s established in the year %s",
    "Director": "the %s directed by %s",
    "Founder": "the %s founded by %s",
    "Affiliated university": "the %s affiliated with %s",
    "Emergency department": "the %s that has %s emergency department",
    "Helipad": "the %s that has a %s",
    "Date opened": "the %s that was opened on %s",
    "Capacity": "the %s that can hold %s spectators",
    "Construction": "the %s that was constructed in %s",
    "Former names": "the %s with the former name %s",
    "Motto": "the %s that has %s as its motto",
    "Mascot": "the %s that has %s as its mascot",
    "Nickname": "the %s that has the nickname %s",
    "Designed by": "the %s that was designed by %s",
    "Nearest\xa0city": "the %s that has %s as its nearest city",
}


def get_wikipedia_url_from_wikidata_id(wikidata_id, lang="en"):
    time.sleep(10)
    r = SESSION.get("https://www.wikidata.org/w/api.php", params={
        "action": "wbgetentities", "ids": wikidata_id,
        "props": "sitelinks", "sitefilter": f"{lang}wiki", "format": "json",
    }, timeout=15)
    r.raise_for_status()
    entity = r.json().get("entities", {}).get(wikidata_id, {})
    site = entity.get("sitelinks", {}).get(f"{lang}wiki")
    if not site or "title" not in site:
        return None
    return f"https://{lang}.wikipedia.org/wiki/{site['title'].replace(' ', '_')}"


def get_wikipedia_info(wikipedia_url, wikidataid):
    import pandas as pd
    cache_path = HERE / "wikipedia_cache" / f"{wikidataid}.json"
    txt_path   = cache_path.with_suffix(".txt")
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text())
        except Exception:
            return None
    if txt_path.exists():
        return None
    response = requests.get(wikipedia_url)
    txt_path.write_text(response.text)
    for cls, typ in [("infobox vcard", "vcard"), ("infobox", "infobox")]:
        try:
            info = pd.read_html(StringIO(response.text), index_col=0, attrs={"class": cls})
            j = {"data": info[0].to_dict()[info[0].columns[0]], "type": typ}
            cache_path.write_text(json.dumps(j, indent=2))
            return j
        except Exception:
            pass
    return None


def generate_multi_source(n, knn_sql, knn_verifier, style: int, output_dir: Path):
    """Generate knn+name+multi_source{style} questions (1 or 2)."""
    text_templates = _load_templates("knn+name")
    questions = []
    pbar = tqdm(total=n, desc=f"knn+name+multi_source{style}")

    while len(questions) < n:
        q = question_generator(text_templates, KNN_VARIABLE_TYPES, KNN_TEMPLATE_TOKENS,
                               knn_sql, "name", knn_verifier, 1, disable_progress=True)[0]
        if style == 1:
            answer    = q["answers"][0]
            sub_cat   = next((answer[k] for k in ("leisure", "amenity", "tourism") if k in answer), "")
            if sub_cat not in picked_featuress: continue
            if "wikidata" not in answer: continue
            try:
                url = get_wikipedia_url_from_wikidata_id(answer["wikidata"])
            except Exception:
                continue
            if url is None: continue
            info = get_wikipedia_info(url, answer["wikidata"])
            if info is None: continue
            possible = list(set(info["data"]) & set(picked_features_q_phrases))
            if not possible: continue
            key = random.choice(possible)
            q["question"] = picked_features_q_phrases[key].replace("[1]", q["question_entities"]["[1]"]["sub_category"]).replace("[2]", q["question_entities"]["[2]"]["display_name"])
            q["answers"][0]["multi_source_answer"]    = str(info["data"][key])
            q["answers"][0]["multi_source_attribute"] = key
            q["answers"][0]["multi_source_long_answer"] = q["answers"][0]["poi_name"] + " " + key + ": " + str(info["data"][key])
        else:  # style == 2
            e2        = q["question_entities"]["[2]"]
            sub_cat   = e2["sub_category"]
            disp_name = e2["display_name"]
            poi_name  = e2["poi"]["poi_name"]
            wikidata  = e2["poi"].get("wikidata")
            if sub_cat not in picked_featuress or wikidata is None: continue
            url = get_wikipedia_url_from_wikidata_id(wikidata)
            if url is None: continue
            info = get_wikipedia_info(url, wikidata)
            if info is None: continue
            possible = list(set(info["data"]) & set(picked_features_descriptors))
            if not possible: continue
            key   = random.choice(possible)
            attr  = str(info["data"][key])
            for ch in ("(", "[", ";"):
                if ch in attr: attr = attr[:attr.find(ch)]
            attr = attr.strip()
            if key in ("Built", "Created", "Established"):
                try:
                    attr = str(parse(attr, fuzzy=True).year)
                except Exception:
                    continue
            descriptor   = picked_features_descriptors[key] % (sub_cat.replace("_", " "), attr)
            new_disp     = disp_name.replace(poi_name + ",", descriptor + " in")
            q["question"] = q["question"].replace(disp_name, new_disp)

        questions.append(q)
        pbar.update(1)

    pbar.close()
    save(questions, f"knn+name+multi_source{style}.jsonl", output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    registry = build_registry()
    all_names = [name for name, *_ in registry]

    parser = argparse.ArgumentParser(description="Generate SpatialQA benchmark questions")
    parser.add_argument("--types", nargs="*", default=None,
                        metavar="TYPE", help="Template types to generate (default: all). Use --list to see options.")
    parser.add_argument("--list", action="store_true", help="List all available template types and exit.")
    parser.add_argument("--n", type=int, default=1000, help="Number of questions per type (default: 1000).")
    parser.add_argument("--output-dir", default="./questions", help="Output directory (default: ./questions).")
    args = parser.parse_args()

    if args.list:
        print("Available template types:")
        for name in all_names:
            print(f"  {name}")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = set(args.types) if args.types else set(all_names)
    unknown  = selected - set(all_names)
    if unknown:
        print(f"Unknown types: {sorted(unknown)}")
        print(f"Available: {all_names}")
        return

    knn_sql = "SELECT * FROM pois\nWHERE [1_type] = '[1]'\nORDER BY geometry <-> ST_GeomFromText('[2_wkt]',4326)::geography ASC\nLIMIT 1;\n"
    knn_v   = _make_knn_verifier()

    for entry in registry:
        type_name = entry[0]
        if type_name not in selected:
            continue
        print(f"\n{'='*60}\nGenerating {type_name} (n={args.n})\n{'='*60}")

        if type_name == "knn+name+multi_source1":
            generate_multi_source(args.n, knn_sql, knn_v, 1, output_dir)
            continue
        if type_name == "knn+name+multi_source2":
            generate_multi_source(args.n, knn_sql, knn_v, 2, output_dir)
            continue

        _, text_templates, variable_types, template_tokens, template_sql, answer_type, verifier = entry
        questions = question_generator(
            text_templates, variable_types, template_tokens, template_sql, answer_type, verifier, args.n
        )
        save(questions, f"{type_name}.jsonl", output_dir)

    print("\nAll done.")


if __name__ == "__main__":
    main()
