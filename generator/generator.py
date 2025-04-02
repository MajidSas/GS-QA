# %% [markdown]
# # Setup

# %%
import json
import random
import psycopg2
import shapely
from tqdm import tqdm
from textblob.blob import Word
import language_tool_python
import pandas as pd
import requests
import os
from dateutil.parser import parse
from io import StringIO  
def run_sql_select(sql, return_dict=False):
    conn = psycopg2.connect(
        host = 'localhost',
        dbname = 'osm_ca',
        user = 'postgres',
        password = 'postgres',
        port = 5432
    )
    cur = conn.cursor()
    cur.execute(sql)
    colnames = [desc[0] for desc in cur.description]
    records = cur.fetchall()
    cur.close()
    conn.close()
    if return_dict:
        answers_dict = []
        for i in range(len(records)):
            answers_dict.append({
                colnames[j]: records[i][j] for j in range(len(colnames)) if records[i][j] is not None
            })
        return answers_dict
    return records

output_folder = './questions/'
N = 1000

# %%
language_tool = language_tool_python.LanguageTool('en-US')
is_bad_rule = lambda rule: 'spelling' in rule.message.lower() and len(rule.replacements) and rule.replacements[0][0].isupper()


# %%
pois_selector = {
    'tourism': ['aquarium', 'attraction',
                       'viewpoint', 'art gallery', 'theme park', 'museum',
                       'bed and breakfast', 'gallery', 'zoo', 'hotel'],
    'amenity': ['restaurant', 'hospital', 'university', 'food', 'fast food restaurant', 'coffee shop', 'cafe'],
    'leisure': ['park', 'beach resort', 'golf course', 'nature reserve', 'garden', 'stadium']}

poi_filters = json.loads(open('filter_labels.json','r').read())


park_lake_selector = {
    'parks': ["recreation ground", "common", "nature reserve", "park", "garden", "sports centre", "golf course", "marina"],
    'water': ["bay", "harbour", "lake", "reservoir"],
}

road_waterway_selector = {
    'roads': ["road", "footway", "cycleway", "secondary", "pedestrian", "primary", "residential", "track", "motorway"],
    'waterway': ["canal", "river", "stream"],
    #'natural': ['coastline']
}

direction_filters = [
    ['north', '((degrees(ST_Azimuth(origin.geometry, pois.geometry)) BETWEEN 0.0 AND 22.5) OR (degrees(ST_Azimuth(origin.geometry, pois.geometry)) BETWEEN 337.5 AND 360))'],
    ['northeast', 'degrees(ST_Azimuth(origin.geometry, pois.geometry)) BETWEEN 22.5 AND 67.5'],
    ['east', 'degrees(ST_Azimuth(origin.geometry, pois.geometry)) BETWEEN 67.5 AND 112.5'],
    ['southeast', 'degrees(ST_Azimuth(origin.geometry, pois.geometry)) BETWEEN 112.5 AND 157.5'],
    ['south', 'degrees(ST_Azimuth(origin.geometry, pois.geometry)) BETWEEN 157.5 AND 202.5'],
    ['southwest', 'degrees(ST_Azimuth(origin.geometry, pois.geometry)) BETWEEN 202.5 AND 247.5'],
    ['west', 'degrees(ST_Azimuth(origin.geometry, pois.geometry)) BETWEEN 247.5 AND 292.5'],
    ['northwest', 'degrees(ST_Azimuth(origin.geometry, pois.geometry)) BETWEEN 292.5 AND 337.5'],
]


def poi_category_to_sql_name(value):
    if ' ' in value:
        if value == 'coffee shop':
            value = 'coffee'
        elif value == 'fast food restaurant':
            value = 'fast_food'
        else:
            value = value.replace(' ', '_')
    return value

regions_selector = ["city","town","village", "island","municipality","county","neighbourhood","suburb","state"]

sql_no_ref = "SELECT * FROM TABLE2  TABLESAMPLE SYSTEM(5) WHERE PREDICATE MUST_HAVE IS NOT NULL LIMIT 1"
sql_with_ref = "SELECT * FROM TABLE2  TABLESAMPLE SYSTEM(5) WHERE PREDICATE MUST_HAVE IS NOT NULL AND ST_DWithin(TABLE.geometry, ST_GeomFromText('WKT',4326)::geography, 1E5) LIMIT 1"

# %%
def question_generator(text_templates, variable_types, template_tokens, sql_template, answer_type, verifier, n=100, disable_progress=False):
    '''
    @text_templates: an array containing text templates with substrings that contain the template variables.
    @variable_types: the entity type for each variable
    @template_tokens: defines the tokens to be replaced in the template and how they should be populated from the selected instances in the based on the variable types
    @template_sql: sql query to retreive answers based on selected variables
    @answer_type: a string identifyibng the expected final answer for the text question
    @verifier: a function that verifies the selected variables and checks the answer
    @n: the number of questions to be generated
    '''
    generated_questions = []
    progress = None
    if not disable_progress:
        progress = tqdm(total=n)
    while len(generated_questions) < n:
        selected_entities = {}
        ref_geo_wkt = None
        skip_iteration = False
        geo_wkt = None
        poi_mbr = [1e100, 1e100, -1e100, -1e100]
        for k,v in variable_types:
            if 'poi_filter' in v:
                sub_category = random.choice(list(poi_filters.keys()))
                main_cateogory = None
                for kk in pois_selector:
                    if sub_category in pois_selector[kk]:
                        main_category = kk
                        break
                selected_filter = random.choice(poi_filters[sub_category])
                selected_entities[k] = {'main_category': main_category, 'sub_category': sub_category, 'poi_filter_desc':  selected_filter[0], 'poi_filter_sql': selected_filter[1]}  
            elif 'poi' in v:
                main_category = random.choice(list(pois_selector.keys()))
                sub_category = random.choice(pois_selector[main_category])
                if 'distance_limited' in v and ref_geo_wkt is not None:
                    output = run_sql_select(sql_with_ref.replace('TABLE2', 'pois').replace('MUST_HAVE', 'addr_state').replace('WKT', ref_geo_wkt).replace('PREDICATE', "%s ILIKE '%s' AND " % (main_category, poi_category_to_sql_name(sub_category))), return_dict=True)
                    if len(output) == 0:
                        skip_iteration = True
                        break
                    else:
                        random_poi = output[0]
                elif 'mbr_limited' in v and poi_mbr is not None:
                    mbr_wkt = shapely.box(*poi_mbr).wkt
                    output = run_sql_select(sql_no_ref.replace('TABLE2', 'pois').replace('MUST_HAVE', 'addr_state').replace('PREDICATE', "ST_Intersects(geometry, ST_GeomFromText('%s',4326)::geography) AND " % (mbr_wkt)), return_dict=True)
                    if len(output) == 0:
                        skip_iteration = True
                        break
                    else:
                        random_poi = output[0]
                        main_category = None
                        sub_category = None
                        for kk in pois_selector:
                            if kk in random_poi:
                                main_category = kk
                                sub_category = random_poi[kk]
                                break

                        # main category
                        # sub category
                else:
                    output = run_sql_select(sql_no_ref.replace('TABLE2', 'pois').replace('MUST_HAVE', 'addr_state').replace('PREDICATE', "%s ILIKE '%s' AND " % (main_category, poi_category_to_sql_name(sub_category))), return_dict=True)
                    if len(output) == 0:
                        skip_iteration = True
                        break
                    else:
                        random_poi = output[0]
                        ref_geo_wkt = shapely.to_wkt(shapely.from_wkb(random_poi['geometry']))
                display_name = random_poi['poi_name']
                if 'addr_city' in random_poi:
                    display_name += ', ' + random_poi['addr_city']
                if 'addr_state' in random_poi:
                    display_name += ', ' + random_poi['addr_state']
                geo_wkt = shapely.to_wkt(shapely.from_wkb(random_poi['geometry']))
                p = shapely.from_wkb(random_poi['geometry'])
                poi_mbr = [min(poi_mbr[0], p.x), min(poi_mbr[1], p.y), max(poi_mbr[2], p.x), max(poi_mbr[3], p.y)]
                random_poi['geometry'] = geo_wkt
                selected_entities[k] = {'main_category': main_category, 'sub_category': sub_category, 'display_name': display_name, 'geo_wkt': geo_wkt, 'poi': random_poi}
            elif 'region' in v:
                predicate = ''
                # if 'mbr_limited' in v:
                predicate =  " ST_Intersects(geometry, ST_GeomFromText('%s',4326)::geography) " % (geo_wkt)
                ss = '''SELECT * FROM regions WHERE PREDICATE AND wikipedia IS NOT NULL LIMIT 1'''.replace('PREDICATE', predicate)
                output = run_sql_select(ss, return_dict=True)
                print(len(output))
                if len(output) == 0:
                    skip_iteration = True
                    break
                random_region = output[0]
                print(random_region['wikipedia'])
                geo_wkt = shapely.to_wkt(shapely.from_wkb(random_region['geometry']))
                random_region['geometry'] = geo_wkt
                selected_entities[k] = {'region_name': random_region['wikipedia'][3:], 'geo_wkt': geo_wkt, 'region': random_region}
            elif 'park_lake' in v:
                __tmp = random.choice(list(park_lake_selector.keys()))
                table = 'parks' if __tmp == 'parks' else 'lakes'
                main_category = 'leisure' if table == 'parks' else __tmp
                sub_category = random.choice(list(park_lake_selector[__tmp]))
                predicate = 'leisure IS NOT NULL AND ' if main_category == 'leisure' else " (waterway ILIKE '%s' OR water ILIKE '%s') AND " % (sub_category, sub_category)
                predicate = predicate + ' ST_Area(geometry) > 0 AND '
                output = run_sql_select(sql_no_ref.replace('TABLE2', table).replace('MUST_HAVE', 'wikipedia').replace('PREDICATE', predicate), return_dict=True)
                if len(output) == 0:
                    skip_iteration = True
                    break
                random_obj = output[0]
                geo_wkt = shapely.to_wkt(shapely.from_wkb(random_obj['geometry']))
                random_obj['geometry'] = geo_wkt
                selected_entities[k] = {'park_name': random_obj['wikipedia'][3:], 'main_category': main_category, 'sub_category': sub_category, 'table': table, Word(table).singularize(): random_obj}
            elif 'road_waterway' in v:
                __tmp = random.choice(list(road_waterway_selector.keys()))
                table = 'roads' if __tmp == 'roads' else 'lakes'
                main_category = 'highway' if table == 'roads' else __tmp
                sub_category = random.choice(list(road_waterway_selector[__tmp]))
                if main_category == 'highway' and not (sub_category[-3:] == 'way' or sub_category == 'road'):
                    sub_category_label = sub_category + ' road'
                else:
                    sub_category_label = sub_category
                predicate = "(road_name IS NOT NULL OR wikipedia IS NOT NULL) AND highway =  '%s' " % sub_category if main_category == 'highway' else "(lake_name IS NOT NULL OR wikipedia IS NOT NULL) AND (waterway = '%s' OR water = '%s') " % (sub_category, sub_category)
                # predicate = predicate + ' ST_Length(geometry) > 0 AND '
                ss = '''
                SELECT * FROM TABLE2
                TABLESAMPLE SYSTEM(5)
                WHERE PREDICATE
                LIMIT 1;
                '''
                ss = ss.replace('TABLE2', table).replace('PREDICATE', predicate)
                # # ss = sql_no_ref.replace('TABLE2', table).replace('MUST_HAVE', 'wikipedia').replace('PREDICATE', predicate)
                # # print(ss)
                output = run_sql_select(ss, return_dict=True)
                print(sub_category, len(output))
                if len(output) == 0:
                    skip_iteration = True
                    break
                random_obj = output[0]
                geo_wkt = shapely.to_wkt(shapely.from_wkb(random_obj['geometry']))
                random_obj['geometry'] = geo_wkt
                selected_entities[k] = {'main_category': main_category, 'sub_category': sub_category, 'sub_category_label': sub_category_label, 'table': table, Word(table).singularize(): random_obj}
                # selected_entities[k] = {'main_category': main_category, 'sub_category': sub_category, 'sub_category_label': sub_category_label, 'table': table}
            elif 'distance' in v:
                distance = random.randint(1,200)
                distance = (distance - (distance % 10)) * 1000
                selected_entities[k] = {'distance': distance, 'text': '%d kilometers' % int(distance/1000.0)}
            elif 'direction' in v:
                dir_desc, dir_sql = random.choice(direction_filters)
                selected_entities[k] = {'direction_desc': dir_desc, 'direction_predicate': dir_sql}
        if skip_iteration:
            continue
        # print(selected_entities)
        t = random.choice(text_templates)
        sql = sql_template
        for tt in template_tokens:
            k, v = tt[0], tt[1]
            for _k in selected_entities:
                if _k in v:
                    _value = selected_entities[_k][v[v.find(' ')+1:]]
                    if k in sql:
                        if 'sub_category' in v:
                            value = poi_category_to_sql_name(str(_value))
                        else:
                            value = str(_value)
                        if 'name' in k:
                            value = value.replace("'", "''")
                        sql = sql.replace(k, value)
                    if k in t:
                        value = _value.replace('_', ' ')
                        if 'append_an' in tt:
                            an = 'an ' if value[0] in 'aeoiu' else 'a '
                            if 'any [1]' in t:
                                an = ''
                            value = an + value
                        elif 'pluralize' in tt:
                            value = Word(value).pluralize()
                        t = t.replace(k, value)
        # print(sql)
        # if not verifier(template_tokens, selected_entities, []):
        #     continue
        # answers = []
        # try:
        # predicate = "(road_name IS NOT NULL OR wikipedia IS NOT NULL) AND highway =  '%s' " % sub_category if main_category == 'highway' else "(lake_name IS NOT NULL OR wikipedia IS NOT NULL) AND (waterway = '%s' OR water = '%s') " % (sub_category, sub_category)
        _sql = sql.replace('WHERE', 'WHERE %s AND ' % predicate)
        answers = run_sql_select(_sql, return_dict=True)
        # except:
        #     answers = []
        if verifier(template_tokens, selected_entities, answers):
            matches = language_tool.check(t)
            matches = [rule for rule in matches if not is_bad_rule(rule)]
            t = language_tool_python.utils.correct(t, matches)
            for i in range(len(answers)):
                if 'geometry' in answers[i]:
                    answers[i]['geometry'] = shapely.to_wkt(shapely.from_wkb(answers[i]['geometry']))
            generated_questions.append({'question': t, 'sql': sql, 'answers': answers, 'answer_type': answer_type, 'question_entities': selected_entities})
            # print(len(generated_questions))
            if progress:
                progress.update(1)
    if progress:
        progress.close()
    return generated_questions
                

def save(questions, filename):
    with open(output_folder + filename, 'w') as f:
        for q in questions:
            f.write(json.dumps(q)+'\n')

# %%
# source: https://stackoverflow.com/questions/37079989/how-to-get-wikipedia-page-from-wikidata-id
def get_wikipedia_url_from_wikidata_id(wikidata_id, lang='en', debug=False):
    import requests
    from requests import utils
    url = (
        'https://www.wikidata.org/w/api.php'
        '?action=wbgetentities'
        '&props=sitelinks/urls'
        f'&sitefilter={lang}wiki'
        f'&ids={wikidata_id}'
        '&format=json')
    json_response = requests.get(url).json()
    if debug: print(wikidata_id, url, json_response) 

    entities = json_response.get('entities')    
    if entities:
        entity = entities.get(wikidata_id)
        if entity:
            sitelinks = entity.get('sitelinks')
            if sitelinks:
                if lang:
                    # filter only the specified language
                    sitelink = sitelinks.get(f'{lang}wiki')
                    if sitelink:
                        wiki_url = sitelink.get('url')
                        if wiki_url:
                            return requests.utils.unquote(wiki_url)
                else:
                    # return all of the urls
                    wiki_urls = {}
                    for key, sitelink in sitelinks.items():
                        wiki_url = sitelink.get('url')
                        if wiki_url:
                            wiki_urls[key] = requests.utils.unquote(wiki_url)
                    return wiki_urls
    return None   

# %%
def get_wikipedia_info(wikipedia_url, wikidataid):
    path = f'./wikipedia_cache/{wikidataid}.json'
    if os.path.exists(path):
        with open(path, 'r') as file:
            try:
                return json.loads(file.read())
            except:
                return None
    if os.path.exists(path.replace('json', 'txt')):
        return None
    response = requests.get(wikipedia_url)
    with open(path.replace('json', 'txt'), 'w') as file:
        file.write(response.text) 
    try:
        info = pd.read_html(StringIO(response.text), index_col=0, attrs={"class":"infobox vcard"})
        j = {"data": info[0].to_dict()[info[0].columns[0]], "type" : "vcard"}
        with open(path, 'w') as file:
            file.write(json.dumps(j, indent=2))
        return j
    except Exception as e1:
        print('e1', e1)
        try:
            info = pd.read_html(StringIO(response.text), index_col=0, attrs={"class":"infobox"})
            j = {"data": info[0].to_dict()[info[0].columns[0]], "type" : "infobox"}
            with open(path, 'w') as file:
                file.write(json.dumps(j, indent=2))
            return j
        except Exception as e2:
            print('e2', e2)
            return None

picked_featuress = {"park" : ["Architect", "Built", "Created"], 
                    "museum" : ["Established", "Built", "Director", "Architect", "Founder", "Headquarters"],
                    "zoo" : ["Date opened"],
                    "hospital" : ["Opened", "Affiliated university", "Emergency department", "Helipad"],
                    "aquarium" : ["Date opened", "Volume of largest tank"],
                    "stadium" : ["Capacity", "Opened", "Construction", "Architect", "Former names"],
                    "university" : ["Established", "Motto", "Mascot", "Nickname", "Former names"],
                    "hotel" : ["Founded", "Number of locations"],
                    "nature_reserve" : ["Established", "Nearest\xa0city"],
                    "theme_park" : ["Founded", "Number of locations"],
                    "golf_course" : ["Designed by", "Architect"],
                    "attraction" : ["Opening date", "Built"]}

picked_features_q_phrases = {
    "Architect": "What is the name of the architect that designed the closest [1] from [2]?",
    "Built": "When was the the nearest [1] from [2] built?",
    "Created": "Can you tell me when was the the closest [1] from [2] created?",
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
    "Nearest\xa0city": "What is the closest city from the nearest [1] from [2]?"
}

picked_features_descriptors = {
    "Architect": "the %s designed by the architect %s",
    "Built": "the %s that was built in %s",
    "Created": "the %s created in %s",
    "Established": "the %s established in the year %s",
    "Director": "the %s directed by %s",
    "Founder": "the %s founded by %s",
    "Headquarters": "the %s that has its headquarteres in %s",
    # "Opened": "the %s that was opened in the year %s",
    # "Opening date": "the %s that was opened on %s",
    "Affiliated university": "the %s affiliated with %s",
    "Emergency department": "the %s that has %s emergency department",
    "Helipad": "the %s that has a %s",
    "Date opened": "the %s that was opened on %s",
    "Volume of largest tank": "the %s with that has a tank with capacity %s",
    "Capacity": "the %s that can hold %s spectators",
    "Construction": "the %s that was constructed in %s",
    "Former names": "the %s with the former name %s",
    "Motto": "the %s that has %s as its motto",
    "Mascot": "the %s that has %s as its mascot",
    "Nickname": "the %s that has the nickname %s",
    "Designed by": "the %s that was designed by %s",
    "Nearest\xa0city": "the %s that has %s as its neasrest city"
}


# %% [markdown]
# # PART I: Range Templates

# %%
def verifier(template_tokens, selected_entities, answers):
    if poi_category_to_sql_name(selected_entities['[1]']['sub_category']) not in ['viewpoint', 'attraction', 'museum', 'theme park', 'hotel', 'gallery', 'zoo', 'aquarium', 'art_gallery', 'cafe', 'restaurant', 'fast_food', 'university', 'hospital', 'coffee', 'park', 'golf_course', 'nature_reserve', 'stadium', 'garden', 'beach_resort']:
        return False
    if poi_category_to_sql_name(selected_entities['[3]']['sub_category']) not in ['aquarium', 'attraction', 'viewpoint', 'art_gallery', 'theme_park', 'museum', 'gallery', 'zoo', 'hotel', 'university', 'park', 'nature_reserve', 'garden', 'stadium', 'hospital']:
        return False
    if poi_category_to_sql_name(selected_entities['[3]']['sub_category']) == poi_category_to_sql_name(selected_entities['[1]']['sub_category']):
        return False
    if len(answers) == 0:
        return False
    return True


# %%
variable_types = [
    ('[1]', 'poi'),
    ('[2]', 'distance'),
    ('[3]', 'poi'),
]

template_tokens = [
    ('[1_type]', '[1] main_category'),
    ('[1]', '[1] sub_category', 'append_an'),
    ('[2]', '[2] distance'),
    ('[2_text]', '[2] text'),
    ('[3]', '[3] display_name'),
    ('[3_wkt]', '[3] geo_wkt'),
]




template_sql = '''SELECT * FROM pois
WHERE ST_DWithin(pois.geometry, ST_GeomFromText('[3_wkt]',4326)::geography, [2])
AND [1_type] = '[1]';
'''


# %%
# Output type: entity name
text_templates =  [l.replace('[2]', '[2_text]').strip() for l in open('templates/range+name.txt','r').readlines()]
answer_type = 'name'
n = N
questions = question_generator(text_templates, variable_types, template_tokens, template_sql, answer_type, verifier, n)
save(questions, 'range+name.jsonl')

# %%
# Output type: location
text_templates =  [l.replace('[2]', '[2_text]').strip() for l in open('templates/range+loc.txt','r').readlines()]
answer_type = 'loc'
n = N
questions = question_generator(text_templates, variable_types, template_tokens, template_sql, answer_type, verifier, n)
save(questions, 'range+loc.jsonl')

# %%
# Output type: count

template_tokens = [
    ('[1_type]', '[1] main_category'),
    ('[1]', '[1] sub_category', 'pluralize'),
    ('[2]', '[2] distance'),
    ('[2_text]', '[2] text'),
    ('[3]', '[3] display_name'),
    ('[3_wkt]', '[3] geo_wkt'),
]

template_sql = '''SELECT COUNT(*)FROM pois
WHERE ST_DWithin(pois.geometry, ST_GeomFromText('[3_wkt]',4326)::geography, [2])
AND [1_type] = '[1]';
'''

text_templates =  [l.replace('[2]', '[2_text]').strip() for l in open('templates/range+count.txt','r').readlines()]
answer_type = 'count'
n = N
questions = question_generator(text_templates, variable_types, template_tokens, template_sql, answer_type, verifier, n)
save(questions, 'range+count.jsonl')

# %%
# Filters: range
# Output type: angle
template_tokens = [
    ('[1_type]', '[1] main_category'),
    ('[1]', '[1] sub_category', 'append_an'),
    ('[2]', '[2] distance'),
    ('[2_text]', '[2] text'),
    ('[3]', '[3] display_name'),
    ('[3_wkt]', '[3] geo_wkt'),
]

template_sql = '''SELECT *, degrees(ST_Azimuth(ST_GeomFromText('[3_wkt]',4326)::geography, pois.geometry)) AS angle FROM pois
WHERE ST_DWithin(pois.geometry, ST_GeomFromText('[3_wkt]',4326)::geography, [2])
AND [1_type] = '[1]';
'''

text_templates =  [l.replace('[2]', '[2_text]').strip() for l in open('templates/range+angle.txt','r').readlines()]
answer_type = 'angle'
n = N
questions = question_generator(text_templates, variable_types, template_tokens, template_sql, answer_type, verifier, n)
save(questions, 'range+angle.jsonl')

# %%
# Filters: range
# Output type: distance

template_sql = '''SELECT *, ST_Distance(ST_GeomFromText('[3_wkt]',4326)::geography, pois.geometry) AS distance FROM pois
WHERE ST_DWithin(pois.geometry, ST_GeomFromText('[3_wkt]',4326)::geography, [2])
AND [1_type] = '[1]';
'''

text_templates =  [l.replace('[2]', '[2_text]').strip() for l in open('templates/range+distance.txt','r').readlines()]
answer_type = 'distance'
n = N
questions = question_generator(text_templates, variable_types, template_tokens, template_sql, answer_type, verifier, n)
save(questions, 'range+distance.jsonl')

# %%
# Filters: range + non spatial filter
# Output type: entity name


text_templates =  [l.replace('[1]', '[1_filter]').replace('[2]', '[2_text]').strip() for l in open('templates/range:non_spat_filter+name.txt','r').readlines()]

variable_types = [
    ('[1]', 'poi_filter'),
    ('[2]', 'distance'),
    ('[3]', 'poi'),
]

template_tokens = [
    ('[1_type]', '[1] main_category'),
    ('[1]', '[1] sub_category'),
    ('[1_filter]', '[1] poi_filter_desc', 'append_an'),
    ('[1_predicate]', '[1] poi_filter_sql'),
    ('[2]', '[2] distance'),
    ('[2_text]', '[2] text'),
    ('[3]', '[3] display_name'),
    ('[3_wkt]', '[3] geo_wkt')
]


template_sql = '''SELECT * FROM pois
WHERE ST_DWithin(pois.geometry, ST_GeomFromText('[3_wkt]',4326)::geography, [2])
AND [1_type] = '[1]'
AND [1_predicate];
'''

answer_type = 'name'
n = N
questions = question_generator(text_templates, variable_types, template_tokens, template_sql, answer_type, verifier, n)
save(questions, 'range:non_spat_filter+name.jsonl')

# %%
# Filters: range + non spatial filter
# Output type: loc

text_templates =  [l.replace('[1]', '[1_filter]').replace('[2]', '[2_text]').strip() for l in open('templates/range:non_spat_filter+loc.txt','r').readlines()]
answer_type = 'loc'

questions = question_generator(text_templates, variable_types, template_tokens, template_sql, answer_type, verifier, n)
save(questions, 'range:non_spat_filter+loc.jsonl')

# %%
# Filters: range + direction
# Output type: name

variable_types = [
    ('[1]', 'poi'),
    ('[2]', 'distance'),
    ('[3]', 'poi'),
    ('[4]', 'direction')
]

template_tokens = [
    ('[1_type]', '[1] main_category'),
    ('[1]', '[1] sub_category'),
    ('[2]', '[2] distance'),
    ('[2_text]', '[2] text'),
    ('[3]', '[3] display_name'),
    ('[3_wkt]', '[3] geo_wkt'),
    ('[4]', '[4] direction_desc'),
    ('[4_predicate]', '[4] direction_predicate')
]



text_templates =  [l.replace('[2]', '[2_text]').strip() for l in open('templates/range:direction+name.txt','r').readlines()]

template_sql = '''
WITH origin AS  (SELECT ST_GeomFromText('[3_wkt]',4326)::geography AS geometry)
SELECT * FROM pois, origin
WHERE ST_DWithin(pois.geometry, ST_GeomFromText('[3_wkt]',4326)::geography, [2])
AND [1_type] = '[1]'
AND [4_predicate];
'''
answer_type = 'name'
n = N
questions = question_generator(text_templates, variable_types, template_tokens, template_sql, answer_type, verifier, n)
save(questions, 'range:direction+name.jsonl')

# %%
# Filters: range + direction
# Output type: loc
template_tokens = [
    ('[1_type]', '[1] main_category'),
    ('[1]', '[1] sub_category', 'append_an'),
    ('[2]', '[2] distance'),
    ('[2_text]', '[2] text'),
    ('[3]', '[3] display_name'),
    ('[3_wkt]', '[3] geo_wkt'),
    ('[4]', '[4] direction_desc'),
    ('[4_predicate]', '[4] direction_predicate')
]

text_templates =  [l.replace('[2]', '[2_text]').strip() for l in open('templates/range:direction+loc.txt','r').readlines()]

answer_type = 'loc'

questions = question_generator(text_templates, variable_types, template_tokens, template_sql, answer_type, verifier, n)
save(questions, 'range:direction+loc.jsonl')

# %%
# Filters: range + towards
# Output type: name

def verifier(template_tokens, selected_entities, answers):
    if poi_category_to_sql_name(selected_entities['[1]']['sub_category']) not in ['viewpoint', 'attraction', 'museum', 'theme park', 'hotel', 'gallery', 'zoo', 'aquarium', 'art gallery', 'cafe', 'restaurant', 'fast_food', 'university', 'hospital', 'coffee', 'park', 'golf_course', 'nature_reserve', 'stadium', 'garden', 'beach_resort']:
        return False
    if poi_category_to_sql_name(selected_entities['[3]']['sub_category']) not in ['aquarium', 'attraction', 'viewpoint', 'art_gallery', 'theme_park', 'museum', 'gallery', 'zoo', 'hotel', 'university', 'park', 'nature_reserve', 'garden', 'stadium', 'hospital']:
        return False
    if poi_category_to_sql_name(selected_entities['[4]']['sub_category']) not in ['aquarium', 'attraction', 'viewpoint', 'art_gallery', 'theme_park', 'museum', 'gallery', 'zoo', 'hotel', 'university', 'park', 'nature_reserve', 'garden', 'stadium', 'hospital']:
        return False
    if len(set([poi_category_to_sql_name(selected_entities['[1]']['sub_category']), poi_category_to_sql_name(selected_entities['[3]']['sub_category']), poi_category_to_sql_name(selected_entities['[4]']['sub_category'])])) < 3:
        return False
    if len(answers) == 0:
        return False
    return True


variable_types = [
    ('[2]', 'distance'),
    ('[3]', 'poi'),
    ('[4]', 'poi distance_limited'),
    ('[1]', 'poi mbr_limited'),
]

template_tokens = [
    ('[1_type]', '[1] main_category'),
    ('[1]', '[1] sub_category', ),
    ('[2]', '[2] distance'),
    ('[2_text]', '[2] text'),
    ('[3]', '[3] display_name'),
    ('[3_wkt]', '[3] geo_wkt'),
    ('[4]', '[4] display_name'),
    ('[4_wkt]', '[4] geo_wkt')
]

template_sql = '''WITH
    angle AS (SELECT degrees(ST_Azimuth(ST_GeomFromText('[3_wkt]',4326)::geography, ST_GeomFromText('[4_wkt]',4326)::geography)) AS value)
SELECT * FROM pois,angle
WHERE [1_type] = '[1]'
AND ST_DWithin(pois.geometry, ST_GeomFromText('[3_wkt]',4326)::geography, [2])
AND (degrees(ST_Azimuth(ST_GeomFromText('[3_wkt]',4326)::geography, pois.geometry)) BETWEEN angle.value - 22.5 AND angle.value + 22.5);
'''

n = N

text_templates =  [l.replace('[2]', '[2_text]').strip() for l in open('templates/range:towards+name.txt','r').readlines()]

answer_type = 'name'
questions = question_generator(text_templates, variable_types, template_tokens, template_sql, answer_type, verifier, n)
save(questions, 'range:towards+name.jsonl')

# %%
# Filters: range + towards
# Output type: loc


text_templates =  [l.replace('[2]', '[2_text]').strip() for l in open('templates/range:towards+loc.txt','r').readlines()]

answer_type = 'loc'
questions = question_generator(text_templates, variable_types, template_tokens, template_sql, answer_type, verifier, n)
save(questions, 'range:towards+loc.jsonl')

# %% [markdown]
# # KNN Templates

# %%
def verifier(template_tokens, selected_entities, answers):
    if poi_category_to_sql_name(selected_entities['[1]']['sub_category']) not in ['viewpoint', 'attraction', 'museum', 'theme_park', 'hotel', 'gallery', 'zoo', 'aquarium', 'art_gallery', 'cafe', 'restaurant', 'fast_food', 'university', 'hospital', 'coffee', 'park', 'golf_course', 'nature_reserve', 'stadium', 'garden', 'beach_resort']:
        return False
    if poi_category_to_sql_name(selected_entities['[2]']['sub_category']) not in ['aquarium', 'attraction', 'viewpoint', 'art_gallery', 'theme_park', 'museum', 'gallery', 'zoo', 'hotel', 'university', 'park', 'nature_reserve', 'garden', 'stadium', 'hospital']:
        return False
    if poi_category_to_sql_name(selected_entities['[2]']['sub_category']) == poi_category_to_sql_name(selected_entities['[1]']['sub_category']):
        return False
    if len(answers) == 0:
        return False
    return True


variable_types = [
    ('[1]', 'poi'),
    ('[2]', 'poi'),
]

template_tokens = [
    ('[1_type]', '[1] main_category'),
    ('[1]', '[1] sub_category'),
    ('[2]', '[2] display_name'),
    ('[2_wkt]', '[2] geo_wkt'),
]




template_sql = '''SELECT * FROM pois
WHERE [1_type] = '[1]'
ORDER BY  geometry <-> ST_GeomFromText('[2_wkt]',4326)::geography ASC
LIMIT 1;
'''

n = N



# %%
text_templates =  [l.strip() for l in open('templates/knn+name.txt','r').readlines()]

answer_type = 'name'
questions = question_generator(text_templates, variable_types, template_tokens, template_sql, answer_type, verifier, n)
save(questions, 'knn+name.jsonl')

# %%
text_templates =  [l.strip() for l in open('templates/knn+loc.txt','r').readlines()]

answer_type = 'loc'
questions = question_generator(text_templates, variable_types, template_tokens, template_sql, answer_type, verifier, n)
save(questions, 'knn+loc.jsonl')


# %%

template_sql = '''SELECT * FROM pois
WHERE [1_type] = '[1]'
ORDER BY  geometry <-> ST_GeomFromText('[2_wkt]',4326)::geography ASC
LIMIT 1;
'''
text_templates =  [l.strip() for l in open('templates/knn+name.txt','r').readlines()]

answer_type = 'name'
# type 1: append another question at the end, asking about attribute extracted from wikipedia
questions = []
while len(questions) < 150:
    question = question_generator(text_templates, variable_types, template_tokens, template_sql, answer_type, verifier, 1, True)[0]
    answer = question['answers'][0]
    sub_category = ''
    if 'leisure' in answer:
        sub_category = answer['leisure']
    elif 'amenity' in answer:
        sub_category = answer['amenity']
    elif 'tourism' in answer:
        sub_category = answer['tourism']
    if sub_category not in picked_featuress:
        continue
    if 'wikidata' not in answer:
        print('no wikidata id')
        continue
    wikipedia_url = get_wikipedia_url_from_wikidata_id(answer['wikidata'])
    if wikipedia_url == None:
        continue
    info = get_wikipedia_info(wikipedia_url, answer['wikidata'])
    if info == None:
        continue
    possible_values = list(set(info['data'].keys()) & set(picked_features_q_phrases.keys()))
    if len(possible_values) == 0:
        continue
    selected_key = random.choice(possible_values)
    multihop_answer = info['data'][selected_key]
    multihop_q_phrase = picked_features_q_phrases[selected_key]
    question['question'] = multihop_q_phrase.replace('[1]', question['question_entities']['[1]']['sub_category']).replace('[2]', question['question_entities']['[2]']['display_name'])
    question['answers'][0]['multihop_answer'] = str(multihop_answer)
    question['answers'][0]['multihop_attribute'] = selected_key
    question['answers'][0]['multihop_long_answer'] = question['answers'][0]['poi_name'] + ' ' + selected_key + ': ' + str(multihop_answer)
    questions.append(question)
    print('Questions count: ' + str(len(questions)))
save(questions, 'knn+name+multihop1.jsonl')

# %%
# type 2: replace the name of the anchoring entity with one of its attributes from wikipedia
template_sql = '''SELECT * FROM pois
WHERE [1_type] = '[1]'
ORDER BY  geometry <-> ST_GeomFromText('[2_wkt]',4326)::geography ASC
LIMIT 1;
'''
text_templates =  [l.strip() for l in open('templates/knn+name.txt','r').readlines()]

answer_type = 'name'

questions = []
while len(questions) < 150:
    question = question_generator(text_templates, variable_types, template_tokens, template_sql, answer_type, verifier, 1, disable_progress=True)[0]
    answer = question['answers'][0]
    sub_category = question['question_entities']['[2]']['sub_category']
    display_name = question['question_entities']['[2]']['display_name']
    poi_name = question['question_entities']['[2]']['poi']['poi_name']
    wikidata = None if 'wikidata' not in question['question_entities']['[2]']['poi'] else question['question_entities']['[2]']['poi']['wikidata']
    if sub_category not in picked_featuress:
        print('sub category not accepted')
        continue
    if wikidata == None:
        print('no wikidata id')
    wikipedia_url = get_wikipedia_url_from_wikidata_id(wikidata)
    if wikipedia_url == None:
        print('no wikipedia url')
        continue
    info = get_wikipedia_info(wikipedia_url, wikidata)
    if info == None:
        print('no wikipedia info')
        continue
    # break
    possible_values = list(set(info['data'].keys()) & set(picked_features_descriptors.keys()))
    # print(set(info['data'].keys()), possible_values)
    if len(possible_values) == 0:
        print('no possible values')
        continue
    selected_key = random.choice(possible_values)
    multihop_attribute = str(info['data'][selected_key])
    ob = multihop_attribute.find('(')
    if ob != -1:
        multihop_attribute = multihop_attribute[:ob]
    ob = multihop_attribute.find('[')
    if ob != -1:
        multihop_attribute = multihop_attribute[:ob]
    ob = multihop_attribute.find(';')
    if ob != -1:
        multihop_attribute = multihop_attribute[:ob]
    if selected_key in ['Built', 'Created', 'Established']:
        try:
            multihop_attribute = parse(multihop_attribute, fuzzy=True).year
        except Exception as e:
            print(e)
            continue
    multihop_discriptor = picked_features_descriptors[selected_key] % (sub_category.replace('_', ' '), str(multihop_attribute))
    new_display_name = display_name.replace(poi_name + ',', multihop_discriptor + ' in')
    print(display_name, '##' , new_display_name)
    question['question'] = question['question'].replace(display_name, new_display_name)
    questions.append(question)
    print('Questions count: ' + str(len(questions)))
save(questions, 'knn+name+multihop2.jsonl')

# %%

# %%
text_templates =  [l.strip() for l in open('templates/knn+distance.txt','r').readlines()]

template_sql = '''SELECT *, ST_Distance(geometry, ST_GeomFromText('[2_wkt]',4326)::geography)  AS distance  FROM pois
WHERE [1_type] = '[1]'
ORDER BY geometry <-> ST_GeomFromText('[2_wkt]',4326)::geography ASC
LIMIT 1;
'''

answer_type = 'distance'
questions = question_generator(text_templates, variable_types, template_tokens, template_sql, answer_type, verifier, n)
save(questions, 'knn+distance.jsonl')


# %%
text_templates =  [l.strip() for l in open('templates/knn+angle.txt','r').readlines()]

template_sql = '''SELECT *, degrees(ST_Azimuth(ST_GeomFromText('[2_wkt]',4326)::geography, pois.geometry)) AS angle  FROM pois
WHERE [1_type] = '[1]'
ORDER BY geometry <-> ST_GeomFromText('[2_wkt]',4326)::geography ASC
LIMIT 1;
'''

answer_type = 'angle'
questions = question_generator(text_templates, variable_types, template_tokens, template_sql, answer_type, verifier, n)
save(questions, 'knn+angle.jsonl')


# %%
variable_types = [
    ('[1]', 'poi_filter'),
    ('[2]', 'poi'),
]

template_tokens = [
    ('[1_type]', '[1] main_category'),
    ('[1]', '[1] sub_category'),
    ('[1_filter]', '[1] poi_filter_desc'),
    ('[1_predicate]', '[1] poi_filter_sql'),
    ('[2]', '[2] display_name'),
    ('[2_wkt]', '[2] geo_wkt'),
]




template_sql = '''SELECT * FROM pois
WHERE [1_type] = '[1]'
AND [1_predicate]
ORDER BY geometry <-> ST_GeomFromText('[2_wkt]',4326)::geography ASC
LIMIT 1;
'''

n = N

text_templates =  [l.replace('[1]', '[1_filter]').strip() for l in open('templates/knn:non_spat_filter+name.txt','r').readlines()]

answer_type = 'name'
questions = question_generator(text_templates, variable_types, template_tokens, template_sql, answer_type, verifier, n)
save(questions, 'knn:non_spat_filter+name.jsonl')


# %%
text_templates =  [l.replace('[1]', '[1_filter]').strip() for l in open('templates/knn:non_spat_filter+loc.txt','r').readlines()]

answer_type = 'loc'
questions = question_generator(text_templates, variable_types, template_tokens, template_sql, answer_type, verifier, n)
save(questions, 'knn:non_spat_filter+loc.jsonl')


# %%
def verifier(template_tokens, selected_entities, answers):
    if poi_category_to_sql_name(selected_entities['[1]']['sub_category']) not in ['viewpoint', 'attraction', 'museum', 'theme park', 'hotel', 'gallery', 'zoo', 'aquarium', 'art gallery', 'cafe', 'restaurant', 'fast_food', 'university', 'hospital', 'coffee', 'park', 'golf_course', 'nature_reserve', 'stadium', 'garden', 'beach_resort']:
        return False
    if poi_category_to_sql_name(selected_entities['[2]']['sub_category']) not in ['aquarium', 'attraction', 'viewpoint', 'art_gallery', 'theme_park', 'museum', 'gallery', 'zoo', 'hotel', 'university', 'park', 'nature_reserve', 'garden', 'stadium', 'hospital']:
        return False
    if poi_category_to_sql_name(selected_entities['[1]']['sub_category']) == poi_category_to_sql_name(selected_entities['[2]']['sub_category']):
        return False
    if len(answers) == 0:
        return False
    return True

variable_types = [
    ('[1]', 'poi'),
    ('[2]', 'poi'),
    ('[3]', 'direction')
]

template_tokens = [
    ('[1_type]', '[1] main_category'),
    ('[1]', '[1] sub_category', ),
    ('[2]', '[2] display_name'),
    ('[2_wkt]', '[2] geo_wkt'),
    ('[3]', '[3] direction_desc'),
    ('[3_predicate]', '[3] direction_predicate')
]

template_sql = '''
WITH origin AS (SELECT ST_GeomFromText('[2_wkt]',4326)::geography AS geometry)
SELECT * FROM pois,origin
WHERE pois.[1_type] = '[1]'
AND [3_predicate]
ORDER BY pois.geometry <-> origin.geometry ASC
LIMIT 1;
'''


text_templates =  [l.strip() for l in open('templates/knn:direction+name.txt','r').readlines()]

answer_type = 'name'
questions = question_generator(text_templates, variable_types, template_tokens, template_sql, answer_type, verifier, n)
save(questions, 'knn:direction+name.jsonl')


# %%
text_templates =  [l.strip() for l in open('templates/knn:direction+loc.txt','r').readlines()]

answer_type = 'loc'
questions = question_generator(text_templates, variable_types, template_tokens, template_sql, answer_type, verifier, n)
save(questions, 'knn:direction+loc.jsonl')


# %%
def verifier(template_tokens, selected_entities, answers):
    if poi_category_to_sql_name(selected_entities['[1]']['sub_category']) not in ['viewpoint', 'attraction', 'museum', 'theme park', 'hotel', 'gallery', 'zoo', 'aquarium', 'art gallery', 'cafe', 'restaurant', 'fast_food', 'university', 'hospital', 'coffee', 'park', 'golf_course', 'nature_reserve', 'stadium', 'garden', 'beach_resort']:
        return False
    if poi_category_to_sql_name(selected_entities['[2]']['sub_category']) not in ['aquarium', 'attraction', 'viewpoint', 'art_gallery', 'theme_park', 'museum', 'gallery', 'zoo', 'hotel', 'university', 'park', 'nature_reserve', 'garden', 'stadium', 'hospital']:
        return False
    if poi_category_to_sql_name(selected_entities['[3]']['sub_category']) not in ['aquarium', 'attraction', 'viewpoint', 'art_gallery', 'theme_park', 'museum', 'gallery', 'zoo', 'hotel', 'university', 'park', 'nature_reserve', 'garden', 'stadium', 'hospital']:
        return False
    if len(set([poi_category_to_sql_name(selected_entities['[1]']['sub_category']), poi_category_to_sql_name(selected_entities['[2]']['sub_category']), poi_category_to_sql_name(selected_entities['[3]']['sub_category'])])) < 3:
        return False
    if len(answers) == 0:
        return False
    return True


variable_types = [
    ('[2]', 'poi'),
    ('[3]', 'poi distance_limited'),
    ('[1]', 'poi mbr_limited'),
]

template_tokens = [
    ('[1_type]', '[1] main_category'),
    ('[1]', '[1] sub_category', ),
    ('[2]', '[2] display_name'),
    ('[2_wkt]', '[2] geo_wkt'),
    ('[3]', '[3] display_name'),
    ('[3_wkt]', '[3] geo_wkt')
]

template_sql = '''WITH
    angle AS (SELECT degrees(ST_Azimuth(ST_GeomFromText('[2_wkt]',4326)::geography, ST_GeomFromText('[3_wkt]',4326)::geography)) AS value)
SELECT * FROM pois,angle
WHERE [1_type] = '[1]'
AND (degrees(ST_Azimuth(ST_GeomFromText('[2_wkt]',4326)::geography, pois.geometry)) BETWEEN angle.value - 22.5 AND angle.value + 22.5)
ORDER BY pois.geometry <-> ST_GeomFromText('[2_wkt]',4326)::geography ASC
LIMIT 1;
'''


text_templates =  [l.strip() for l in open('templates/knn:towards+name.txt','r').readlines()]

answer_type = 'name'
questions = question_generator(text_templates, variable_types, template_tokens, template_sql, answer_type, verifier, n)
save(questions, 'knn:towards+name.jsonl')

# %%
text_templates =  [l.strip() for l in open('templates/knn:towards+loc.txt','r').readlines()]

answer_type = 'loc'
questions = question_generator(text_templates, variable_types, template_tokens, template_sql, answer_type, verifier, n)
save(questions, 'knn:towards+loc.jsonl')

# %% [markdown]
# # Topological Templates

# %%
# Intersects + Count

def verifier(template_tokens, selected_entities, answers):
    if selected_entities['[1]']['sub_category'] not in ['viewpoint', 'attraction', 'museum', 'theme park', 'hotel', 'gallery', 'zoo', 'aquarium', 'art gallery', 'cafe', 'restaurant', 'fast food restaurant', 'university', 'hospital', 'coffee shop', 'park', 'golf course', 'nature reserve', 'stadium', 'garden', 'beach resort']:
        return False
    if len(answers) == 0:
        return False
    elif answers[0]['count'] == 0:
        return False
    return True

variable_types = [
    ('[1]', 'poi'),
    ('[2]', 'region'),
]

template_tokens = [
    ('[1_type]', '[1] main_category'),
    ('[1]', '[1] sub_category', 'pluralize'),
    ('[2]', '[2] region_name'),
    ('[2_wkt]', '[2] geo_wkt'),
]




template_sql = '''SELECT COUNT(*) FROM pois
WHERE [1_type] = '[1]'
AND ST_Intersects(pois.geometry, ST_GeomFromText('[2_wkt]',4326))
'''

n = N

text_templates =  [l.strip() for l in open('templates/intersects+count.txt','r').readlines()]

answer_type = 'count'
questions = question_generator(text_templates, variable_types, template_tokens, template_sql, answer_type, verifier, n)
save(questions, 'intersects+count.jsonl')


# %%
# Intersects + area max

def verifier(template_tokens, selected_entities, answers):
    # print(selected_entities)
    if selected_entities['[1]']['sub_category'] not in ["bay", "harbour", "lake", "reservoir"] and selected_entities['[1]']['main_category'] != 'leisure':
        return False
    if len(answers) == 0:
        return False
    return True

variable_types = [
    ('[1]', 'park_lake'),
    ('[2]', 'region'),
]

template_tokens = [
    ('[1_type]', '[1] main_category'),
    ('[1_table]', '[1] table'),
    ('[1]', '[1] sub_category'),
    ('[2]', '[2] region_name'),
    ('[2_wkt]', '[2] geo_wkt'),
]




template_sql = '''SELECT *, ST_Area([1_table].geometry::geography) AS computed_area FROM [1_table]
WHERE [1_type] = '[1]'
AND ST_Intersects([1_table].geometry::geography, ST_GeomFromText('[2_wkt]',4326)::geography)
ORDER BY computed_area DESC
LIMIT 1;
'''

n = N

text_templates =  [l.strip() for l in open('templates/intersects:area_max+name.txt','r').readlines()]

answer_type = 'name'
questions = question_generator(text_templates, variable_types, template_tokens, template_sql, answer_type, verifier, n)
save(questions, 'intersects:area_max+name.jsonl')


# %%
def verifier(template_tokens, selected_entities, answers):
    # print(selected_entities)
    if selected_entities['[1]']['sub_category'] not in ["bay", "harbour", "lake", "reservoir"] and selected_entities['[1]']['main_category'] != 'leisure':
        return False
    if len(answers) == 0:
        return False
    if 'area' not in answers[0] or answers[0]['area'] == 0:
        return False
    return True

variable_types = [
    ('[1]', 'park_lake'),
    ('[2]', 'region'),
]

template_tokens = [
    ('[1_type]', '[1] main_category'),
    ('[1_table]', '[1] table'),
    ('[1]', '[1] sub_category', 'pluralize'),
    ('[2]', '[2] region_name'),
    ('[2_wkt]', '[2] geo_wkt'),
]




template_sql = '''SELECT SUM(ST_Area([1_table].geometry::geography)) AS area FROM [1_table]
WHERE [1_type] = '[1]'
AND ST_Intersects([1_table].geometry::geography, ST_GeomFromText('[2_wkt]',4326)::geography)
'''

n = N

text_templates =  [l.strip() for l in open('templates/intersects:area_total+area.txt','r').readlines()]

answer_type = 'area'
questions = question_generator(text_templates, variable_types, template_tokens, template_sql, answer_type, verifier, n)
save(questions, 'intersects:area_total+area.jsonl')


# %%


# %%


def verifier(template_tokens, selected_entities, answers):
    # print(selected_entities)
    # if selected_entities['[1]']['sub_category'] not in ["bay", "harbour", "lake", "reservoir"] and selected_entities['[1]']['main_category'] != 'leisure':
    #     return False
    if len(answers) == 0:
        return False
    # print(answers[0])
    if not ('wikipedia' in answers[0] or 'lake_name' in answers[0] or 'road_name' in answers[0]):
        return False
    # if 'length' not in answers[0] or answers[0]['length'] == 0:
    #     return False
    return True

variable_types = [
    ('[1]', 'road_waterway'),
    ('[2]', 'region'),
]

template_tokens = [
    ('[1_type]', '[1] main_category'),
    ('[1_table]', '[1] table'),
    ('[1]', '[1] sub_category_label'),
    ('[2]', '[2] region_name'),
    ('[2_wkt]', '[2] geo_wkt'),
]




template_sql = '''SELECT * FROM [1_table]
WHERE [1_type] = '[1]'
AND ST_Intersects([1_table].geometry, ST_GeomFromText('[2_wkt]',4326)::geography)
ORDER BY ST_Length([1_table].geometry) DESC
LIMIT 1;
'''

n = N

text_templates =  [l.strip() for l in open('templates/intersects:length_max+name.txt','r').readlines()]

answer_type = 'name'
questions = question_generator(text_templates, variable_types, template_tokens, template_sql, answer_type, verifier, n)
save(questions, 'intersects:length_max+name.jsonl')

# %%
def verifier(template_tokens, selected_entities, answers):
    if len(answers) == 0:
        return False
    if 'length' not in answers[0] or answers[0]['length'] == 0:
        return False
    return True

variable_types = [
    ('[1]', 'road_waterway'),
    ('[2]', 'region'),
]

template_tokens = [
    ('[1_type]', '[1] main_category'),
    ('[1_table]', '[1] table'),
    ('[1]', '[1] sub_category_label', 'pluralize'),
    ('[2]', '[2] region_name'),
    ('[2_wkt]', '[2] geo_wkt'),
]




template_sql = '''SELECT SUM(ST_Length([1_table].geometry::geography)) AS length FROM [1_table]
WHERE [1_type] = '[1]'
AND ST_Intersects([1_table].geometry::geography, ST_GeomFromText('[2_wkt]',4326)::geography);
'''

n = N

text_templates =  [l.strip() for l in open('templates/intersects:length_total+length.txt','r').readlines()]

answer_type = 'length'
questions = question_generator(text_templates, variable_types, template_tokens, template_sql, answer_type, verifier, n)
save(questions, 'intersects:length_total+length.jsonl')

