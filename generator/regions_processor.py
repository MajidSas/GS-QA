# %%
import json
from shapely.geometry import shape
from shapely import wkb, wkt
import psycopg2
from glob import glob

# %%
region_files = glob('./osm_extract/postal_codes/*.geojson')


# %%
type_map = {
            type(""): "string",
            type(2342): "integer",
            type(322.2323): "float",
            type({'abc': 231}): "string"
        }
def define_schema(features, useTags = True):
    aggregate_schema = {}
    for i in range(len(features)):
        tags = features[i]['properties']['tagsMap'] if useTags else features[i]['properties']
        for k in tags:
            # if 'wiki' in k or 'addr' in k or 'gnis' in k or 'source' in k or ':' in k:
            #     continue
            aggregate_schema[k] = aggregate_schema.get(k, 0) + 1
    filtered_schema = {k: aggregate_schema[k] for k in aggregate_schema if (aggregate_schema[k] > 100 or not useTags)}
    final_schema = {
        "geometry": {
            "type": "string",
            "coordinates": "array[double]"
        }
    }
    for i in range(len(features)):
        tags = features[i]['properties']['tagsMap'] if useTags else features[i]['properties']
        for k in tags:
            if k not in filtered_schema:
                continue
            final_schema[k] = type_map[type(tags[k])]
        if len(final_schema) == len(filtered_schema)+1:
            break
    return final_schema

def schema_to_sql(schema, table_name):
    columns = ''
    type_map = {
        "string": "VARCHAR(255)",
        "integer": "BIGINT",
        "float": "DOUBLE"
    }
    for k in schema:
        if k != 'geometry':
            columns += '        %s %s,\n' % (k, type_map[schema[k]])
    columns = columns[:-2]
    'customer_name VARCHAR(255) NOT NULL'
    create_table = '''
    DROP TABLE IF EXISTS %s CASCADE;
    CREATE TABLE %s (
        id SERIAL PRIMARY KEY,
        geometry GEOGRAPHY(GEOMETRY, 4326),\n%s
    );
    ''' % (table_name, table_name, columns)
    return create_table

def run_sql(sql):
    conn = psycopg2.connect(
        host = 'localhost',
        dbname = 'osm_ca',
        user = 'postgres',
        password = 'postgres',
        port = 5432
    )
    cur = conn.cursor()
    cur.execute(sql)
    conn.commit()
    cur.close()
    conn.close()


def insert_rows_sql(table_name, schema, features, useTags=True):
    columns = []
    for k in schema:
        if k != 'geometry':
            columns.append(k)
    values = ''
    for f in features:
        if 'name' in f['properties']['tagsMap']:
        #     continue
            f['properties']['tagsMap']['region_name'] = f['properties']['tagsMap']['name']
        f['properties']['tagsMap']['osm_id'] = f['properties']['id']
        value = "\n        ('%s'" % shape(f['geometry']).wkt
        tags = f['properties']['tagsMap'] if useTags else f['properties']
        for c in columns:
            if c in tags:
                if schema[c] == 'string':
                    value += ",'%s'" % str(tags[c]).replace("'", "''")
                else:
                    value += ",%s" % str(tags[c])
            else:
                value += ',NULL'
        value += '),'
        values += value
    values = values[:-1]
    columns = 'geometry,' + ','.join(columns)
    query = '''
    INSERT INTO %s (%s) 
    VALUES%s;
    ''' % (table_name, columns, values)
    return query


# %%
regions = []
for f in region_files:
    regions += json.loads(open(f, "r").read())['features']
    print(len(regions))

# %%
# schema = define_schema(regions)
# {k: schema[k] for k in schema if 'tiger' not in k and 'name:' not in k and 'alt_' not in k}

# %%
# regions[:10]

# %%
# key = 'admin_level'# 'admin_level' # 'is_in' # 'place' # border_type
# set([(r['properties']['tagsMap'][key], r['properties']['tagsMap']['border_type']) for r in regions if key in r['properties']['tagsMap'] and 'border_type' in r['properties']['tagsMap']])

# %%
schema = json.loads(open('region_schema.json','r').read())

run_sql(schema_to_sql(schema, 'regions'))


# %%
def insert_features(features):
    for i in range(0,len(features), 100):
        _features = features[i:min(len(features),i+100)]
        run_sql(insert_rows_sql('regions', schema, _features, useTags=True))
        # print(str(i+100) + ' out of ' + len(features))

insert_features(regions)



