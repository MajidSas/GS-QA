# GS-QA: A Benchmark for Geospatial Question Answering

### Abstract

Recent advances in Large Language Models (LLMs) have led to dramatic improvements in question answering (QA). 
To address the challenge of evaluating QA systems, standardized benchmarks have been introduced. 
This work focuses on the problem of geospatial QA, where a large collection of geospatial data is available in the form of a spatial database or other forms.
There is limited work on creating or evaluating geospatial QA systems. This work has various limitations including a small number of questions, reliance on a knowledge graph, limited geospatial operators, and no complex reasoning. 
We present GS-QA, an extensible geospatial QA benchmark with 2800 question-answer pairs on top of Open Street Maps (OSM) and Wikipedia data, covering various spatial objects, predicates, and answer types. 
A key feature of GS-QA is that some of the questions require combining information from multiple sources, e.g.,  geospatial information from OSM and other information from Wikipedia. 
GS-QA includes a comprehensive evaluation methodology that combines text-based QA measures with geospatial-specific measures. 
We implemented various LLM-based geospatial QA baselines, using combinations of LLMs, retrieval, and structured querying. 
Our results show that existing solutions have very low accuracy, which warrants more research in this direction.

## Features

- **Diverse Question Types**: Includes questions about various geometry objects (points, lines, polygons, etc.)
- **Spatial Predicates**: Tests understanding of spatial relationships (nearest neighbor, distance, intersects, direction, etc.)
- **Comprehensive Coverage**: Designed to evaluate different aspects of geospatial reasoning capabilities
- **Structured Format**: Benchmark data provided in JSON format for easy integration
- **Text2SQL**: All questions have associated SQL queries to retreive the answers given a refernce database.
- **Extensibility**: Benchmark includes associated python scripts to generate more question types and extend the dataset.

## Requirements

- Python 3.8+
- PostgreSQL with PostGIS extension enabled
- Dependencies listed in `requirements.txt`
- Ollama 0.6.1 with LLAMA 3.2 (ID: a80c4f17acd5)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/MajidSas/GS-QA
   cd GS-QA
   ```

2. Install Python dependencies:
   ```
   pip install -r ./generator/requirements.txt
   pip install -r ./baselines/requirements.txt
   ```

3. Set up PostgreSQL with PostGIS:
   - Ensure PostgreSQL is installed on your system
   - Install PostGIS extension:
     ```
     CREATE EXTENSION postgis;
     ```
   - Create a database and configure connection parameters in the config file

4. Install Ollama and get OpenAI API token

Ollama is only required for evaluating the baselines: https://ollama.com

OpenAI API is required to evaluate GPT4o.

The scripts can be easily modified for any model, and are based on LangChain.


### Benchmark Data

The benchmark questions are located in the `benchmark/` directory. The folder contains 28 directories, one for each question type. Inside each directory there are 100 folders, one for each question. Inside each of these folders, there are two JSON files, one for the question itself and one for the answers we obtaiend from our baselines.

The `question.json` files have the following general schema:

- `"question"`: contains the question text.
- `"sql"`: the sql query used to get the answers from our reference database.
- `"answer_type"`: proivdes the answer type for each template.
- `"answers"`: an array of objects, with schema depending on the question type. 
   * For entity name (T1-T11, except T7), the answers are in an attribute that ends with the suffix `"_name"`, e.g. `"poi_name"`.
   * For the first type of multihop questions (T7), it is `"multihop_answer"`, and the type is `"multihop_attribute"`.
   * For the location (T12-T20), the attribute is `"geometry"`, which needs to be geocoded if evaluating on address and not location coordinates.
   * For the direction (T21,T22), the attribute is `"angle"`.
   * For other output types it is under `"count"`, `"distance"`, `"length"`, `"area"`.
   For non-aggregate answers, the entire record is stored in addition to the answer attribute.
- `"question_entities"`: stores the objects that were used to create the question such as the anchoring points or the region. This can be helpful for more advanced evaluation.
- `"id"`: a unique identifier for each question in the benchmark.
- `"type"`: a unique identifier for each template, which includes the names of predicates and output type.

## Usage

### Question Generation

To run the question generation script, first we need to setup the database.

First, download our data from: https://drive.google.com/drive/folders/1pz895-lpGAaNJXz2mzjnB7SAgWwD0Uag?usp=share_link

We obtained this data using this tool: https://bitbucket.org/bdlabucr/osmx/src/master/
With the source: https://www.geofabrik.de/data/download.html
We only downloaded the data of the US. You can obtain data over a different region or a more recent copy.

The data must be in geojson format, with the following folder structure:
```
- osm_extract:
-- lakes
-- parks
-- pois
-- postal_codes 
-- roads
```

Once the data is ready, and a database in PostgresQL is created, we can use the folloing scripts to create the tables and insert the data into the database:
- "pois_processor.py"
- "regions_processor.py"
- "roads_processor.py"
- "parks_processor.py"
- "lakes_processor.py"

In all of these scripts, you will need to update your database connection information. Modify the connection lines:
```
conn = psycopg2.connect(
      host = 'localhost',
      dbname = 'database_name',
      user = 'postgres',
      password = 'postgres',
      port = 5432
    )
```

Additionally, each of these scripts has an associated json schema which is used to select the relevant attributes and define the table. E.g. `"poi_schema.json"` each attribute in this schema corresponds to one column in the POI table.

After populating the database, you can run `python generator.py`, which will generate 1000 questions for all the templates. You can modify the code to generate a smaller or a larger subset as desired.

To modify the text templates, such as changing the language of the questions, you can modify the texts in the files inside the folder `./generator/templates/*.txt`. There is one file for each template. You can add additional templates by first creating a file with one version of the question in each line. You can then add additional functions inside the generator file to populate this template. Here is an example code that populates one template, with comments:

```python
variable_types = [
    ('[1]', 'poi'),
    ('[2]', 'distance'),
    ('[3]', 'poi'),
] # determines for which cateogory of values these parameters are populated all text version of this template must contain all three parameters

template_tokens = [
    ('[1_type]', '[1] main_category'),
    ('[1]', '[1] sub_category', 'append_an'), # for this question [1] is preprended with `a` or `an` to make the question grammatically correct
    ('[2]', '[2] distance'), 
    ('[2_text]', '[2] text'), # we differntiate between the distance number in meters and the text which is appended by the unit in kilometers
    ('[3]', '[3] display_name'), # this populates it with the display_name of the poi
    ('[3_wkt]', '[3] geo_wkt'), # this one populates the geometry of the poi in the SQL template
] # 


template_sql = '''SELECT * FROM pois
WHERE ST_DWithin(pois.geometry, ST_GeomFromText('[3_wkt]',4326)::geography, [2])
AND [1_type] = '[1]';
'''
text_templates =  [l.replace('[2]', '[2_text]').strip() for l in open('templates/range+name.txt','r').readlines()]
answer_type = 'name'
n = N
questions = question_generator(text_templates, variable_types, template_tokens, template_sql, answer_type, verifier, n) # this function will generate the questions
# if new types of entities are used the question_generator function will need to be modified to support them
save(questions, 'range+name.jsonl') # saves the questiosn one in JSON format one per line
```

Finally, the notebook `./generator/question_selector.ipynb` is used to filter the questions based on our criteria and select 100 questions from each template. Further, we manually evaluated this final set of questions that are included in the benchmark.

### Baselines and evaluation

The folder `./baselines/` contains multiple scripts to run and evaluate the baselines. See `./baselines/run_baselines.sh` for usage. The baselines require OLLAMA to be setup and also OpenAI and Anthropic API keys to evaluate their models. 

The questions can be read using the following code:
```python
from glob import glob
import json
files = glob('../benchmark/**/question.json', recursive=True)
questions = []
for path in files:
    with open(path, 'r') as file:
      question = json.loads(file.read())
      questions.append(question)
```

Running the remainder of the script will generate two files one for full text evaluation and one for parsed evaluation. The file `./baselines/evaluate.py` contains the evaluation functions that we proposed.

## Citation

```
[Paper under review will be added later]
```


## Contact

For questions or feedback, please open an issue or contact msaee007@ucr.edu
