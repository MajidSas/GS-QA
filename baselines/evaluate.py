import re
from num2words import num2words
from geopy import distance
from shapely import from_wkt
import time
import logging
import os
import json
import nltk, string
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

import transformers
transformers.tokenization_utils.logger.setLevel(logging.ERROR)
transformers.configuration_utils.logger.setLevel(logging.ERROR)
transformers.modeling_utils.logger.setLevel(logging.ERROR)

from functools import lru_cache


def get_angle_desc(angle):
    if 0.0 <= angle < 22.5 or 337.5 < angle <= 360:
        return 'north'
    if 22.5 <= angle <= 67.5:
        return 'northeast'
    if 67.5 < angle < 112.5:
        return 'east'
    if 112.2 <= angle <= 157.5:
        return 'southeast'
    if 157.5 < angle < 202.5:
        return 'south'
    if 202.5 <= angle <= 247.5:
        return 'southwest'
    if 247.5 < angle < 292.5:
        return 'west'
    if 292.5 <= angle <= 337.5:
        return 'northwest'
    

# def evaluate_entity_names(scorer, predictions, answers):
#     P, R, F1 = scorer.score(predictions, answers)
#     return P, R, F1
@lru_cache(maxsize=1024)
def normalize_text(s):
    tokens = word_tokenize(s)
    tokens = [word.lower() for word in tokens if (word not in string.punctuation and word.lower() not in stop_words)]
    return tokens

def evaluate_entity_names(prediction, truth):
    pred_tokens = normalize_text(prediction)
    truth_tokens = normalize_text(truth)
    # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return 0, 0, 0
    
    common_tokens = set(pred_tokens) & set(truth_tokens)
    
    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0, 0, 0 #float('nan')
    
    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)
    f1 = 2 * (prec * rec) / (prec + rec)
    return prec, rec, f1
def evaluate_location(geod, pred_locs, true_locs):
    distances = []
    for i in range(len(pred_locs)):
        lon1, lat1 = true_locs[i]['lon'], true_locs[i]['lat']
        lon2, lat2 = pred_locs[i]['lon'], pred_locs[i]['lat']
        az12,az21,dist = geod.inv(lon1,lat1,lon2,lat2)
        distances.append(dist)
    return distances

@lru_cache(maxsize=None)
def geocode_address(geocoder, address):
    return geocoder.geocode(address)
    
def get_location_by_address(geocoder, address, fails=0):
    """This function returns a location as raw from an address
    will repeat until success"""
    # return {'lon': 0, 'lat': 0}
    cache_path = './address_cache/%s.json' % '_'.join(normalize_text(address)).replace('/', ':')
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as file:
            text = file.read()
            if text == 'None':
                return None
            else:
                return json.loads(text)
    if fails == 5:
        with open(cache_path, 'w') as file:
            file.write('None')
        return None
    time.sleep(2)
    try:
        obj = geocode_address(geocoder, address)
        if obj == None:
            with open(cache_path, 'w') as file:
                file.write('None')
        else:
            with open(cache_path, 'w') as file:
                file.write(json.dumps(obj.raw, indent=2))
        return None if obj == None else obj.raw
    except Exception as e:
        print(e)
        time.sleep(2)
        return get_location_by_address(geocoder, address, fails+1)

def evaluate_address(predictions, answers):
    pred_locs = []
    for a in predictions:
        pred_locs.append(get_location_by_address(a))
    true_locs = []
    for a in answers:
        true_locs.append(get_location_by_address(a))
    return evaluate_location(pred_locs, true_locs)

def evaluate_angle(predictions, answers):
    scores = []
    for i in range(len(predictions)):
        score = abs(predictions[i]-answers[i])
        if score > 180:
            score = 360-score
        scores.append(score/180.0)
    return scores

def evaluate_measurement(pred, true):
    return abs(true-pred)/true

def get_osm_value(json_obj, value_label):
    if value_label == 'address':
        addr = []
        keys = ['addr_house_number', 'addr_street', 'addr_city', 'addr_state', 'addr_postcode']
        for k in keys:
            if k in json_obj:
                addr.append(json_obj[k])
        return ', '.join(addr)
    if value_label == 'location':
        point = from_wkt(json_obj['geometry']).centroid
        return {'lon': point.x, 'lat': point.y}
    if value_label == 'angle_description':
        angle = json_obj['angle']
        if 0.0 <= angle < 22.5 or 337.5 < angle <= 360:
            return 'north'
        if 22.5 <= angle <= 67.5:
            return 'northeast'
        if 67.5 < angle < 112.5:
            return 'east'
        if 112.2 <= angle <= 157.5:
            return 'southeast'
        if 157.5 < angle < 202.5:
            return 'south'
        if 202.5 <= angle <= 247.5:
            return 'southwest'
        if 247.5 < angle < 292.5:
            return 'west'
        if 292.5 <= angle <= 337.5:
            return 'northwest'
    if value_label == 'name':
        keys = ['poi_name', 'lake_name', 'park_name', 'road_name', 'region_name']
        for k in keys:
            if k in json_obj:
                return json_obj[k]
        if 'wikipedia' in json_obj:
            return json_obj['wikipedia'][3:]
    return json_obj.get(value_label, None)