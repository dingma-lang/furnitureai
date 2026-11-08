"""Strict parser for the acquired V7 dictionary, which includes a CSV header."""
import csv
import re
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/phase5-v1'))
import detection_prep as d


def read_dictionary(path):
    with open(path,newline='') as stream:
        reader=csv.reader(stream)
        if next(reader,None)!=['LabelName','DisplayName']:raise ValueError('Unexpected V7 dictionary header')
        names={}
        for row in reader:
            if len(row)!=2 or not re.fullmatch('/m/[a-zA-Z0-9_]+',row[0]) or not row[1] or row[0] in names:raise ValueError('Invalid/duplicate dictionary class')
            names[row[0]]=row[1]
    return names


def class_mapping(path):
    names=read_dictionary(path)
    return dict(mapping={mid:d.NAMES[name] for mid,name in names.items() if name in d.NAMES},
                missing_exact_names=sorted(set(d.NAMES)-set(names.values())),dictionary_sha256=d.sha(path),
                dictionary_classes=len(names),dictionary_header=['LabelName','DisplayName'],
                policy='exact display names only; no automatic hierarchy propagation')
