import json
import re

with open('/shared/data3/siruo2/SWE-agent/trajectories/siruo2/gpt4__SWE-bench_Lite__default__t-0.00__p-0.95__c-3.00__install-1/pvlib__pvlib-python-1072.traj', 'r') as f:
    data = json.load(f)

pattern = r"'fname': '([^']+)',\s"
files = []

for item in data['codegraph']:
    if 'fname' in item['codegraph_cxt']:
        for f in re.findall(pattern, item['codegraph_cxt']):
            if 'test' not in f:
                files.append(f)

print(len(list(set(files))))
