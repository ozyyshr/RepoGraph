# retrieve code graph scripts

import pickle
import sys
import json

def main(func_name):
    with open('/graph.pkl', 'rb') as f:
        G = pickle.load(f)
    with open('/tags.json', 'r') as f:
        tags = f.readlines()
    tags = [json.loads(tag) for tag in tags]

    try:
        successors = list(G.successors(func_name))
        predecessors = list(G.predecessors(func_name))
        tags2names = {tag['name']: tag for tag in tags}
        returned_files = []
        for item in successors+[func_name]+predecessors:
            if 'test' in tags2names[item]['fname']:
                continue
            returned_files.append({
                "fname": tags2names[item]['fname'],
                'line': tags2names[item]['line'],
                'name': tags2names[item]['name'],
                'kind': tags2names[item]['kind'],
                'category': tags2names[item]['category'],
                'info': tags2names[item]['info'],
            })
        print(returned_files)
    except:
        print("None")

if __name__ == '__main__':
    func_name = sys.argv[1]
    main(func_name)