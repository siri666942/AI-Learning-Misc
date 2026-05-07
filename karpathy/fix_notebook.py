import json
with open('engine.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'from graphviz import Digraph' in ''.join(cell['source']):
        existing = ''.join(cell['source'])
        if 'Graphviz' not in existing:
            new_source = [
                'import os\n',
                'os.environ["PATH"] += os.pathsep + r"C:\\Program Files\\Graphviz\\bin"\n',
                '\n',
            ]
            new_source.extend(cell['source'])
            cell['source'] = new_source
        break

with open('engine.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Done')
