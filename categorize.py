import re
from collections import defaultdict

# 파일 읽기
with open('/home/wok1909/workplace/RAG/hnsw/cuhnsw/dump_searchgraph.txt', 'r') as f:
    lines = f.readlines()

# Query별로 라인을 저장할 dict
query_dict = defaultdict(list)

# 정규표현식으로 [Query #] 추출
pattern = re.compile(r'\[Query\s*(\d+)\]')

for line in lines:
    line = line.strip()
    match = pattern.search(line)
    if match:
        query_num = int(match.group(1))
        query_dict[query_num].append(line + '\n' if not line.endswith('\n') else line)

# Query 번호 순서대로 출력
with open('/home/wok1909/workplace/RAG/hnsw/cuhnsw/searchgraph.txt', 'w') as f:
    for query_num in sorted(query_dict.keys()):
        for line in query_dict[query_num]:
            f.write(line)
