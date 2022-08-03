import sys
from rich.progress import track
import sqlparse

args = sys.argv

file_path = args[1]
# head_sz = 5
with open(file_path, 'r') as json_file:
    json_list = list(json_file)
    result = []
    for row in track(json_list, description="Processing..."):
        # if head_sz > 0:
        items = sqlparse.parse(row)
        if "SELECT " not in row:
            continue
        elif "\\u" in row or "$" in row or "double precision" in row:
            continue

        result.append(items)
        # head_sz -= 1

    res_file_path = f"{file_path.split('.')[0]}_cleaned.{file_path.split('.')[1]}"
    with open(res_file_path, 'w') as res_file:
        res_file.write(''.join('%s' % x for x in result))

print("Completed.")
