DATASET = "movie"
mapping_last_line = open(DATASET + '/item_index2entity_id.txt', encoding='utf-8').readlines()[-1]
mapping_last_array = mapping_last_line.strip().split('\t')
max_item_id = int(mapping_last_array[0])

writer = open(DATASET + '/kg_filtered.txt', 'w', encoding='utf-8')

for line in open(DATASET + '/kg.txt', encoding='utf-8'):
    array = line.strip().split('\t')
    item_old_id = int(array[0])
    if item_old_id >= 1 and item_old_id <= max_item_id:
        writer.write('%s\t%s\t%s\n' % (array[0], array[1], array[2]))

writer.close()

print(max_item_id)
