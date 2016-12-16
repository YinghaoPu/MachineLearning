from chapter3 import trees

my_data_set, labels = trees.create_data_set()
result = trees.calc_shannon_ent(my_data_set)
print result