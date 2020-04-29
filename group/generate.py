
import random
GROUP_SIZE=5
students_num_list = list(range(1,51))
random.shuffle(students_num_list)

group_nums=len(students_num_list)//GROUP_SIZE + 1
for group_num in range(group_nums):
    students = students_num_list[:GROUP_SIZE]
    students_num_list = students_num_list[GROUP_SIZE:]
    for student in students:
        print ('\t'.join(map(str, [group_num, student])))

