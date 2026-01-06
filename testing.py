# import os
# import camera
# from camera import stop_flag

# print(os.cpu_count())

# print(camera.stop_flag)
# print(stop_flag)

# stop_flag = True

# print(camera.stop_flag)
# print(stop_flag)

# stop_flag = False
# camera.stop_flag = True

# print(camera.stop_flag)
# print(stop_flag)

# data = {'a': (10, 3), 'b': (5, 10), 'c': (15, 20), 'd': (2, 30)}
# print(data)
# (v1, v2) = list(data.values())[2]
# print(v1)
# print(v2)
# threshold = 6

# # Keep items where the value is >= threshold
# filtered_data = {k: (v1, v2) for k, (v1, v2) in data.items() if v2 >= threshold}

# print(len(filtered_data))
# print(filtered_data.items())

# my_list = [1, 3, 5, 7, 8]
# print(len(my_list) > 0)
# print(filtered_data.copy())

# my_actual_list = list(filtered_data.values())
# print(my_actual_list)
# (list1, list2) = zip(*my_actual_list)
# print(list1[0])
# print(list2)

# print('loop:')
# for i, (name, frame) in enumerate(my_actual_list):
#     print(i)
#     print(name)
#     print(frame)

# my_list.extend([9, 10, 100])
# print(my_list.copy())
# print(my_list.pop(0))
# print(my_list)

# Output: {'a': 10, 'c': 15}


# temp = (0, 1, 2)
# print(temp[2])
# max_value = 5
# for i in range(0, max_value):
#     print(i)
#     print(max_value)
#     max_value -= 1


# import cv2
# import time

# caps = {}
# camera_in_use = 2
# while camera_in_use >= 0:
#     cap = cv2.VideoCapture(camera_in_use)
#     if cap.isOpened():
#         caps[cap] = (None, time.time())
#     else:
#         print("Error: Could not open camera " + str(camera_in_use))

#     camera_in_use -= 1

# if len(caps) == 0:
#     exit()

# stop_flag = False
# while not stop_flag:
#     for cap, value in caps.items():
#         ret, frame = cap.read()
#         if ret:
#             caps[cap] = (frame.copy(), time.time())
#         else:
#             time.sleep(0.001)

#     # for i, (cap, (latest_frame, ts)) in enumerate(caps.items()):
#     #     cv2.namedWindow("window_name " + str(i), cv2.WINDOW_NORMAL)
#     #     cv2.imshow("window_name " + str(i), latest_frame)
    
#     for cap, value in caps.items():
#         # cv2.namedWindow("window_name " + str(i), cv2.WINDOW_NORMAL)
#         # cv2.imshow("window_name " + str(i), latest_frame)
#         print(cap)
#     if cv2.waitKey(1) & 0xFF == 27:
#         stop_flag = True

# for cap, value in caps.items():
#     cap.release()

text = 'happyt'
print(text[-2])