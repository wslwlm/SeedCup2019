#!/usr/bin/env python
# coding=utf-8
import csv
import datetime

source_file = 'data/SeedCup_final_test.csv'

sub = open('submit.txt', 'w')
pay_time_lines = open('test_output/model1.txt', 'r').readlines()
preselling_time_lines = open('test_output/model2.txt', 'r').readlines()

counts = 0
counts1 = 0
i_range = 0
with open(source_file, 'r') as f:
    i_range  = len(f.readlines()) - 1

with open(source_file, 'r') as f:
    reader = csv.reader(f)
    header_row = next(reader)[0]

    for i in range(i_range):
        data = next(reader)[0].split('\t')

        temp_payed_time = datetime.datetime.strptime(data[4], "%Y-%m-%d %H:%M:%S")
        if data[9] != '':
            temp_preselling_shipped_time = datetime.datetime.strptime(data[9], "%Y-%m-%d %H:%M:%S")
            # 对一部分的preselling_shipped_time使用model2的预测结果
            if (temp_preselling_shipped_time - temp_payed_time).days == 4 and counts < 7000:
                # counts += 1
                # print('counts: ', counts)
                # print('preselling_time: {}, pred: {}'.format(temp_preselling_shipped_time, preselling_time_lines[i]))
                # print('payed_time: {}, pred: {}'.format(temp_payed_time, pay_time_lines[i]))
                sub.write(preselling_time_lines[i])
            elif (temp_preselling_shipped_time - temp_payed_time).days > 4:
                # counts1 += 1
                # print('counts1: ', counts1)
                #print('preselling_time: {}, pred: {}'.format(temp_preselling_shipped_time, preselling_time_lines[i]))
                #print('payed_time: {}, pred: {}'.format(temp_payed_time, pay_time_lines[i]))
                sub.write(preselling_time_lines[i])
            else:
                sub.write(pay_time_lines[i])
        else:
            sub.write(pay_time_lines[i])
    print('counts: ', counts)

