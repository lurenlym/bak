import sys

if __name__ == '__main__':
	file_list = sys.argv[1]
	label_list = sys.argv[2]
	f = open(file_list, 'r')
	t = open(label_list, 'w')

	label = {'BQ':0, 'JGBTH':1, 'JGTH':2, 'NORMAL':3, 'YWB':4, 'ZC':5, 'ZD':6}

	index = 0
	line = f.readline().strip()
	while line:
		img_class = line.split('/')[6]
		line_label = line + ' ' + str(label[img_class])
		t.write(line_label + '\n')

		index += 1
		line = f.readline().strip()

	print 'image num:', index
	f.close()
	t.close()
