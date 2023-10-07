import os
# Function to rename multiple files
def main():
	path= os.path.dirname(os.path.abspath(__file__)) + "/organ"
	for filename in os.listdir(path):
		my_source = path + "/" + filename
		os.rename(my_source, path +  filename[1:])

if __name__ == '__main__':
	main()
