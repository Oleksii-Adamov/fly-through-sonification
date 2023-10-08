import os
# Function to rename multiple files
def main():
	path= os.path.dirname(os.path.abspath(__file__)) + "/hang"
	for filename in os.listdir(path):
		my_source = path + "/" + filename
		if filename[-6:-4] in ["02", "03", "04", "05", "06", "07"]:
			os.remove(my_source)
		else:
			os.rename(my_source, path + "/_" + filename[:-7] + ".wav")

if __name__ == '__main__':
	main()
