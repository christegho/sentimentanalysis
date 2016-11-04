documents = {}
for root, dirs, filenames in os.walk(indir):
    for filename in filenames:
		file = open(indir+'\\'+filename,'r')
		text = file.read().lower()    
		file.close()
		words = re.findall(r"[\w']+|[.,!?;]*", text)
		documents[filename]=filter(None, words)

os.path.abspath('')