import re 

database_file = open('../data/ohsumed.88-91')
data = database_file.read()

# Can this be optimized? 
pattern = re.compile("\.I.*\n\.U\n.*\n\.S\n.*\n\.M\n.*\n\.T\n.*\n\.P\n.*\n\.W\n.*\n\.A\n.*\n")



