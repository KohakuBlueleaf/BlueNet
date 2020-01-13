base = open('txt.txt',mode='r',encoding="utf-8")
temp = base.read()
txt = temp.replace('.',' .').replace('"',' " ').replace('(',' ( ').replace(')',' ) ').replace(':',' : ').replace(',',' ,').replace(';',' ; ').replace('?',' ? ').replace('!',' ! ').replace('--',' -- ').replace('_',' _ ').replace("'ll"," 'll ").replace("'s"," 's ").replace("'re"," 're ")
base.close()