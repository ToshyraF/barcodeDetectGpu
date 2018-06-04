import os
import time

p = os.popen('ls ./image')  
barcode = []
for line in p.readlines():
	print line
	z = os.popen('zbarimg ./image/'+line)
	barcode.append(z.read())
	# print z.read()
	# time.sleep(3)
# print(p.read())  
print barcode