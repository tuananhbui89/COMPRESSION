import numpy as np 

EOF = 12
MAXLENGTH = 10
PRECISION = 32
WHOLE = 2**PRECISION
HALF = WHOLE/2
QUARTER = WHOLE/4
X3QUARTER = 3*QUARTER

def arithemic_encode_inf(probs, bins, sequence):
	outbins = []

	a = 0.
	b = WHOLE
	s = 0 
	probs_cum, probs_d = get_cumulative_probs(probs)
	probs_cum = create_dict(bins, probs_cum)
	probs_d = create_dict(bins, probs_d)

	for symbol in sequence:
		w = b - a 
		b = a + w * probs_d[symbol]
		a = a + w * probs_cum[symbol]

	while b<HALF or a>HALF:
		if b<HALF:
			outbins = emit(outbins, '0')
			a = 2*a 
			b = 2*b 

		elif a>HALF:
			outbins = emit(outbins, '1')
			a = 2*(a-HALF)
			b = 2*(b-HALF) 

	while a>QUARTER and b<X3QUARTER:
		s = s + 1
		a = 2*(a-QUARTER)
		b = 2*(b-QUARTER)

	print(s, a, b)
	s = s + 1 
	if a <= QUARTER:
		outbins = emit(outbins, '0'+'1'*s)
	else:
		outbins = emit(outbins, '1'+'0'*s) 

	return outbins

def arithemic_encode(probs, bins, sequence):
	outbins = []

	a = 0
	b = WHOLE
	s = 0 
	probs_cum, probs_d = get_cumulative_probs(probs)
	probs_cum = create_dict(bins, probs_cum)
	probs_d = create_dict(bins, probs_d)
	R = np.sum(probs)

	idx = 0
	for symbol in sequence:
		idx += 1
		w = b - a 
		b = a + np.floor(w * probs_d[symbol] / R)
		a = a + np.floor(w * probs_cum[symbol] / R)
		
		assert(b-a <= HALF)
		
		while b<=HALF or a>=HALF:
			if b<=HALF:
				outbins = emit(outbins, '0')
				a = 2*a 
				b = 2*b 
				if s > 0:
					outbins = emit(outbins, '1'*s)
					s = 0
			elif a>=HALF:
				outbins = emit(outbins, '1')
				a = 2*(a-HALF)
				b = 2*(b-HALF) 
				if s > 0:
					outbins = emit(outbins, '0'*s)
					s = 0

		while a>QUARTER and b<X3QUARTER:
			s = s + 1
			a = 2*(a-QUARTER)
			b = 2*(b-QUARTER)
		
		assert(a != QUARTER)
		assert(b != X3QUARTER)
		# assert(a != 0)
		assert(b <= WHOLE)
		assert(b >= HALF)
		if symbol == EOF:
			assert(idx == len(sequence)) 
			break

	print(s, a/WHOLE, b/WHOLE)

	s = s + 1 
	if a < QUARTER:
		outbins = emit(outbins, '0'+'1'*s)
	elif b > X3QUARTER:
		outbins = emit(outbins, '1'+'0'*s) 
	else:
		print('Wrong at finish', s, a, b)

	return outbins

def arithemic_decode_inf(probs, bins, outbins):
	sequence = []
	a = 0
	b = WHOLE
	probs_cum, probs_d = get_cumulative_probs(probs)
	probs_cum = create_dict(bins, probs_cum)
	probs_d = create_dict(bins, probs_d)
	z = get_prob_range(outbins)
	while 1: 
		for symbol in bins: 
			w = b - a 
			b0 = a + w * probs_d[symbol]
			a0 = a + w * probs_cum[symbol]
			if z >= a0 and z < b0: 
				sequence = emit(sequence, [symbol])
				a = a0
				b = b0
				if symbol == EOF:
					return sequence

def arithemic_decode(probs, bins, outbins):
	sequence = []
	a = 0
	b = WHOLE
	z = 0
	i = 0
	probs_cum, probs_d = get_cumulative_probs(probs)
	probs_cum = create_dict(bins, probs_cum)
	probs_d = create_dict(bins, probs_d)
	R = np.sum(probs)

	z = get_prob_range(outbins, method='FINITE')

	while 1: 
		for symbol in bins: 
			w = b - a 
			b0 = a + np.floor(w * probs_d[symbol] / R)
			a0 = a + np.floor(w * probs_cum[symbol] / R)

			assert(b0-a0 <= HALF)

			if z >= a0 and z < b0: 
				sequence = emit(sequence, [symbol])
				a = a0
				b = b0
				if symbol == EOF:
					return sequence
		
		while b <= HALF or a >= HALF:
			if b <= HALF: 
				a = 2*a 
				b = 2*b 
				z = 2*z 
			elif a >= HALF:
				a = 2*(a-HALF)
				b = 2*(b-HALF)
				z = 2*(z-HALF)
			if i < len(outbins) and outbins[i] == 1:
				z = z + 1 
			i = i + 1


		while a > QUARTER and b < X3QUARTER: 
			a = 2*(a-QUARTER)
			b = 2*(b-QUARTER)
			z = 2*(z-QUARTER)
			if i < len(outbins) and outbins[i] == 1:
				z = z + 1
			i = i + 1 

		# assert(i < len(outbins))
		assert(a != QUARTER)
		assert(b != X3QUARTER)
		# assert(a != 0)
		assert(b <= WHOLE)

def get_prob_range(outbins, method='INFINITE'):
	z = 0
	i = 0
	if method == 'INFINITE':
		while i <= MAXLENGTH and i < len(outbins):
			z = z + outbins[i] * 2.**(-(i+1))
			i += 1
		return z
	elif method == 'FINITE':
		# while i <= PRECISION and i < len(outbins):
		while i < len(outbins):	
			if outbins[i] == 1: 
				z = z + 2.**(PRECISION - i - 1)
			i += 1
		return z 	

def emit(outbins, symbols):
	# assert('0' in symbols or '1' in symbols)
	# print('e', symbols)
	if len(symbols) == 1:
		outbins.extend([int(symbols[0])])
		return outbins
	else:
		for symbol in symbols:
			outbins.extend([int(symbol)])
		return outbins

def get_cumulative_probs(probs):
	probs_cum = np.zeros(np.shape(probs))
	probs_d = np.zeros(np.shape(probs))
	probs_d[0] = probs[0]

	for idx in range(1,len(probs),1):
		probs_cum[idx] = probs_cum[idx-1] + probs[idx-1]
		probs_d[idx] = probs_cum[idx] + probs[idx]

	assert(probs_d[-1] == np.sum(probs))
	assert(probs_cum[-1] == (np.sum(probs) - probs[-1]))
	return probs_cum, probs_d

def create_dict(key_list, value_list):
	assert(len(key_list) == len(value_list))
	outdict = dict()
	
	for idx in range(len(key_list)):
		# print(key_list[idx], value_list[idx])
		outdict[key_list[idx]] = value_list[idx]

	return outdict 

def count_value(listvalue, value): 
	return np.sum(np.array(listvalue) == value)

def create_probs_dict(sequence):
	bins = list(set(sequence))
	bins = np.sort(bins)
	probs = []
	for digit in bins: 
		p = count_value(sequence, digit)
		probs.append(p)
	probs = np.array(probs)
	probs = probs / np.float(len(sequence))
	return probs, bins

def read_all_lines(fname):
    with open(fname) as f: 
        content = f.readlines()
    content = [x.strip() for x in content]
    return content

def get_data_from_line(line):
	temp = line.split(' ')
	data = []
	for x in temp:
		if x != '':
			data.extend([int(x)])
	return data

def test1():
	counter = 0
	seqs = read_all_lines('sequence_example.txt')
	targets = read_all_lines('encoded_code_example.txt')
	probs = [5, 50, 40, 5, 5, 50, 40, 5, 5, 50, 40, 5]
	bins = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, EOF]

	for idx in range(len(seqs)):
		sequence = get_data_from_line(seqs[idx])
		target = get_data_from_line(targets[idx])
		outbins = arithemic_encode(probs, bins, sequence)
		sequence_ = arithemic_decode(probs, bins, outbins)
		mse = np.sum(sequence != sequence_)
		mse2 = np.sum(target != outbins)
		# print(outbins)
		if mse > 0:
			counter += 1
			print(mse, mse2, len(outbins), len(sequence))
	print('Failure %d/%d'%(counter, len(seqs)))

def test2():
	sequence = np.random.randint(low=1, high=4, size=(5,))
	sequence = list(sequence)
	sequence.extend([EOF])
	# sequence = [2, 3, 2, EOF]
	probs = [0.05, 0.05, 0.5, 0.4]
	bins = [EOF, 1, 2, 3]
	outbins = arithemic_encode_inf(probs, bins, sequence)
	sequence_ = arithemic_decode_inf(probs, bins, outbins)
	mse = np.sum(sequence != sequence_)
	print(mse, len(outbins), len(sequence))
	print(outbins)
	print(sequence)
	print(sequence_)

def test3():
	sequence = [2, 3, 2, EOF]
	probs = [5, 5, 50, 40]
	bins = [EOF, 1, 2, 3]
	outbins = arithemic_encode(probs, bins, sequence)
	sequence_ = arithemic_decode(probs, bins, outbins)
	mse = np.sum(sequence != sequence_)
	print(mse, len(outbins), len(sequence))
	print(outbins)
	print(sequence)
	print(sequence_)

if __name__ == '__main__':
	test1()

