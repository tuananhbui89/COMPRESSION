import numpy as np 
import random 

def est_entropy(probs): 
	res = 0 
	nbz = 0
	for prob in probs: 
		if prob == 0:
			nbz += 1
		else: 
			res = res - prob * np.log2(prob)
	return res, nbz 

def count_value(listvalue, value): 
	return np.sum(np.array(listvalue) == value)

def get_probs(x):
	minrange = int(np.min(x))
	maxrange = int(np.max(x))
	x_flat = np.reshape(x, -1)
	# nbbins = maxrange - minrange + 1
	# bins = np.array(range(minrange, maxrange+1, 1))
	bins = list(set(x_flat))
	bins = [int(x) for x in bins]
	freqs = []
	for digit in bins:
		freq_ = count_value(x_flat, digit)
		freqs.extend([freq_])
	freqs = np.array(freqs)
	probs = freqs / np.float(len(x_flat))
	entropy, nbz = est_entropy(probs)
	return freqs, probs, bins, entropy, nbz 

# Standardize bins from range [-a, b] to range[0, maxrange] with maxrange = 2**(log2(maxabs(a, b)))
def standardize_bins(freqs, bins, gain=1, method='freq'):
	assert(len(freqs) == len(bins))
	assert(all(freqs) != 0)
	# assert(np.min(bins) < 0)
	print('Range of bins before standardize: %d %d'%(np.min(bins), np.max(bins)))

	if np.min(bins) != np.max(bins):
		if np.min(bins) < 0:
			nbbits = int(np.ceil(np.log2(np.max(bins) - np.min(bins))))
			maxrange = np.max(bins) - np.min(bins) + 1
			bias = -np.min(bins)

		else:
			nbbits = int(np.ceil(np.log2(np.max(bins) - np.min(bins))))
			maxrange = np.max(bins) - np.min(bins) + 1
			bias = -np.min(bins)

		# newbins = (np.array(bins) + bias).tolist()
		newbins = list(range(0, maxrange))

		if method == 'freq':
			newfreqs = (np.ones(shape=(maxrange,))).tolist()
			
			idx = 0
			for bin_ in bins:
				newfreqs[bin_ + bias] = freqs[idx]*gain
				idx += 1

		elif method == 'prob':
			# assert(np.sum(freqs) == 1)
			print('Total prob input',np.sum(freqs))
			delta = np.min(freqs)/gain/10.
			
			newfreqs = (delta/np.float(maxrange)*np.ones(shape=(maxrange,))).tolist()
			
			idx = 0
			for bin_ in bins:
				newfreqs[bin_ + bias] = freqs[idx] + delta/np.float(maxrange) - delta/np.float(len(bins))
				idx += 1

			print('Total prob output after standardize',np.sum(newfreqs))
			# assert(np.sum(newfreqs) == 1)

		if method == 'freq':
			newfreqs = [int(x) for x in newfreqs]

	else:
		newfreqs = freqs
		newbins = bins
		maxrange = 1
		bias = -np.min(bins)

	print('Range of bins after standardize: %d %d'%(np.min(newbins), np.max(newbins)))

	return newfreqs, newbins, maxrange, bias

def get_cum_freq(x):
	print('Start get prob')
	_, probs, bins, _, _ = get_probs(x)
	print('Start standardize_bins')
	probs, bins, maxrange, bias = standardize_bins(probs, bins, method='prob', gain=1)
	print('Start prob_to_cum_freq')
	cum_freq = my_prob_to_cum_freq(probs)

	return cum_freq, bias 


# maxrange=256, symbol has range [0, 255]
# size=2, value will map to two symbol: x0*maxrange + x1 
# value cannot exceed limit value 
def map_decimal2symbol(value, size=2, maxrange=256):
	limit_value = 200000 
	# assert(value >= 0 and value < limit_value)
	assert(value >= 0)

	symbols = []
	for s in range(size):
		scale = maxrange**(size-s-1)
		temp = value // scale
		value = value - temp * scale
		symbols.extend([temp])

	assert(len(symbols) == size)

	return symbols

def map_symbol2decimal(symbols, maxrange=256):
	value = 0
	size = len(symbols)

	for s in range(size):
		symbol = symbols[s]
		scale = maxrange**(size-s-1)
		value = value + symbol * scale

	return int(value)

def my_prob_to_cum_freq(prob, resolution=2**17):
	from range_coder import prob_to_cum_freq, cum_freq_to_prob

	assert(all(prob) != 0)
	return prob_to_cum_freq(prob, resolution=resolution)

def my_cum_freb_to_prob(cum_freq):
	from range_coder import prob_to_cum_freq, cum_freq_to_prob

	return cum_freq_to_prob(cum_freq)
	
# header in syntax: 
# header_size*2 + symbol_size*1 + { [nb_pathes, pH, pW, pC, iH, iW]*s + data_size*s }
def add_header(data, header, maxrange=256):
	
	h = header

	h.extend([len(data)])

	symbol_size = get_size(np.max(h), maxrange=maxrange)

	header_size = len(h)*symbol_size 

	# use fixed 2 symbol size for header size
	assert(header_size < maxrange**2)

	# add header_size to data sequence
	header_size_code = map_decimal2symbol(header_size, size=2, maxrange=maxrange)
	d = header_size_code
	print('Add: header_size %d'%(header_size))

	# add size to data sequence 
	assert(symbol_size < maxrange)
	d.extend([symbol_size])
	print('Add: header_symbol_size %d'%(symbol_size))

	# add header and data size to data sequence
	for value in h:
		symbol = map_decimal2symbol(value, size=symbol_size, maxrange=maxrange)
		d.extend(symbol)

	print('Add: header %s'%(h[:-1]))
	print('Add: len_data %d'%(h[-1]))

	d.extend(data)

	return d 

# get header from a mapped header sequence, convert to decimal 
def get_header(hsymbols, size, maxrange=256): 
	h = []
	assert(len(hsymbols)%size == 0)
	
	for idx in range(len(hsymbols)//size):
		t = hsymbols[idx*size:idx*size+size]
		t = map_symbol2decimal(t, maxrange=maxrange)
		h.extend([t])

	return h 


def range_encode(data, cum_freq, filepath, *argv): 
	from range_coder import RangeEncoder

	assert(all(data) >= 0)

	if len(argv) == 1:
		encoder = argv[0]
		print('[*] Reuse encoder')
	else:
		encoder = RangeEncoder(filepath)
		print('[*] Open encoded %s' % (filepath))

	encoder.encode(data, cum_freq)

	if len(argv) == 1:
		print('Reuse encoder - Please close after finish')
	else:
		print('Encode success')
		encoder.close()


def range_decode(filepath, cum_freq, maxrange, *argv):
    from range_coder import RangeDecoder

    if len(argv) == 1:
        decoder = argv[0]
    else:
        decoder = RangeDecoder(filepath)

    # decode header_size code
    header_size_seq = decoder.decode(2, cum_freq)

    header_size = map_symbol2decimal(header_size_seq, maxrange=maxrange)

    print('Decode: header_size: %d'%(header_size))
    # decode size code 
    size = decoder.decode(1, cum_freq)
    size = size[0]

    print('Decode: header_symbol_size: %d'%(size))
    # decode header 
    headerRec_seq = decoder.decode(header_size, cum_freq)

    headerRec = get_header(headerRec_seq[:-size], size, maxrange=maxrange)

    print('Decode: header: %s'%(headerRec))

    len_data = map_symbol2decimal(headerRec_seq[-size:], maxrange=maxrange)

    print('Decode: len_data: %d'%(len_data))
    # decode data
    dataRec = decoder.decode(len_data, cum_freq)

    if len(argv) == 1:
        print('Reuse decoder - Please close after finish')
    else:
        print('Decode success')
        decoder.close()

    dataRec = [int(x) for x in dataRec]
    
    return dataRec, headerRec


def get_size(value, maxrange=256):
	s = 1
	while 1: 
		scale = maxrange**(s)
		if value >= scale: 
			s += 1 
		else: 
			break

	return s 

# ---------------------------------------------------------------------

def test_decimal2symbol():
	value = 9999
	maxrange = 256
	symbols = map_decimal2symbol(value, size=2, maxrange=maxrange)
	value_ = map_symbol2decimal(symbols, maxrange=maxrange)
	assert(value == value_)

def test_standardize_bins():
	bins = [-3, -2, -1, 0 , 1, 2]

	freqs = [10, 10, 10 , 20 , 20, 30]
	newfreqs, newbins, bias = standardize_bins(freqs, bins, 1, 'freq')
	print(newfreqs, newbins, bias)

	freqs = [0.1, 0.10, 0.10 , 0.20 , 0.20, 0.30]
	newfreqs, newbins, bias = standardize_bins(freqs, bins, 1, 'prob')
	print(newfreqs, newbins, bias)

def test_real_decode():
	# import matplotlib.pyplot as plt 
	import os
	from utils import list_dir

	folder_dir = '/home/zhou/BTA/workspace/compress/temp/Pred_latent/'

	alldirs = list_dir(folder_dir, '.npy')

	filepath = 'Prediction/test.txt'

	header = [25, 960, 1664, 3, 915, 1632]

	cumFreq = (np.load('Prediction/RC_cum_freq.npy')).tolist()
	
	print('Load sucess: cumFreq', type(cumFreq), len(cumFreq))

	bins = (np.load('Prediction/RC_bins.npy')).tolist()

	print('Load sucess: bins', type(bins), len(bins))

	maxrange = (np.load('Prediction/RC_maxrange.npy')).tolist()

	print('Load sucess: maxrange', type(maxrange), maxrange)

	nbbits = np.ceil(np.log2(maxrange))

	all_file_size = []
	all_data_size = []
	for filedir in alldirs:
		header = [25, 960, 1664, 3, 915, 1632]

		# data = [random.randint(0, len(cumFreq) - 2) for _ in range(32000)]	

		data = np.load(filedir)

		alpha = 1/8.*maxrange/2.

		data = np.clip(data, np.round(maxrange/2-alpha), np.round(maxrange/2+alpha))

		data = data.tolist()

		data_ = add_header(data, header, maxrange=maxrange)	

		print('Load sucess: data', type(data), len(data))

		range_encode(data_, cumFreq, filepath)

		dataRec, headerRec = range_decode(filepath, cumFreq, maxrange=maxrange)

		# plt.figure()
		# plt.hist(data, bins=40)
		# plt.show()

		assert(header[:-1] == headerRec)
		
		assert(data == dataRec)

		size = os.stat(filepath).st_size

		all_file_size.extend([size])
		all_data_size.extend([len(data)])

		print('File %s - Data size %d File size %d Rate %0.3f'%(filepath, size, len(data), np.float(size)/len(data)/nbbits*8.))

	totalsize = 4722341.
	
	ave_file_size = np.mean(np.asarray(all_file_size, dtype=np.float))
	ave_data_size = np.mean(np.asarray(all_data_size, dtype=np.float))

	bitrate = ave_file_size / totalsize * 0.15 * len(alldirs)	

	print('Average - maxrange %d, nbbits %d, Data size %0.1f, File size %0.1f, Rate %0.3f'%(maxrange, nbbits, ave_data_size, ave_file_size, ave_file_size/ave_data_size/nbbits*8.))
	print('Requirement - Total %d bytes - Encoded %d bytes - Bitrate %0.3f'%(totalsize, ave_file_size*len(alldirs), bitrate))

def test_decode():
	
	filepath = 'Encode/test.txt'

	header = [10, 20, 30, 40, 50, 100]

	cumFreq = [0, 4, 6, 8, 10, 20, 40, 60, 100, 120, 140, 160, 180]

	nbbins = len(cumFreq) - 2

	data = [random.randint(0, len(cumFreq) - 2) for _ in range(10000)]

	data_ = add_header(data, header, maxrange=nbbins+1)

	range_encode(data_, cumFreq, filepath)

	dataRec, headerRec = range_decode(filepath, cumFreq, maxrange=nbbins+1)

	assert(header[:-1] == headerRec)
	assert(data == dataRec)

def test_prob_to_cum_freq():
    from range_coder import cum_freq_to_prob, prob_to_cum_freq

    """
    Tests whether prob_to_cum_freq produces a table with the expected number
    of entries, number of samples, and that non-zero probabilities are
    represented by non-zero increases in frequency.

    Tests that cum_freq_to_prob is normalized and consistent with prob_to_cum_freq.
    """

    randomState = np.random.RandomState(190)
    resolution = 1024

    p0 = randomState.dirichlet([.1] * 50)
    cumFreq0 = prob_to_cum_freq(p0, resolution)
    p1 = cum_freq_to_prob(cumFreq0)
    cumFreq1 = prob_to_cum_freq(p1, resolution)

    # number of hypothetical samples should correspond to resolution
    assert cumFreq0[-1] == resolution
    assert len(cumFreq0) == len(p0) + 1

    # non-zero probabilities should have non-zero frequencies
    assert np.all(np.diff(cumFreq0)[p0 > 0.] > 0)

    # probabilities should be normalized.
    assert np.isclose(np.sum(p1), 1.)

    # while the probabilities might change, frequencies should not
    assert cumFreq0 == cumFreq1


def test_prob_to_cum_freq_zero_prob():
    from range_coder import cum_freq_to_prob, prob_to_cum_freq

    """
    Tests whether prob_to_cum_freq handles zero probabilities as expected.
    """

    prob1 = [0.5, 0.25, 0.25]
    cumFreq1 = prob_to_cum_freq(prob1, resolution=20)
    print(cumFreq1)

    prob0 = [0.5, 0., 0.25, 0.25, 0., 0.]
    cumFreq0 = prob_to_cum_freq(prob0, resolution=20)
    print(cumFreq0)

    # removing entries corresponding to zeros
    assert [cumFreq0[0]] + [cumFreq0[i + 1] for i, p in enumerate(prob0) if p > 0.] == cumFreq1

def test_get_size():
	import os 
	from utils import list_dir

	folder_dir = '/home/zhou/BTA/workspace/compress/temp/Encoded/64_1.0_0.9_16x16x3_multiscale_dim64_fsize5/'

	alldir = list_dir(folder_dir, '.bin')

	allsize = []
	for filedir in alldir:
		
		size = os.stat(filedir).st_size

		print('File %s - Size %d bytes'%(filedir, size))

		allsize.extend([size])

	totalsize = 4722341.
	
	bitrate = np.mean(allsize) / totalsize * 0.15 * np.max([102, len(allsize)])

	print('Average %0.1f Total %d birate %0.3f'%(np.mean(allsize), np.sum(allsize), bitrate))

def test_encode():
	import os 
	from range_coder import RangeEncoder

	filepath = 'Test/test_encode.bin'

	cum_freq = [0, 900] + range(901, 1000, 1)

	print(len(cum_freq))

	data = 100000*[1] + 5000*[1] + [0] + 5000*[1]

	encoder = RangeEncoder(filepath)
	encoder.encode(data, cum_freq)
	encoder.close()

	encoded_size = os.stat(filepath).st_size

	print('Len data %d encoded_size %d'%(len(data) , encoded_size))
	
def Encode_Decode(x, gain):
	import os 
	from range_coder import RangeEncoder, RangeDecoder, prob_to_cum_freq

	filepath = 'Test/test_entropy.bin'

	x_flat = np.reshape(x, (-1))
	bias = np.min(x_flat)
	x_flat = x_flat - bias 
	x_flat = x_flat.tolist()

	x_quant = [int(xi * gain) for xi in x_flat]

	freqs, probs, bins, entropy, nbz = get_probs(x_quant)
	cum_freq = prob_to_cum_freq(probs)	

	encoder = RangeEncoder(filepath)
	encoder.encode(x_quant, cum_freq)
	encoder.close()

	decoder = RangeDecoder(filepath)
	x_quant_decoded = decoder.decode(len(x_quant), cum_freq)
	decoder.close()

	# Reconstruction 
	x_quant_decoded = [float(xi / gain) for xi in x_quant_decoded]
	x_quant_decoded = np.asarray(x_quant_decoded, dtype=float)
	x_quant_decoded = x_quant_decoded + bias 
	x_decoded = np.reshape(x_quant_decoded, np.shape(x))

	data_size = len(x_quant)
	encoded_size = os.stat(filepath).st_size
	compress_rate = encoded_size / np.float(data_size)

	return x_decoded, data_size, encoded_size, entropy, compress_rate



def test_entropy_res():
	import os 
	from utils import list_dir, writelog
	from imlib import imextract, imsave, RGB2YUV, YUV2RGB
	from range_coder import RangeEncoder, RangeDecoder, prob_to_cum_freq
	from quantize import myconvert_F2I
	from Evaluate.psnr import psnr

	folder_dir = '/home/zhou/BTA/workspace/compress/20180322_1651/aec/sample/test_quant/0.125/'
	alldir = list_dir(folder_dir, '.png')
	color = 'YUV'

	logfile = 'Test/log_test_entropy_%s.txt'  % (color)



	gain_list = [1/256.]

	for gain in gain_list:
		all_e_size = []
		all_d_size = []
		all_entropy = []
		all_PSNR = []
		all_PSNR_p = []
		idx = 0
		for filedir in alldir:
			idx += 1
			Pred, Grd = imextract(filedir)

			Res = Grd - Pred

			if color == 'YUV':
				Res = RGB2YUV(Res, base=255)
				gain_0 = gain 
				gain_1 = 1/256.
				gain_2 = 1/256.
			elif color == 'RGB':
				gain_0 = gain
				gain_1 = gain
				gain_2 = gain

			Res_0 = Res[:,:,0]
			Res_1 = Res[:,:,1]
			Res_2 = Res[:,:,2]

			Res_0_decoded, Res_0_data_size, Res_0_encoded_size, Res_0_entropy, Res_0_rate = Encode_Decode(Res_0, gain_0)
			Res_1_decoded, Res_1_data_size, Res_1_encoded_size, Res_1_entropy, Res_1_rate = Encode_Decode(Res_1, gain_1)
			Res_2_decoded, Res_2_data_size, Res_2_encoded_size, Res_2_entropy, Res_2_rate = Encode_Decode(Res_2, gain_2)

			Res_0_decoded = np.expand_dims(Res_0_decoded, axis=2)
			Res_1_decoded = np.expand_dims(Res_1_decoded, axis=2)
			Res_2_decoded = np.expand_dims(Res_2_decoded, axis=2)

			Res_recons = np.concatenate([Res_0_decoded, Res_1_decoded, Res_2_decoded], axis=2)

			if color == 'YUV':
				Res_recons = YUV2RGB(Res_recons, base=255)

			encoded_size = Res_0_encoded_size + Res_1_encoded_size + Res_2_encoded_size
			data_size = Res_0_data_size = Res_1_data_size + Res_2_data_size
			entropy = (Res_0_entropy + Res_1_entropy  + Res_2_entropy) / 3
			rate = encoded_size / np.float(data_size)

			all_e_size.extend([encoded_size])
			all_d_size.extend([data_size])
			all_entropy.extend([entropy])

			
			Grd_recons = Pred + Res_recons
			

			PSNR = psnr(Grd, Grd_recons)
			PSNR_p = psnr(Grd, Pred)
			all_PSNR.extend([PSNR])
			all_PSNR_p.extend([PSNR_p])

			basename = os.path.basename(filedir)

			filename =  'Test/images/' + basename.replace('.png', '_Range_Decoded_%s.png'%(color))
			filename2 = filename.replace('_Range_Decoded_', '_Range_Decoded_Res_Recons_')
			filename3 = filename.replace('_Range_Decoded_', '_Range_Decoded_Grd_Recons_')
			image_recons = np.concatenate([Grd, Pred, Res], axis=1)

			imsave(filename, image_recons)
			imsave(filename2, Res_recons)
			imsave(filename3, Grd_recons)

			writestr = 'File %d/%d color %s gain %s\n data_size %d encoded_size %d %d %d %d \
compress_rate %.7f %.7f %.7f %.7f entropy %.7f %.7f %.7f %.7f \
PSNR_Pred %f PSNR_recons %f' % \
			(idx, len(alldir), color, gain, data_size, encoded_size, Res_0_encoded_size, Res_1_encoded_size, Res_2_encoded_size, 
			rate, Res_0_rate, Res_1_rate, Res_2_rate, entropy, Res_0_entropy, Res_1_entropy, Res_2_entropy,
			PSNR_p, PSNR) 
			
			print(writestr)
			
			writelog(logfile, writestr)

		avg_e_size = np.mean(all_e_size)
		avg_d_size = np.mean(all_d_size)
		avg_entropy = np.mean(all_entropy)
		avg_PSNR = np.mean(all_PSNR)
		avg_PSNR_p = np.mean(all_PSNR_p)
		avg_rate = avg_e_size / np.float(avg_d_size)

		writestr = 'Average - gain %s data_size %d encoded_size %d compress_rate %.7f entropy %.7f PSNR_Pred %.7f PSNR_recons %.7f' % \
		(gain, avg_d_size, avg_e_size, avg_rate, avg_entropy, avg_PSNR_p, avg_PSNR) 
		print(writestr)
		writelog(logfile, writestr)


if __name__ == '__main__':
	test_get_size()

