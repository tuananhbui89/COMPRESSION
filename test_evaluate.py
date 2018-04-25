# TODO: plot error line 

from utils import read_all_lines, mkdir_p
from parse import parse
import matplotlib.pyplot as plt
import numpy as np 


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plotres(filedir, exptime, task, wd):
	process = read_all_lines(filedir)
	print(len(process))
	time_line = []
	en_loss = []
	re_loss = []
	l_loss =[]
	e_loss = []
	sim_loss = []
	mse = []
	psnr = []
	for each_line in process:
		#'20180425_1659: epoch 0 step 0.0% entropy_loss: 0.000000 recon_loss 0.708479 l_loss 1.384947 e_loss 0.692773 sim_loss 0.000000 mse: 0.708481 psnr: 1.496716 max_z: 127.0 min_z: -117.0'
		data = parse('{}: epoch {} step {} entropy_loss: {} recon_loss {} l_loss {} e_loss {} sim_loss {} mse: {} psnr: {} max_z: {} min_z: {}',each_line)
		t = np.float(data[1]) + np.float(data[2][:-1]) / 100.
		time_line.append(t)
		en_loss.append(np.float(data[3]))
		re_loss.append(np.float(data[4]))
		l_loss.append(np.float(data[5]))
		e_loss.append(np.float(data[6]))
		sim_loss.append(np.float(data[7]))
		mse.append(np.float(data[8]))
		psnr.append(np.float(data[9]))

	outdir = 'Result/%s/' % (exptime) 
	mkdir_p(outdir)
	plt.figure(1)
	plt.plot(time_line, re_loss)
	plt.xlabel('epoch')
	plt.ylabel('recons_loss')
	plt.savefig(outdir + 'recons_loss_%s.png' % (task))

	time_line = moving_average(time_line, wd)
	en_loss = moving_average(en_loss, wd)
	re_loss = moving_average(re_loss, wd)
	l_loss = moving_average(l_loss, wd)
	e_loss = moving_average(e_loss, wd)
	sim_loss = moving_average(sim_loss, wd)
	psnr = moving_average(psnr, wd)


	plt.figure(2)
	plt.plot(time_line, psnr)
	plt.xlabel('epoch')
	plt.ylabel('PSNR')
	plt.savefig(outdir + 'psnr_%s.png' % (task))

	plt.figure(3)
	plt.plot(time_line, en_loss)
	plt.hold(True)
	plt.plot(time_line, re_loss)
	plt.plot(time_line, l_loss)
	plt.plot(time_line, e_loss)
	plt.plot(time_line, sim_loss)
	plt.hold(False)
	plt.legend(['en_loss', 're_loss', 'l_loss', 'e_loss', 'sim_loss'])
	plt.savefig(outdir + 'all_loss_%s.png' % (task))


def test1():
	exptime = '20180425_1219'
	log_train = '/home/zhou/BTA/workspace/compress/densenet/%s/model/log_dense_net_prediction.txt' % (exptime)

	plotres(log_train, exptime, 'train', 20)
	


if __name__ == '__main__':
	test1()