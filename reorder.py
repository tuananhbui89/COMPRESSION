import numpy as np 

def insideout(H, W):
	curX = 0
	curY = 0 
	direction = "down"
	positions = []
	positions.append((curX, curY))
	ltop = 0
	lbot = H
	llef = 0 
	lrig = W 
	while (len(positions) < H*W):
		if direction == 'down':
			while curY < lbot-1:
				curY += 1	
				positions.append((curX, curY))			
			llef += 1 
			direction = 'right'
		elif direction == 'right':
			while curX < lrig-1:
				curX += 1
				positions.append((curX, curY))
			lbot -= 1 
			direction = 'up'
		elif direction == 'up':
			while curY > ltop:
				curY -= 1 
				positions.append((curX, curY))
			lrig -= 1 
			direction = 'left'
		elif direction == 'left':
			while curX > llef:
				curX -= 1 
				positions.append((curX, curY))
			ltop += 1 
			direction = 'down'

	return positions


# Reference from : https://stackoverflow.com/questions/15201395/zig-zag-scan-an-n-x-n-array
def wave(H, W):
    curX = 0
    curY = 0
    direction = "down"
    positions = []
    positions.append((curX, curY))
    while not (curX == W-1 and curY == H-1):
        if direction == "down":
            if curY == H-1: #can't move down any more; move right instead
                curX += 1
            else:
                curY += 1
            positions.append((curX, curY))
            #move diagonally up and right
            while curX < W-1 and curY > 0:
                curX += 1
                curY -= 1
                positions.append((curX, curY))
            direction = "right"
            continue
        else: #direction == "right"
            if curX == W-1: #can't move right any more; move down instead
                curY += 1
            else:
                curX += 1
            positions.append((curX, curY))
            #move diagonally down and left
            while curY < H-1 and curX > 0:
                curX -= 1
                curY += 1
                positions.append((curX, curY))
            direction = "down"
            continue

    return positions

def create_order(H, W, method='ziczac'):
	H = int(H)
	W = int(W)
	if method == 'ziczac':
		positions = wave(H, W)
	elif method == 'insideout':
		positions = insideout(H, W)
	order = []
	for x, y in positions:
		index = x + y*W 
		order.extend([index])
		# print index, x, y
	return order 


def reorder(x, H, W, direction='forward', method='ziczac'):

	if direction == 'forward':
		H , W = np.shape(x)
		order = create_order(H, W, method=method)

		x_flat = np.reshape(x, (-1))
		x_flat = x_flat.tolist()

		x_reorder = [x_flat[index] for index in order]
		
		return x_reorder

	elif direction == 'backward':
		order = create_order(H, W, method=method)
		
		x_reorder = H * W * [0]
		for index in range(H*W):
			x_reorder[order[index]] = x[index]

		x_reorder = np.asarray(x_reorder)
		x_reorder = np.reshape(x_reorder, (H, W))

		return x_reorder

def reorder_1(x, direction='forward'): 
	assert(len(np.shape(x)) == 2)
	H, W = np.shape(x)
	if direction == 'forward': 
		t = reorder(x, H, W, direction='forward', method='ziczac')
		y = reorder(t, H, W, direction='backward', method='insideout')
		return y 

	elif direction == 'backward':
		t = reorder(x, H, W, direction='forward', method='insideout')
		y = reorder(t, H, W, direction='backward', method='ziczac')
		return y 

def reorder_3(x, direction='forward'):
	assert(len(np.shape(x)) == 3)
	H, W, C = np.shape(x)
	assert(C == 3)
	x0 = x[:,:,0]
	x1 = x[:,:,1]
	x2 = x[:,:,2]

	y0 = reorder_1(x0, direction=direction)
	y1 = reorder_1(x1, direction=direction)
	y2 = reorder_1(x2, direction=direction)

	y0 = np.expand_dims(y0, axis=2)
	y1 = np.expand_dims(y1, axis=2)
	y2 = np.expand_dims(y2, axis=2)

	y = np.concatenate([y0, y1, y2], axis=2)

	return y 


def reorder_n(x, direction='forward'):
	if (len(np.shape(x)) == 2): 
		return reorder_1(x, direction=direction)
		
	elif (len(np.shape(x)) == 3):
		return reorder_3(x, direction=direction)

	elif (len(np.shape(x)) == 4):
		B, H, W, C = np.shape(x)
		assert(B == 1)
		y = reorder_3(x[0,:,:,:], direction=direction)
		y = np.expand_dims(y, axis=0)

		return y 


def Test_reorder():
	s = 64
	h = 16 
	w = 256 
	a = list(range(s**2))
	a = np.reshape(a, (h, w))

	b = reorder(a, h, w, direction='forward', method='ziczac')
	b1 = reorder(b, h, w, direction='backward', method='insideout')
	c1 = reorder(b1, h, w, direction='forward', method='insideout')
	c = reorder(c1, h, w, direction='backward', method='ziczac')

	mse = np.mean(np.square(a - c))
	assert(mse == 0)

def Test_reorder_n():
	shape = [1,64,64,3]
	x = np.random.uniform(size=shape)

	y = reorder_n(x, direction='forward')

	x_ = reorder_n(y, direction='backward')

	mse = np.mean(np.square(x - x_))
	print(mse)
	assert(mse == 0)

if __name__ == '__main__':
	Test_reorder_n()
