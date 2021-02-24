from common import resolve_single
from wdsr import wdsr_b
from utils import load_image
from PIL import Image

model = wdsr_b(scale=4, num_res_blocks=32)
model.load_weights('./test/TF_WDSR_x4.h5')

lr = load_image('./test/c4.jpg')
sr = resolve_single(model, lr)

srnum = sr.numpy()

im = Image.fromarray(srnum)
im.save('./test/coutput44.png')



