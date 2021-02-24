from common import resolve_single
from spsr import generator
from utils import load_image
from PIL import Image

model = generator()
model.load_weights('./test/spsr_gan_gen.h5')

lr = load_image('./test/bd.png')
sr = resolve_single(model, lr)

srnum = sr.numpy()

im = Image.fromarray(srnum)
im.save('./test/bd441.png')
