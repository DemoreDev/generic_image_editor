import numpy as np

# função recursiva de busca, enquanto for mesma cor, continuamos a busca e continuamos pintando
def _paint(img, cur_p, nb, bg, fg):
    if not (cur_p[0] < img.shape[0] and cur_p[1] < img.shape[1] and cur_p[0] >= 0 and cur_p[1] >= 0):
        return

    # print(img[*cur_p, 0])
    if img[*cur_p, 0] == bg and img[*cur_p, 1] == bg and img[*cur_p, 2] == bg:
        img[*cur_p, :] = fg
        for next_p in nb:
            _paint(img, (cur_p[0] + next_p[0], cur_p[1] + next_p[1]), nb, bg, fg)

    return







# função de luminosidade para converter imagens coloridas para preto&branco
def luminosity(img):
    return (0.21*img[..., 0] + 0.72*img[..., 1] + 0.07*img[..., 2]).astype(np.uint8)

def histogram(img):

    # cria um vetor acumulador
    accum = np.zeros((256,))
    h, w = img.shape

    # conta as intensidades da imagem toda
    for i in range(h):
        for j in range(w):
            accum[img[i, j]] += 1

    # versão alt. mais rápida usando slice por máscara do numpy
    accum = np.zeros((256,))
    for h in range(256):
        accum[h] = np.sum(img == h)

    return accum

# simplesmente acumulamos valores ao longo de 0-255
def cum_histogram(hist):
    accum = np.zeros((256,))
    accum[0] = hist[0]
    for h in range(1, 256):
        accum[h] = hist[h] + accum[h-1]

    # versão alt. usando função pronta do numpy
    accum = hist.cumsum()
    return accum

# o histograma acumulado tem a característica de uma função de transformação
# só precisamos normalizar a saída para [0-255]
def histogram_equalization(img):
    hist = histogram(img)

    # normaliza histograma para que soma total seja 1.
    chist = cum_histogram(hist / float(hist.sum()))

    # muda valores de [0-1] para [0-255]
    chist *= 255.
    chist = chist.astype(np.uint8)

    h, w = img.shape
    n_img = np.zeros_like(img)
    for i in range(h):
        for j in range(w):
            n_img[i, j] = chist[img[i, j]]
    return n_img

# observem como a função gamma altera o histograma de uma imagem
def f_gamma(v, gamma):
    v = v.astype(float)**gamma
    v = v*(255/(255**gamma))
    return v.astype(np.uint8)

# função de luminosidade para converter imagens coloridas para preto&branco
def luminosity(img):
    return (0.21*img[..., 0] + 0.72*img[..., 1] + 0.07*img[..., 2]).astype(np.uint8)



def conv_op(i, j, img, kernel):

    # como não conseguimos definir um kernel centralizado em índice 0
    # calculamos o deslocamento nas duas direções como a e b
    k_h, k_w = kernel.shape
    a = (k_h-1) // 2
    b = (k_w-1) // 2

    # kernel fica centralizado sobre (i, j) na imagem
    # selecionamos a região em torno de (i, j)
    neighbourhood = img[i-a:i+a+1, j-b:j+b+1]

    # multiplicação ponto a ponto e somatório no final
    c_mul = kernel*neighbourhood
    return c_mul.sum()

def convolve(img, kernel):

    new_img = np.zeros_like(img)
    h, w = img.shape

    k_h, k_w = kernel.shape
    a = (k_h-1) // 2
    b = (k_w-1) // 2

    # como o kernel precisa estar centralizado em cada pixel, 
    # consideramos aqui que não vamos trabalhar nos cantos
    for i in range(a, h-a):
        for j in range(b, w-b):
            new_img[i, j] = conv_op(i, j, img, kernel)

    return new_img

def cont_fun_sin(x, y, freq=10):
    return np.sin(2*np.pi*x*freq) + np.sin(2*np.pi*y*freq)

def quantization(img, B):
    img = img >> 8 - B
    img = img << 8 - B
    return img



def inv_rot_matrix(theta):
    return np.array([[np.cos(theta), np.sin(theta), 0],
            [-np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]])

def inv_scale_matrix(si, sj):
    return np.array([[1.0 / si, 0, 0],
            [0, 1.0 / sj, 0],
            [0, 0, 1]] )

def inv_translation_matrix(ti, tj):
    return np.array([[1, 0, -ti],
            [0, 1, -tj],
            [0, 0, 1]])



