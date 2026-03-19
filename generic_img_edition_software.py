"""
NOME: Leonardo Doro Demore
NÚMERO USP: 15674786
CÓDIGO DO CURSO: SCC0251 e SCC0651
ANO/SEMESTRE: 2026/1
TÍTULO: SMDC Image Editor (Simples, Mas é De Coração)
"""

# Biliotecas da disciplina
import numpy as np
import imageio.v3 as iio
import scipy.ndimage

# Para a interface gráfica
import customtkinter as ctk 

# ============================= Espaço para as funções: =============================
# Funções auxiliares:
# Modifica o intervalo da imagem
def norm_minmax(image_array:  np.ndarray, 
                scale_factor: float = 255, 
                offset_value: float = 0):  
    image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min())
    image_array *= scale_factor # Transforma o intervalo [0,1] em [0,255] (controla a amplitude)
    image_array -= offset_value # Transforma o intervalo [0,1] em [-1,1] (controla o ponto de partida)
    return image_array

# Verifica se a imagem é RGBA
def is_rgba(img: np.ndarray):
    if img.ndim == 3: # Verifica se não é cinza
        if img.shape[2] == 4: # E se possui 4 canais
            return True
    return False

# -----------------------------------------------------------------------------------

# Funções de I/O:
# Carrega e trata (se necessário) a imagem escolhida
def load_image(path: str) -> np.ndarray:
    try:
        img = iio.imread(path)
    except Exception as e:
        raise ValueError(f"Erro ao ler a imagem '{path}': {e}") from e

    # Simplifica Imagens RGBA para RGB (caso aplicável)
    if is_rgba(img):
        img = img[:, :, :3] # Mantém apenas os 3 primeiros canais (descarta o Alpha)
    
    return img

# Trata (se necessário) e salva a imagem escolhida
def save_image(img: np.ndarray, path: str):
    img_to_save = norm_minmax(img) # Define o intervalo correto [0,255]
    img_to_save = img_to_save.astype(np.uint8) # Converte para o formato esperado

    try:
        iio.imwrite(path, img_to_save)
        print(f"Imagem salva com sucesso!\nImagem salva em: {path}")
    except Exception as e:
        print(f"Erro ao salvar a imagem: {e}")

# -----------------------------------------------------------------------------------

# Funções de transformação de intensidade
#
def f_inv(light):
    return 255-light

#
def f_log(light):
    img = np.log(light.astype(float)+1)
    img = img * 255/np.log(255+1)
    return img.astype(np.uint8)

#
def f_gamma(light, gamma=2.2):
    img = light.astype(float)**(1/gamma)
    img = img * 255/(255**(1/gamma))
    return img.astype(np.uint8)

#
def f_mod(light, a=30, b=200, c=0, d=255):
    return np.clip((light.astype(float)-a)*((d-c)/(b-a)) + c, 0, 255).astype(np.uint8)

#
def f_solarize(light):
    """
    Função Criativa: Solarização (Efeito Sabattier).
    Aplica uma transformação parabólica que inverte tons claros e escuros 
    em relação ao ponto médio.
    """
    # Convertemos para float para evitar estouro (overflow) durante a multiplicação
    v = light.astype(float)
    
    # Aplicamos a parábola: 4/255 * r * (255 - r)
    # Isso garante que o valor máximo (255) ocorra no input 127.5
    res = (4 / 255.0) * v * (255.0 - v)
    
    # Retornamos como uint8 (inteiros de 0 a 255)
    return res.astype(np.uint8)

# -----------------------------------------------------------------------------------
# Funções de transformação geométrica
#
def interp(img, i_cont, j_cont):

    i0 = np.floor(i_cont).astype(int)
    i1 = i0 + 1 if i0 < img.shape[0]-1 else i0
    j0 = np.floor(j_cont).astype(int)
    j1 = j0 + 1 if j0 < img.shape[1]-1 else j0

    c0 = img[i0, j0]
    c1 = img[i0, j1]
    c2 = img[i1, j0]
    c3 = img[i1, j1]

    t = i_cont - i0
    s = j_cont - j0

    c01 = c0*(1 - s) + c1*s
    c23 = c2*(1 - s) + c3*s
    c = c01*(1 - t) + c23*t
    return c

#
def apply_geometric_transform(img, matrix):
    """
    Função genérica para aplicar transformações geométricas usando mapeamento inverso.
    """
    h, w = img.shape[:2]
    # Se for colorida, mantém 3 canais, se for cinza, 1.
    new_img = np.zeros_like(img)
    
    for i in range(h):
        for j in range(w):
            # Vetor de coordenadas homogêneas [i, j, 1]
            coords = np.array([i, j, 1])
            # Multiplicação pela matriz inversa para achar a origem
            origin_coords = matrix @ coords
            i_orig, j_orig = origin_coords[0], origin_coords[1]
            
            # Verifica se a coordenada de origem está dentro da imagem original
            if 0 <= i_orig < h - 1 and 0 <= j_orig < w - 1:
                # Usa sua função interp para evitar serrilhamento
                new_img[i, j] = interp(img, i_orig, j_orig)
            else:
                # Aqui entra o tratamento de 'pixels vazios'
                # Por padrão, preenchemos com 0 (preto), mas o requisito
                # pede para evitar isso no software final.
                pass
                
    return new_img

# ============================= Fim do espaço das funções =============================

# --- CLASSE DA INTERFACE ---
class ImageEditor(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("USP Image Editor")
        self.geometry("1000x600")
        
        # Configurar Grid
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # Painel Lateral (Sidebar)
        self.sidebar = ctk.CTkFrame(self, width=200)
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Botões e Sliders aqui...
        self.btn_load = ctk.CTkButton(self.sidebar, text="Abrir Imagem", command=self.load_image)
        self.btn_load.pack(pady=10)
        
        # Área da Imagem
        self.canvas = ctk.CTkLabel(self, text="Nenhuma imagem carregada")
        self.canvas.grid(row=0, column=1, padx=20, pady=20)

    def load_image(self):
        # Lógica para carregar com imageio
        pass

# Execução
if __name__ == "__main__":
    app = ImageEditor()
    app.mainloop()