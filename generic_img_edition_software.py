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
def norm_minmax(image_array: np.ndarray, 
                scale_factor: float = 255, 
                offset_value: float = 0):
    
    # Evita divisão por zero
    diff = image_array.max() - image_array.min()
    if diff == 0:
        return image_array
    
    image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min())
    image_array *= scale_factor
    image_array -= offset_value
    
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
    np.clip(img, 0, 255) # Define o intervalo correto [0,255]
    img_to_save = img.astype(np.uint8) # Converte para o formato esperado

    try:
        iio.imwrite(path, img_to_save)
        print(f"Imagem salva com sucesso!\nImagem salva em: {path}")
    except Exception as e:
        print(f"Erro ao salvar a imagem: {e}")

# -----------------------------------------------------------------------------------

# Funções de transformação de intensidade
# Inverte as intensidades da imagem (claro -> escuro) e vice-versa
def f_inv(light):
    return 255-light

# Realça regiões escuras (expande tons baixos)
def f_log(light):
    # Aplica log em todos os pixels
    img = np.log(light.astype(float)+1)

    # Normalização (volta o intervalo para [0,255])
    img = img * 255/np.log(255+1)

    # Retorna no formato esperado
    return img.astype(np.uint8)

# Ajusta o brilho e o contraste 
def f_gamma(light, gamma: float = 1.0):
    # Se gamma > 1, clareia a imagem; se gamma < 1, escurece a imagem
    img = light.astype(float)**(1/gamma) # 1/gamma pois aumentar o slider clareia a imagem

    # Mapeia para o intervalo correto [0,255]
    img = img * 255/(255**(1/gamma))

    # Retorna no formato esperado
    return img.astype(np.uint8)

# Modula o contraste (em uma região definida)
def f_mod(light, 
          input_min: float = 0, 
          input_max: float = 255, 
          output_min: float = 0, 
          output_max: float = 255):
    
    # Evita divisão por zero
    diff = (input_max - input_min)
    if diff == 0: return light
    
    # Normaliza para o intervalo [0,1]
    normalized_light = ((light.astype(float) - input_min) / (diff))

    # Escalonamento (multiplica pela largura do intervalo de saída)
    scaled_light = normalized_light * (output_max - output_min)

    # Ajusta o mínimo do intervalo de saída
    shifted_light = scaled_light + output_min

    # Ajusta para o intervalo da imagem de 8 bits
    clipped_light = np.clip(shifted_light, 0, 255)

    # Retorna no formato esperado
    return clipped_light.astype(np.uint8)

# Função inventada: Aplica o efeito de solarização usando uma função parabólica (Efeito Sabattier)
def f_solarize(light):
    # Aplica a parábola: 4/255 * r * (255 - r)
    solarized = (4 / 255.0) * light.astype(float) * (255.0 - light.astype(float))
    
    # Retorna no formato esperado
    return solarized.astype(np.uint8)

# -----------------------------------------------------------------------------------
# Funções de transformação geométrica
# Calcula a matriz de rotação inversa
def inv_rot_matrix(theta):
    return np.array([[np.cos(theta), np.sin(theta), 0],
            [-np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]])

# Calcula a matriz de escala inversa
def inv_scale_matrix(si, sj):
    return np.array([[1.0 / si, 0, 0],
            [0, 1.0 / sj, 0],
            [0, 0, 1]] )

# Calcula a matriz de translação inversa
def inv_translation_matrix(ti, tj):
    return np.array([[1, 0, -ti],
            [0, 1, -tj],
            [0, 0, 1]])

# Calcula a matriz para rotacionar a imagem em torno do centro
def inv_central_rot_matrix(theta, width, height):
    # Calcula as coordenadas do centro
    cx = width / 2.0
    cy = height / 2.0
    
    # Move o centro para (0,0)
    T1_inv = inv_translation_matrix(-cx, -cy)
    
    # Rotaciona 
    R_inv = inv_rot_matrix(theta)
    
    # Move de volta para a posição original (cx, cy)
    T2_inv = inv_translation_matrix(cx, cy)
    
    # Compõe a matriz final 
    final_matrix = T2_inv @ R_inv @ T1_inv
    
    return final_matrix

# Realiza a interpolação bilinear: calcula intensidade em coordenadas fracionárias
def interp(img, i_cont, j_cont):

    i0 = np.floor(i_cont).astype(int) 
    i1 = i0 + 1 if i0 < img.shape[0]-1 else i0 
    j0 = np.floor(j_cont).astype(int) # Pixel à esquerda/cima
    j1 = j0 + 1 if j0 < img.shape[1]-1 else j0 # Pixel à direita/baixo

    c0 = img[i0, j0] # Topo esquerda
    c1 = img[i0, j1] # Topo direita
    c2 = img[i1, j0] # Baixo esquerda
    c3 = img[i1, j1] # Baixo direita

    # Calcula as distâncias
    t = i_cont - i0
    s = j_cont - j0

    c01 = c0*(1 - s) + c1*s # Interpolação na linha de cima
    c23 = c2*(1 - s) + c3*s # Interpolação na linha de baixo
    c = c01*(1 - t) + c23*t # Combina as duas linhas 
    return c

# Aplica uma transformação, dependendo da matriz passada
def apply_geometric_transform(img, matrix):
    # Pega a altura e largura da imagem original
    h, w = img.shape[:2]

    # Cria uma imagem vazia totalmente preta
    new_img = np.zeros_like(img)

    # Flag para saber se a transformação gerou pixels vazios
    has_empty_pixels = False
    
    # Loop para iterar por todos os pixels
    for i in range(h):
        for j in range(w):
            # Transforma a posição atual em um vetor de coordenadas homogêneas
            coords = np.array([i, j, 1])
            # O resultado dessa multiplicação é a origem
            origin_coords = matrix @ coords
            i_orig, j_orig = origin_coords[0], origin_coords[1]
            
            # Verifica se a coordenada de origem está dentro da imagem original
            if 0 <= i_orig < h - 1 and 0 <= j_orig < w - 1:
                new_img[i, j] = interp(img, i_orig, j_orig)
            else:
                has_empty_pixels = True
                
    return new_img, has_empty_pixels

# Calcula o zoom necessário para escalar a imagem (caso dos pixels vazios)
def calculate_auto_zoom(angle_degrees: float, width: int, height: int) -> float:
    # Converte ângulo para radianos (garantindo 0 < angle < 90)
    angle_rad = np.abs(np.deg2rad(angle_degrees))
    
    # Seno e Cosseno do ângulo
    s = np.sin(angle_rad)
    c = np.cos(angle_rad)
    
    # Calcula as novas dimensões teóricas (com cantos pretos)
    new_w = (height * s) + (width * c)
    new_h = (height * c) + (width * s)
    
    # Calcula o fator de escala (Zoom)
    scale_w = new_w / width
    scale_h = new_h / height
    
    # O maior fator garante que toda a área visível seja preenchida
    zoom_factor = max(scale_w, scale_h)
    
    return zoom_factor

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