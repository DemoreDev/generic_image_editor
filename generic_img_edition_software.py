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

# Espaço para as funções:

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

# Fim do espaço das funções

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