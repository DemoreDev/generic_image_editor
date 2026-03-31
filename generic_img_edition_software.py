"""
NOME: Leonardo Doro Demore
NÚMERO USP: 15674786
CÓDIGO DO CURSO: SCC0251 e SCC0651
ANO/SEMESTRE: 2026/1
TÍTULO: TSIE - The Simplest Image Editor
"""

# Biliotecas da disciplina
import numpy as np
import imageio.v3 as iio
import scipy.ndimage

# Para a interface gráfica
import customtkinter as ctk 
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image

# ============================= ESPAÇO DAS FUNÇÕES: =============================

# FUNÇÕES AUXILIARES
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

# FUNÇÕES DE I/O
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
    img = np.clip(img, 0, 255) # Define o intervalo correto [0,255]
    img_to_save = img.astype(np.uint8) # Converte para o formato esperado
    
    iio.imwrite(path, img_to_save)

# -----------------------------------------------------------------------------------

# FUNÇÕES DE TRANSFORMAÇÃO DE INTENSIDADE
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

# FUNÇÕES DE TRANSFORMAÇÃO GEOMÉTRICA
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
    T1_inv = inv_translation_matrix(-cy, -cx)
    
    # Rotaciona 
    R_inv = inv_rot_matrix(theta)
    
    # Move de volta para a posição original (cx, cy)
    T2_inv = inv_translation_matrix(cy, cx)
    
    # Compõe a matriz final 
    final_matrix = T2_inv @ R_inv @ T1_inv
    
    return final_matrix

# Realiza a interpolação bilinear vetorizada: calcula intensidade em coordenadas fracionárias
def interp(img, i_valid, j_valid):
    # Encontra os 4 pixels vizinhos para todas as coordenadas de uma vez
    i0 = np.floor(i_valid).astype(int)
    i1 = i0 + 1
    j0 = np.floor(j_valid).astype(int)
    j1 = j0 + 1

    # Pega as cores dessas 4 pontas
    c0 = img[i0, j0] # Topo esquerda
    c1 = img[i0, j1] # Topo direita
    c2 = img[i1, j0] # Baixo esquerda
    c3 = img[i1, j1] # Baixo direita

    # Calcula as distâncias 
    t = i_valid - i0
    s = j_valid - j0

    # Se a imagem for colorida (3 canais), precisa alinhar a matriz 
    # para a multiplicação funcionar corretamente 
    if img.ndim == 3:
        t = t[:, np.newaxis]
        s = s[:, np.newaxis]

    # Combina as cores 
    c01 = c0 * (1 - s) + c1 * s # Interpolação na linha de cima
    c23 = c2 * (1 - s) + c3 * s # Interpolação na linha de baixo
    c_final = c01 * (1 - t) + c23 * t # Combina as duas linhas 
    
    return c_final

# Aplica a matriz de transformação usando vetorização
def apply_geometric_transform(img, matrix):
    h, w = img.shape[:2]
    new_img = np.zeros_like(img)

    # Cria uma grade com todas as coordenadas 
    i_coords, j_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

    # Transforma as grades em um vetor 3D de coordenadas homogêneas
    coords = np.stack([i_coords.ravel(), j_coords.ravel(), np.ones_like(i_coords).ravel()])

    # Aplica a matriz de transformação de uma vez só
    origin_coords = matrix @ coords

    # Volta o resultado para o formato da imagem
    i_orig = origin_coords[0].reshape(h, w)
    j_orig = origin_coords[1].reshape(h, w)

    # Cria uma máscara dizendo quais pixels caíram dentro da imagem original
    valid_mask = (i_orig >= 0) & (i_orig < h - 1) & (j_orig >= 0) & (j_orig < w - 1)
    
    i_rounded = np.round(i_orig, 3)
    j_rounded = np.round(j_orig, 3)
    alarm_mask = (i_rounded >= 0) & (i_rounded <= h - 1) & (j_rounded >= 0) & (j_rounded <= w - 1)
    
    has_empty_pixels = not np.all(alarm_mask)

    # Extrai apenas as coordenadas válidas
    i_valid = i_orig[valid_mask]
    j_valid = j_orig[valid_mask]

    # Chama a função auxiliar para calcular as novas cores
    interpolated_pixels = interp(img, i_valid, j_valid)

    # "Encaixa" os pixels calculados na nova imagem
    new_img[valid_mask] = interpolated_pixels

    return new_img, has_empty_pixels

# Calcula o zoom necessário para escalar a imagem (caso dos pixels vazios)
def calculate_auto_zoom(angle_degrees: float, width: int, height: int) -> float:
    angle_rad = np.abs(np.deg2rad(angle_degrees))
    
    s = np.abs(np.sin(angle_rad))
    c = np.abs(np.cos(angle_rad))
    
    new_w = (height * s) + (width * c)
    new_h = (height * c) + (width * s)
    
    scale_w = new_w / width
    scale_h = new_h / height
    
    zoom_factor = max(scale_w, scale_h)
    
    return zoom_factor

# ============================= FIM DO ESPAÇO DAS FUNÇÕES =============================

# ============================= ESPAÇO DA UI =============================

# CLASSE DO UI
class ImageEditor(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.geometry("1400x800") 
        self.title("")
        
        # Define a fonte padrão (botões e labels)
        self.buttons_font = ctk.CTkFont(family="Inter", size=20, weight="normal")
        self.lbls_font = ctk.CTkFont(family="Inter", size=24, weight="normal")

        # Estado atual da matriz (imagem)
        self.current_image_array = None
        
        # Configurações de tamanho da TopBar e SideBar
        self.grid_rowconfigure(0, weight=0) 
        self.grid_rowconfigure(1, weight=1) 
        self.grid_columnconfigure(0, weight=0) 
        self.grid_columnconfigure(1, weight=1) 
        
        # --- TopBar ---
        self.topbar = ctk.CTkFrame(self, height=60, corner_radius=0) 
        self.topbar.grid(row=0, column=0, columnspan=2, sticky="ew")

        # 3 colunas (para centralizar)
        self.topbar.grid_columnconfigure(0, weight=1) 
        self.topbar.grid_columnconfigure(1, weight=1) 
        self.topbar.grid_columnconfigure(2, weight=1) 

        # Caixa Seletora na coluna esquerda
        self.mode_selector = ctk.CTkOptionMenu(
            self.topbar,
            values=["Intensidade", "Geométricas"],
            command=self.change_sidebar_mode, 
            font=self.buttons_font,
            fg_color="#1E3A5F",
            button_color="#254A7A",
            button_hover_color="#3a7ebf",
            width=180,
            height=50
        )
        self.mode_selector.grid(row=0, column=0, sticky="w", padx=20, pady=15)
        
        # Título na coluna central
        self.title_lbl = ctk.CTkLabel(
            self.topbar, 
            text="TSIE", 
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.title_lbl.grid(row=0, column=1, sticky="e", pady=15)
        
        # Frame para agrupar os botões de ação na coluna direita
        self.action_frame = ctk.CTkFrame(self.topbar, fg_color="transparent")
        self.action_frame.grid(row=0, column=2, sticky="e", padx=20, pady=10)
        
        # Botão de carregar 
        self.load_btn = self.create_button(self.action_frame, "Carregar Imagem", self.open_image_dialog)
        self.load_btn.pack(side="left", padx=(0, 10))
        
        # Botão de salvar
        self.save_btn = self.create_button(self.action_frame, "Salvar Imagem", self.save_image_dialog)
        self.save_btn.pack(side="left")
        
        # --- SideBar ---
        self.sidebar = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)

        # --------------------------------------------------------------------------------

        # CONTÂINER DAS FUNÇÕES DE INTENSIDADE 
        self.intensity_container = ctk.CTkScrollableFrame(self.sidebar, fg_color="transparent")
        self.intensity_container.pack(fill="both", expand=True)

        self.intensity_frame = ctk.CTkFrame(self.intensity_container, fg_color="transparent")
        self.intensity_frame.pack(pady=10, padx=10, fill="x")
        
        # Título do frame
        self.intensity_lbl = ctk.CTkLabel(
            self.intensity_frame, 
            text="Funções de \nIntensidade", 
            font=self.lbls_font
        )
        self.intensity_lbl.pack(pady=(20, 10))
        
        # Adiciona os botões 
        # Botão de inversa
        self.inv_btn = self.create_button(self.intensity_frame, "Inverter", self.apply_inv)
        self.inv_btn.pack(pady=10, padx=20, fill="x")
        
        # Botão de logarítmica  
        self.log_btn = self.create_button(self.intensity_frame, "Logarítmica", self.apply_log)
        self.log_btn.pack(pady=10, padx=20, fill="x")

        # Botão de Solarização (minha escolha)
        self.solar_btn = self.create_button(self.intensity_frame, "Solarizar", self.apply_solar)
        self.solar_btn.pack(pady=10, padx=20, fill="x")

        # Frame invisível para agrupar o texto, slider e botão 
        self.gamma_group = ctk.CTkFrame(self.intensity_container, fg_color="transparent")
        self.gamma_group.pack(pady=10, padx=10, fill="x", expand=True)

        # Slider do gamma
        self.gamma_lbl = ctk.CTkLabel(self.gamma_group, text="Gamma: 1.00", font=self.buttons_font)
        self.gamma_lbl.pack(pady=(10, 0))
        
        self.slider_gamma = ctk.CTkSlider(
            self.gamma_group, 
            from_=0.1, to=3.0, 
            number_of_steps=29, # Passos de 0.1
            command=self.apply_gamma 
        )
        self.slider_gamma.set(1.0) 
        self.slider_gamma.pack(pady=(0, 10), padx=10)

        # Botão de confirmar 
        self.confirm_gamma_btn = ctk.CTkButton(
            self.gamma_group,
            text="Aplicar gamma",
            width=120,          
            height=40,         
            corner_radius=6,  
            fg_color="#1E3A5F",
            hover_color="#254A7A", 
            border_width=1,
            border_color="#1E3A5F",
            font=ctk.CTkFont(family="Inter", size=16, weight="normal"),
            command=self.confirm_gamma
        )
        self.confirm_gamma_btn.pack(pady=(0, 10))

        # Caixas da modulação de contraste
        self.frame_mod = ctk.CTkFrame(self.intensity_container, fg_color="transparent")
        self.frame_mod.pack(pady=10, padx=10, fill="x", expand=True)
        
        self.frame_mod.grid_columnconfigure(0, weight=1)
        self.frame_mod.grid_columnconfigure(1, weight=1)

        # Título
        self.lbl_mod = ctk.CTkLabel(self.frame_mod, text="Modulação de Contraste", font=ctk.CTkFont(size=12, weight="bold"))
        self.lbl_mod.grid(row=0, column=0, columnspan=2, pady=(0, 10))

        self.lbl_in_min = ctk.CTkLabel(self.frame_mod, text="In Mín:")
        self.lbl_in_min.grid(row=1, column=0)
        
        self.lbl_in_max = ctk.CTkLabel(self.frame_mod, text="In Máx:")
        self.lbl_in_max.grid(row=1, column=1)

        self.entry_in_min = ctk.CTkEntry(self.frame_mod, width=60, height=25)
        self.entry_in_min.grid(row=2, column=0, pady=(0, 10))
        self.entry_in_min.insert(0, "0")

        self.entry_in_max = ctk.CTkEntry(self.frame_mod, width=60, height=25)
        self.entry_in_max.grid(row=2, column=1, pady=(0, 10))
        self.entry_in_max.insert(0, "255")

        self.lbl_out_min = ctk.CTkLabel(self.frame_mod, text="Out Mín:")
        self.lbl_out_min.grid(row=3, column=0)
        
        self.lbl_out_max = ctk.CTkLabel(self.frame_mod, text="Out Máx:")
        self.lbl_out_max.grid(row=3, column=1)

        self.entry_out_min = ctk.CTkEntry(self.frame_mod, width=60, height=25)
        self.entry_out_min.grid(row=4, column=0, pady=(0, 15))
        self.entry_out_min.insert(0, "0")

        self.entry_out_max = ctk.CTkEntry(self.frame_mod, width=60, height=25)
        self.entry_out_max.grid(row=4, column=1, pady=(0, 15))
        self.entry_out_max.insert(0, "255")

        self.mod_btn = ctk.CTkButton(
            self.frame_mod,
            text="Aplicar modulação",
            width=120,          
            height=40,         
            corner_radius=6,  
            fg_color="#1E3A5F",
            hover_color="#254A7A", 
            border_width=1,
            border_color="#1E3A5F",
            font=ctk.CTkFont(family="Inter", size=16, weight="normal"),
            command=self.apply_mod
        )
        self.mod_btn.grid(row=5, column=0, columnspan=2)

        # --------------------------------------------------------------------------------

        # CONTÂINER DAS FUNÇÕES GEOMÉTRICAS   
        self.geometry_container = ctk.CTkScrollableFrame(self.sidebar, fg_color="transparent")

        # Título 
        self.geom_lbl = ctk.CTkLabel(
            self.geometry_container, 
            text="Transformações\nGeométricas", 
            font=self.lbls_font
        )
        self.geom_lbl.pack(pady=(20, 10))

        # Translação
        self.frame_trans = ctk.CTkFrame(self.geometry_container, fg_color="transparent")
        self.frame_trans.pack(pady=10, padx=10, fill="x")

        self.frame_trans.grid_columnconfigure(0, weight=1)
        self.frame_trans.grid_columnconfigure(1, weight=1)

        self.lbl_trans = ctk.CTkLabel(self.frame_trans, text="Translação (Pixels)", font=ctk.CTkFont(family="Inter", size=14, weight="bold"))
        self.lbl_trans.grid(row=0, column=0, columnspan=2, pady=(0, 10))

        self.lbl_dx = ctk.CTkLabel(self.frame_trans, text="Eixo X:")
        self.lbl_dx.grid(row=1, column=0)
        
        self.lbl_dy = ctk.CTkLabel(self.frame_trans, text="Eixo Y:")
        self.lbl_dy.grid(row=1, column=1)

        self.entry_dx = ctk.CTkEntry(self.frame_trans, width=60, height=25)
        self.entry_dx.grid(row=2, column=0, pady=(0, 15))
        self.entry_dx.insert(0, "0")

        self.entry_dy = ctk.CTkEntry(self.frame_trans, width=60, height=25)
        self.entry_dy.grid(row=2, column=1, pady=(0, 15))
        self.entry_dy.insert(0, "0")

        self.trans_btn = self.create_button(self.frame_trans, "Aplicar Translação", self.apply_translation)
        self.trans_btn.grid(row=3, column=0, columnspan=2)

        # Escala
        self.frame_scale = ctk.CTkFrame(self.geometry_container, fg_color="transparent")
        self.frame_scale.pack(pady=10, padx=10, fill="x")

        self.frame_scale.grid_columnconfigure(0, weight=1)
        self.frame_scale.grid_columnconfigure(1, weight=1)

        self.lbl_scale = ctk.CTkLabel(self.frame_scale, text="Escala (Multiplicador)", font=ctk.CTkFont(family="Inter", size=14, weight="bold"))
        self.lbl_scale.grid(row=0, column=0, columnspan=2, pady=(0, 10))

        self.lbl_sx = ctk.CTkLabel(self.frame_scale, text="Fator X:")
        self.lbl_sx.grid(row=1, column=0)
        
        self.lbl_sy = ctk.CTkLabel(self.frame_scale, text="Fator Y:")
        self.lbl_sy.grid(row=1, column=1)

        self.entry_sx = ctk.CTkEntry(self.frame_scale, width=60, height=25)
        self.entry_sx.grid(row=2, column=0, pady=(0, 15))
        self.entry_sx.insert(0, "1.0")

        self.entry_sy = ctk.CTkEntry(self.frame_scale, width=60, height=25)
        self.entry_sy.grid(row=2, column=1, pady=(0, 15))
        self.entry_sy.insert(0, "1.0")

        self.scale_btn = self.create_button(self.frame_scale, "Aplicar Escala", self.apply_scale)
        self.scale_btn.grid(row=3, column=0, columnspan=2)

        # Rotação
        self.frame_rot = ctk.CTkFrame(self.geometry_container, fg_color="transparent")
        self.frame_rot.pack(pady=10, padx=10, fill="x")

        self.lbl_rot_title = ctk.CTkLabel(self.frame_rot, text="Rotação (Centro)", font=ctk.CTkFont(family="Inter", size=14, weight="bold"))
        self.lbl_rot_title.pack(pady=(0, 5))

        self.lbl_rot_val = ctk.CTkLabel(self.frame_rot, text="Ângulo: 0°", font=ctk.CTkFont(size=12))
        self.lbl_rot_val.pack()

        self.slider_rot = ctk.CTkSlider(
            self.frame_rot, from_=-180, to=180, number_of_steps=360, command=self.update_rot_label
        )
        self.slider_rot.set(0)
        self.slider_rot.pack(pady=5)

        # A "mágica" anti-pixels vazios
        self.check_auto_zoom = ctk.CTkCheckBox(
            self.frame_rot, 
            text="Preencher bordas (Auto-Zoom)", 
            font=ctk.CTkFont(size=12),
            fg_color="#1E3A5F", 
            hover_color="#254A7A"
        )
        self.check_auto_zoom.pack(pady=5)
        self.check_auto_zoom.select() 

        self.rot_btn = self.create_button(self.frame_rot, "Aplicar Rotação", self.apply_rotation)
        self.rot_btn.pack(pady=(10, 0))
        
        # --------------------------------------------------------------------------------

        # ÁREA DO CANVAS
        self.canvas = ctk.CTkLabel(
            self, 
            text="Nenhuma imagem carregada",
            font=ctk.CTkFont(size=14),
            text_color="gray",
            fg_color="#1E1E1E",  # Cor de fundo levemente destacada
            corner_radius=10     # Bordas arredondadas para a área da foto
        )
        # O sticky="nsew" faz a Label ocupar 100% do espaço da linha 1 / coluna 1
        self.canvas.grid(row=1, column=1, sticky="nsew", padx=20, pady=20)

    # --------------------------------------------------------------------------------
    
    # FUNÇÕES DOS BOTÕES DA UI
    # Controla qual sidebar aparece
    def change_sidebar_mode(self, new_mode):
        # Esconde ambos os contêineres 
        self.intensity_container.pack_forget()
        self.geometry_container.pack_forget()
        
        # 2. Mostra apenas o contêiner correspondente à seleção
        if new_mode == "Intensidade":
            self.intensity_container.pack(fill="both", expand=True)
        elif new_mode == "Geométricas":
            self.geometry_container.pack(fill="both", expand=True)

    # COnfirmação visual para os botões "aplicar"
    def show_feedback(self, button, original_text, success_text):
        # Muda para verde e altera o texto
        button.configure(text=success_text, fg_color="#2FA572", border_color="#2FA572")
        
        self.after(2000, lambda: button.configure(text=original_text, fg_color="#1E3A5F", border_color="#1E3A5F"))

    # Cria um botão estilizado
    def create_button(self, parent, text, command):
        return ctk.CTkButton(
            parent, 
            text=text,
            font=self.buttons_font,
            corner_radius=6,          
            fg_color="#1E3A5F",   
            hover_color="#254A7A",   
            border_width=2,           
            border_color="#1E3A5F",
            text_color="#ffffff", 
            width=180,
            height=50,   
            command=command
        )

    # Abre o explorer para o usuário escolher uma imagem
    def open_image_dialog(self):
        file_path = filedialog.askopenfilename(
            title="Selecione uma imagem"
        )
        
        if file_path: 
            # Chama a função que carrega a imagem
            self.current_image_array = load_image(file_path)
            
            # Manda a matriz para ser desenhada na tela
            self.update_canvas(self.current_image_array)

    # Salva a imagem 
    def save_image_dialog(self):
        if self.current_image_array is None:
            messagebox.showwarning("Aviso", "Não há nenhuma imagem para salvar!")
            return
            
        # Variável para rastrear qual opção a pessoa escolheu no menu inferior
        tipo_escolhido = ctk.StringVar(value="PNG")
        
        # Removemos o defaultextension e adicionamos a typevariable
        file_path = filedialog.asksaveasfilename(
            title="Salvar imagem como",
            typevariable=tipo_escolhido,
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")]
        )
        
        if file_path:
            if not "." in file_path:
                if "JPEG" in tipo_escolhido.get():
                    file_path += ".jpg"
                else:
                    file_path += ".png"
                    
            try:
                save_image(self.current_image_array, file_path)

                messagebox.showinfo("Sucesso", "Sua imagem foi salva com sucesso!")
                
            except Exception as e:
                messagebox.showerror("Erro ao Salvar", f"Ocorreu um problema ao salvar a imagem:\n{e}")

    # Ajusta a imagem para lidar com pixels pretos
    def adjust_edges(self):
        if self.current_image_array is None: return
        
        if not hasattr(self, 'backup_image_array'): return
        
        h, w = self.backup_image_array.shape[:2]
        
        # Verifica se é rotação ou translação
        if abs(self.slider_rot.get()) > 0.1: 
            ang = self.slider_rot.get()
            theta = np.deg2rad(ang)
            zoom = calculate_auto_zoom(ang, w, h)
            
            # Recria a matriz combinada 
            cy, cx = h / 2.0, w / 2.0
            T1_inv = inv_translation_matrix(cy, cx)  
            R_inv = inv_rot_matrix(theta)
            inv_sm = inv_scale_matrix(zoom, zoom)
            T2_inv = inv_translation_matrix(-cy, -cx) 
            
            # Escala no centro antes de girar de volta
            matriz_final = T2_inv @ inv_sm @ R_inv @ T1_inv
            
            feedback_btn = self.rot_btn
            feedback_txt = "Aplicar Rotação"
            
        else: 
            try:
                tx = float(self.entry_dx.get())
                ty = float(self.entry_dy.get())
                tx_abs = abs(tx)
                ty_abs = abs(ty)
            except ValueError:
                return 

            sx = (w + 2 * tx_abs) / w
            sy = (h + 2 * ty_abs) / h
            
            if sx == 0 or sy == 0: return

            # Matriz de Translação Inversa original
            inv_trans = inv_translation_matrix(ty, tx)

            # Matriz de Escala Central Inversa
            inv_smatrix = inv_scale_matrix(sy, sx)
            cy, cx = h / 2.0, w / 2.0
            T1_inv = inv_translation_matrix(cy, cx)   
            T2_inv = inv_translation_matrix(-cy, -cx) 
            inv_center_scale = T2_inv @ inv_smatrix @ T1_inv

            # Combina as matrizes (Escala Central -> Translação)
            matriz_final = inv_center_scale @ inv_trans
            
            feedback_btn = self.trans_btn
            feedback_txt = "Aplicar Translação"

        adjusted, _ = apply_geometric_transform(self.backup_image_array, matriz_final)
        
        # Atualiza o estado e a tela
        self.current_image_array = adjusted
        self.update_canvas(self.current_image_array)
        
        # Confirmação visual
        self.show_feedback(feedback_btn, feedback_txt, "Ajustado!")

        self.slider_rot.set(0)
        self.lbl_rot_val.configure(text="Ângulo: 0°")

    # Se há pixels pretos, aparece uma mensagem e resolve o "problema"
    def solve_empty_pixels(self):
        # Cria uma mini-janela (TopLevel)
        warning = ctk.CTkToplevel(self)
        warning.title("Atenção")
        warning.geometry("350x150")
        warning.resizable(False, False)
        
        # Faz o pop-up ficar sempre na frente e trava a janela principal
        warning.transient(self)
        warning.grab_set()
        
        # Centraliza o pop-up na tela
        warning.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() // 2) - (350 // 2)
        y = self.winfo_y() + (self.winfo_height() // 2) - (150 // 2)
        warning.geometry(f"+{x}+{y}")

        # Mensagem
        lbl_msg = ctk.CTkLabel(
            warning, 
            text="A transformação gerou bordas vazias (pretas)!\nO que você deseja fazer?",
            font=ctk.CTkFont(size=14)
        )
        lbl_msg.pack(pady=(20, 20))

        # Frame para organizar os botões lado a lado
        frame_botoes = ctk.CTkFrame(warning, fg_color="transparent")
        frame_botoes.pack(fill="x", padx=20)

        def undo():
            # Restaura o backup
            if hasattr(self, 'backup_image_array'):
                self.current_image_array = self.backup_image_array
                self.update_canvas(self.current_image_array)
            warning.destroy() # Fecha o pop-up

        def adjust():
            self.adjust_edges()
            warning.destroy()

        # Botões
        undo_btn = ctk.CTkButton(
            frame_botoes, text="Desfazer", command=undo,
            fg_color="#8B0000", hover_color="#A52A2A" 
        )
        undo_btn.pack(side="left", padx=10, expand=True)

        adjust_btn = ctk.CTkButton(
            frame_botoes, text="Ajustar", command=adjust,
            fg_color="#2FA572", hover_color="#3cb371" 
        )
        adjust_btn.pack(side="right", padx=10, expand=True)

    # Mostra a imagem na tela de forma interativa
    def update_canvas(self, img_array):
        # Converte np.array para PIL Image
        pil_image = Image.fromarray(img_array)
        
        # Redimensiona para caber na tela 
        pil_image.thumbnail((800, 600))
        
        # Converte PIL para a imagem nativa do CustomTkinter (biblioteca da interface)
        ctk_image = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=pil_image.size)
        
        # Atualiza o Label da interface
        self.canvas.configure(image=ctk_image, text="") 
        self.canvas.image = ctk_image 

    # Aplica inversão
    def apply_inv(self):
        if self.current_image_array is None:
            messagebox.showwarning("Aviso", "Por favor, carregue uma imagem primeiro!")
            return

        # Aplica a inversão
        proc_img = f_inv(self.current_image_array)

        # Substituímos a imagem atual pela processada. 
        self.current_image_array = proc_img

        # Manda a nova matriz para ser desenhada na interface
        self.update_canvas(self.current_image_array)

    # Aplica logarítmica
    def apply_log(self):
        if self.current_image_array is None:
            messagebox.showwarning("Aviso", "Por favor, carregue uma imagem primeiro!")
            return

        # Aplica a logarítmica
        proc_img = f_log(self.current_image_array)

        # Substitui a imagem atual pela processada
        self.current_image_array = proc_img

        # Manda a nova matriz para ser desenhada na interface
        self.update_canvas(self.current_image_array)

    # Essa função é um pouco diferente: aplica o preview (em tempo real)
    def apply_gamma(self, gamma_value):
        # Atualiza o texto da Label 
        self.gamma_lbl.configure(text=f"Gamma: {gamma_value:.2f}")
        
        if self.current_image_array is None:
            return # sem print para não poluir o terminal

        # Aplica a gamma
        proc_img = f_gamma(self.current_image_array, gamma=gamma_value)

        self.update_canvas(proc_img)
    
    # Essa função complementa a função acima: confirma as mudanças
    def confirm_gamma(self):
        if self.current_image_array is None:
            messagebox.showwarning("Aviso", "Por favor, carregue uma imagem primeiro!")
            return

        valor_g = self.slider_gamma.get()

        # Processa e substitui a imagem na memória 
        proc_img = f_gamma(self.current_image_array, gamma=valor_g)
        self.current_image_array = proc_img

        # Reseta o slider e o texto para o estado neutro
        self.slider_gamma.set(1.0)
        self.gamma_lbl.configure(text="Gamma: 1.00")

        # Atualiza o canvas por segurança 
        self.update_canvas(self.current_image_array)
        
        # Feedback no botão de confirmar 
        self.show_feedback(self.confirm_gamma_btn, "Aplicar gamma", "Gamma aplicado!")

    # Aplica solarização
    def apply_solar (self):
        if self.current_image_array is None:
            messagebox.showwarning("Aviso", "Por favor, carregue uma imagem primeiro!")
            return

        proc_img = f_solarize(self.current_image_array)

        # Substitui a imagem atual pela processada
        self.current_image_array = proc_img

        # Manda a nova matriz para ser desenhada na interface
        self.update_canvas(self.current_image_array)

    # Aplica modulação de contraste
    def apply_mod(self):
        if self.current_image_array is None:
            messagebox.showwarning("Aviso", "Por favor, carregue uma imagem primeiro!")
            return

        try:
            in_min = float(self.entry_in_min.get())
            in_max = float(self.entry_in_max.get())
            out_min = float(self.entry_out_min.get())
            out_max = float(self.entry_out_max.get())
            
            # Aplica a modularização com os valores escolhidos
            proc_img = f_mod(
                self.current_image_array, 
                input_min=in_min, 
                input_max=in_max, 
                output_min=out_min, 
                output_max=out_max
            )
            
            # Substitui a imagem atual pela processada
            self.current_image_array = proc_img
            
            # Manda a nova matriz para ser desenhada na interface
            self.update_canvas(self.current_image_array)

            # Feedback visual no botão de modulação
            self.show_feedback(self.mod_btn, "Aplicar modulação", "Modulação aplicada!")

            # Reseta as caixas de texto para os valores originais
            self.entry_in_min.delete(0, 'end')
            self.entry_in_min.insert(0, "0")
            
            self.entry_in_max.delete(0, 'end')
            self.entry_in_max.insert(0, "255")
            
            self.entry_out_min.delete(0, 'end')
            self.entry_out_min.insert(0, "0")
            
            self.entry_out_max.delete(0, 'end')
            self.entry_out_max.insert(0, "255")
            
        except ValueError:
            messagebox.showerror("Erro de Digitação", "Por favor, insira apenas números válidos nas caixas de texto.")

    # Atualiza o texto do slider da rotação
    def update_rot_label(self, value):
        self.lbl_rot_val.configure(text=f"Ângulo: {int(value)}°")

    # Aplica translação
    def apply_translation(self):
        if self.current_image_array is None: 
            messagebox.showwarning("Aviso", "Por favor, carregue uma imagem primeiro!")
            return
            
        try:
            tx = float(self.entry_dx.get())
            ty = float(self.entry_dy.get())
            
            # Salva o backup
            self.backup_image_array = self.current_image_array.copy()
            
            # Gera a matriz e aplica
            matrix = inv_translation_matrix(ty, tx)
            proc_img, empty = apply_geometric_transform(self.current_image_array, matrix)
            
            # Atualiza a tela
            self.current_image_array = proc_img
            self.update_canvas(self.current_image_array)
            
            # Verifica se gerou bordas vazias
            if empty: 
                self.solve_empty_pixels()
            else:
                # Feedback visual 
                self.show_feedback(self.trans_btn, "Aplicar Translação", "Aplicado!")
                
            # Reseta a caixa do Eixo X
            self.entry_dx.delete(0, 'end')
            self.entry_dx.insert(0, "0")
            
            # Reseta a caixa do Eixo Y
            self.entry_dy.delete(0, 'end')
            self.entry_dy.insert(0, "0")
                
        except ValueError:
            messagebox.showerror("Erro de Digitação", "Por favor, insira apenas números válidos.")

    # Aplica escala
    def apply_scale(self):
        if self.current_image_array is None: 
            messagebox.showwarning("Aviso", "Por favor, carregue uma imagem primeiro!")
            return
            
        try:
            sx = float(self.entry_sx.get())
            sy = float(self.entry_sy.get())
            
            # Prevenção contra divisão por zero
            if sx == 0 or sy == 0: 
                messagebox.showerror("Erro", "O fator de escala não pode ser zero!")
                return
            
            # Salva o backup
            self.backup_image_array = self.current_image_array.copy()
            
            # Gera a matriz e aplica 
            matrix = inv_scale_matrix(sy, sx)
            proc_img, empty = apply_geometric_transform(self.current_image_array, matrix)
            
            # Atualiza a tela
            self.current_image_array = proc_img
            self.update_canvas(self.current_image_array)
            
            # Verifica se gerou bordas vazias 
            if empty: 
                self.solve_empty_pixels()
            else:
                # Feedback visual 
                self.show_feedback(self.scale_btn, "Aplicar Escala", "✓ Aplicado!")
                
                # Reseta a caixa do Fator X para 
                self.entry_sx.delete(0, 'end')
                self.entry_sx.insert(0, "1.0")
                
                # Reseta a caixa do Fator Y para 
                self.entry_sy.delete(0, 'end')
                self.entry_sy.insert(0, "1.0")
                
        except ValueError:
            messagebox.showerror("Erro de Digitação", "Por favor, insira apenas números válidos.")

    # Aplica rotação
    def apply_rotation(self):
        if self.current_image_array is None: 
            messagebox.showwarning("Aviso", "Por favor, carregue uma imagem primeiro!")
            return
        
        ang = self.slider_rot.get()
        theta = np.deg2rad(ang)
        h, w = self.current_image_array.shape[:2]
        
        # Salva o backup
        self.backup_image_array = self.current_image_array.copy()
        
        # Gera a matriz de Rotação Central 
        cx = w / 2.0
        cy = h / 2.0
        
        T1_inv = inv_translation_matrix(cy, cx)
        R_inv = inv_rot_matrix(theta)
        T2_inv = inv_translation_matrix(-cy, -cx)
        matriz_rot = T2_inv @ R_inv @ T1_inv
        
        # Se a Checkbox do Auto-Zoom estiver marcada, combina as matrizes
        if self.check_auto_zoom.get() == 1:
            zoom = calculate_auto_zoom(ang, w, h)
            
            # Impede erros de arredondamento
            zoom = zoom * 1.02 
            
            inv_scale_m = inv_scale_matrix(zoom, zoom)
            matriz_rot = T2_inv @ inv_scale_m @ R_inv @ T1_inv # Escala no centro antes de girar de volta
            
        # Aplica a transformação
        proc_img, empty = apply_geometric_transform(self.current_image_array, matriz_rot)
        self.current_image_array = proc_img
        self.update_canvas(self.current_image_array)
        
        # Se gerou pixel vazio e o usuário não marcou a caixa de segurança:
        if empty and self.check_auto_zoom.get() == 0:
            self.solve_empty_pixels()
        else:
            self.show_feedback(self.rot_btn, "Aplicar Rotação", "Aplicado!")
            self.slider_rot.set(0)
            self.lbl_rot_val.configure(text="Ângulo: 0°")
    
    # ============================= FIM DO ESPAÇO DA UI =============================

# Main
if __name__ == "__main__":
    app = ImageEditor()
    app.mainloop()