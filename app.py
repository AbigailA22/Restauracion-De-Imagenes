import streamlit as st
import cv2
import numpy as np
from PIL import Image
from scipy.signal import convolve2d
from skimage.restoration import richardson_lucy
from numpy.fft import fft2, ifft2
from io import BytesIO

st.title("App de Restauración de imágenes ")

with st.expander("📚 Ver Teoría de los Filtros utilizados"):
    st.markdown("""
# 📖 Teoría de los Filtros utilizados

A continuación se explican brevemente los filtros utilizados en esta aplicación:

---

### 🎯 **1. Gaussian Blur**
Este filtro suaviza la imagen utilizando una convolución con un **kernel gaussiano**. Reduce el ruido y el detalle.
- Se basa en la función gaussiana para dar más peso a los píxeles centrales.
- Útil para eliminar ruido aleatorio.

---

### 🎯 **2. Median Filter**
Filtra cada píxel reemplazándolo por la **mediana** de sus vecinos en una ventana.
- Muy efectivo para eliminar **"ruido sal y pimienta"** sin desenfocar tanto como el Gaussian Blur.

---

### 🎯 **3. Laplacian**
Detecta los bordes de una imagen aplicando el operador **Laplaciano**, que es un filtro de segundo orden.
- Realza los cambios rápidos de intensidad.
- Resalta contornos y bordes.

---

### 🎯 **4. Wiener Filter**
Es un filtro adaptativo que **reduce el ruido** basado en un modelo estadístico del mismo.
- Necesita un kernel (PSF) y un parámetro `K` que controla la cantidad de supresión del ruido.
- Útil para imágenes borrosas con ruido conocido.

---

### 🎯 **5. Adjust Contrast**
Ajusta el contraste de la imagen mediante una transformación lineal:
- `nueva = alpha * imagen + beta`
- `alpha` aumenta o reduce el contraste.
- `beta` ajusta el brillo.

---

### 🎯 **6. Equalización de Histograma**
Redistribuye los niveles de intensidad de la imagen para mejorar el contraste.
- Hace que el histograma de la imagen sea más plano y amplio.
- Útil para mejorar detalles en zonas oscuras o claras.

---

### 🎯 **7. CLAHE (Contrast Limited Adaptive Histogram Equalization)**
Una versión mejorada de la ecualización de histograma que trabaja en bloques pequeños (locales).
- Evita la sobreexposición del contraste.
- Especialmente útil para imágenes con iluminación no uniforme.

---

### 🎯 **8. Deconvolución (Richardson-Lucy)**
Algoritmo iterativo para **restaurar** imágenes borrosas con una estimación del punto de dispersión (PSF).
- Supone que la imagen fue borrosa por un proceso conocido.
- Mejora la nitidez en imágenes desenfocadas.

---

### 🎯 **9. Erosión**
Operación morfológica que **reduce** las regiones blancas en una imagen binaria.
- Elimina pequeños detalles o ruido.
- Útil para separar objetos conectados.

---

### 🎯 **10. Dilatación**
Lo contrario de la erosión. **Expande** las regiones blancas.
- Rellena huecos y conecta componentes.

---

### 🎯 **11. Opening**
Combinación de **erosión seguida de dilatación**.
- Elimina ruido pequeño sin afectar tanto el tamaño de los objetos.

---

### 🎯 **12. Closing**
Combinación de **dilatación seguida de erosión**.
- Rellena pequeños huecos o cavidades dentro de los objetos.

---
""")

def high_boost_filter(img, A=1.5):
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    mask = img - blurred
    return np.clip(img + A * mask, 0, 255)

def low_pass_filter(img):
    kernel = np.ones((5, 5), np.float32) / 25
    return cv2.filter2D(img, -1, kernel)

def high_pass_filter(img):
    kernel = np.array([[-1, -1, -1],
                       [-1,  8, -1],
                       [-1, -1, -1]])
    return cv2.filter2D(img, -1, kernel)


def gaussianBlur(src, size):
    return cv2.GaussianBlur(src, (size, size), 0)

def median(src, size):
    return cv2.medianBlur(src, size)

def laplacian(src):
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]])
    laplacian = cv2.filter2D(src, ddepth=-1, kernel=kernel)
    final = src - laplacian
    return final

def gkern(l=5, sig=1.):
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)

def wiener_filter(src, kernel, K):
    kernel /= np.sum(kernel)
    dummy = np.copy(src)
    dummy = fft2(dummy)
    kernel = fft2(kernel, s=src.shape)
    kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + K)
    dummy = dummy * kernel
    dummy = np.abs(ifft2(dummy))
    return dummy

def contrastAdjustment(src, alpha, beta):
    return cv2.convertScaleAbs(src, alpha=alpha, beta=beta)

def histogramEqualization(src):
    return cv2.equalizeHist(src)

def clahe(src):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(src)

def deconvolution(src):
    psf = np.ones((5, 5)) / 25
    camera = convolve2d(src, psf, 'same')
    rng = np.random.default_rng()
    src_noise = src + 0.1 * src.std() * rng.standard_normal(src.shape)
    deconvolved = richardson_lucy(camera, psf, 5)
    return deconvolved

def erosion(src, kernel):
    return cv2.erode(src, kernel, iterations=1)

def dilatacion(src, kernel):
    return cv2.dilate(src, kernel, iterations=1)

def opening(src, kernel):
    return cv2.morphologyEx(src, cv2.MORPH_OPEN, kernel)

def closing(src, kernel):
    return cv2.morphologyEx(src, cv2.MORPH_CLOSE, kernel)

def normalize_and_convert(img):
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)




imagen_cargada = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"], key="uploader")

filtros = [
    "Original",
    "Gaussian Blur",
    "Median Filter",
    "Laplacian",
    #"Wiener Filter",
    "Adjust Contrast",
    "Equalización de Histograma",
    "CLAHE",
    #"Deconvolución",
    "Erosión",
    "Dilatación",
    "Opening",
    "Closing"
]

if imagen_cargada:
    imagen = Image.open(imagen_cargada).convert("L")  # Escala de grises
    imagen_np = np.array(imagen)

    filtro = st.selectbox("Selecciona un filtro", filtros)

    if filtro == "Original":
        resultado = imagen_np

    elif filtro == "Gaussian Blur":
        size = st.slider("Tamaño del kernel (impar)", 3, 21, 5, step=2)
        resultado = gaussianBlur(imagen_np, size)

    elif filtro == "Median Filter":
        size = st.slider("Tamaño del kernel (impar)", 3, 21, 5, step=2)
        resultado = median(imagen_np, size)

    elif filtro == "Laplacian":
        resultado = laplacian(imagen_np)

    elif filtro == "Wiener Filter":
        size = st.slider("Tamaño del kernel", 3, 15, 5, step=2, key="size_wiener")
        K = st.slider("Valor de K (ruido)", 0.01, 1.0, 0.1, key="k_wiener")
        kernel = gkern(size)
        resultado = wiener_filter(imagen_np, kernel, K)

    elif filtro == "Adjust Contrast":
        alpha = st.slider("Alpha (contraste)", 0.5, 3.0, 1.5)
        beta = st.slider("Beta (brillo)", 0, 100, 25)
        resultado = contrastAdjustment(imagen_np, alpha, beta)

    elif filtro == "Equalización de Histograma":
        resultado = histogramEqualization(imagen_np)

    elif filtro == "CLAHE":
        resultado = clahe(imagen_np)

    elif filtro == "Deconvolución":
        resultado = deconvolution(imagen_np)

    elif filtro in ["Erosión", "Dilatación", "Opening", "Closing"]:
        size = st.slider("Tamaño del kernel", 1, 15, 3)
        kernel = np.ones((size, size), np.uint8)

        if filtro == "Erosión":
            resultado = erosion(imagen_np, kernel)
        elif filtro == "Dilatación":
            resultado = dilatacion(imagen_np, kernel)
        elif filtro == "Opening":
            resultado = opening(imagen_np, kernel)
        elif filtro == "Closing":
            resultado = closing(imagen_np, kernel)

    
    st.subheader("Resultado")
    st.image(resultado, channels="GRAY", clamp=True)

    
    imagen_final = Image.fromarray(np.uint8(resultado))

    
    buffer = BytesIO()
    imagen_final.save(buffer, format="PNG")
    buffer.seek(0)

    # Botón de descarga
    st.download_button(
        label="📥 Descargar imagen procesada",
        data=buffer,
        file_name=f"{filtro.replace(' ', '_').lower()}.png",
        mime="image/png"
    )