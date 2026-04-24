import tkinter as tk
from tkinterdnd2 import TkinterDnD, DND_FILES
from PIL import Image, ImageTk
import shutil
import os
import subprocess

CARPETA_DESTINO = '/home/catariveros/Desktop/INF-471_CC/Proyecto/DexiNed/data/' #carpeta donde se guarda la imagen que la usuaria ingresa

def guardar_imagen():
    if not os.path.exists(CARPETA_DESTINO):
        os.makedirs(CARPETA_DESTINO)

    ruta_imagen_guardada = [None]  

    def procesar_imagen(file_path):
        file_path = file_path.strip('{}')
        extensiones_validas = ('.png', '.jpg', '.jpeg') #formatos de imagen válidos)
        if not file_path.lower().endswith(extensiones_validas):
            label_estado.config(text="Formato no válido. Usa PNG, JPG o JPEG", fg="red")
            return
        try:
            nombre_archivo = os.path.basename(file_path)
            ruta_final = os.path.join(CARPETA_DESTINO, nombre_archivo)
            shutil.copy(file_path, ruta_final)
            ruta_imagen_guardada[0] = ruta_final  
            zona_drop.config(text="Imagen guardada correctamente", bg="pink", fg="crimson", font=("Helvetica", 13, "bold"))
            label_estado.config(text=f"Guardado en: {ruta_final}", fg="crimson")
            ventana.after(2000, ventana.destroy)
        except Exception as e:
            label_estado.config(text=f"Error: {e}", fg="red")

    def on_drop(event):
        print("Drop detectado:", event.data)
        procesar_imagen(event.data)

    def on_drag_enter(event):
        zona_drop.config(bg="pink")

    def on_drag_leave(event):
        zona_drop.config(bg="pink")

    ventana = TkinterDnD.Tk()
    ventana.title("Arrastrar Imagen → DexiNed")
    ventana.geometry("420x500")
    ventana.resizable(False, False)

    tk.Label(ventana, text="Arrastra una imagen aquí", font=("Helvetica", 14, "bold")).pack(pady=(20, 5))
    tk.Label(ventana, text="PNG · JPG · GIF · BMP", font=("Helvetica", 10), fg="gray").pack()

    zona_drop = tk.Label(ventana, text="⬇  Suelta la imagen aquí  ⬇", bg="#f0f0f0",
                         relief="solid", width=35, height=6, font=("Helvetica", 12), fg="#555")
    zona_drop.pack(pady=20, padx=20)
    zona_drop.drop_target_register(DND_FILES)
    zona_drop.dnd_bind('<<Drop>>', on_drop)
    zona_drop.dnd_bind('<<DragEnter>>', on_drag_enter)
    zona_drop.dnd_bind('<<DragLeave>>', on_drag_leave)

    label_imagen = tk.Label(ventana, bg="white", relief="flat")
    label_imagen.pack(pady=5)

    label_estado = tk.Label(ventana, text="", font=("Helvetica", 10))
    label_estado.pack(pady=10)

    ventana.mainloop()
    return ruta_imagen_guardada[0]  #retorna la ruta


def dexined(ruta_imagen): #esta función la debo complementar para que retorne los directorios de las imagenes que se producen 
    #hasta el momento solo corre DexiNet y estas se guardan en las carpetas establecidad por el algorimo. 
    if ruta_imagen is None:
        print("No se guardó ninguna imagen.")
        return

    RUTA_DEXINED = '/home/catariveros/Desktop/INF-471_CC/Proyecto/DexiNed/'

    print(f"Ejecutando DexiNed sobre: {ruta_imagen}")
    subprocess.run(
        ['python', 'main.py', '--choose_test_data', '9'],
        cwd=RUTA_DEXINED
    )



def seleccion_contorno(rutas_contornos):
    #esta función toma las rutas de las 3 imagenes generadas por DexiNet, las muestra en pantalla 
    #y tiene como output la selección del usuario

    return ruta_contorno_seleccionado


def filtro(ruta_imagen_contorno):
    #toma la imagen de contono escogida y le aplica el filtro para resaltar zonas más oscuras
    #retorna la imagen (su ruta) filtrada

    return ruta_imagen_filtrada


def lineas_espirales(ruta_imagen_filtrada):
    #toma la imagen filtrada y le "ajusta" las líneas espirales continuas
    #retorna una imagen de fondo blanco con las líneas negras

    return ruta_lineas_espirales


def decoracion(ruta_lineas_espirales):
    #a partir de las espirales generadas en el punto anterior, aplica un algortimo para ornamentar
    #guarda los 5 diseños generados en una carpeta y retorna su ruta (directorio)
    return rutas_disenos


def seleccion_diseno(rutas_disenos):
    #toma las imagenes generadas en el punto anterior y las muestra en una ventana interactiva para que la usaria escoja su preferida
    #si no le gusta ninguna, se vuelve a ejecutar la función anterior
    #si la persona escoje un diseño, este se guarda automáticamente en pdf para que se pueda imprimir.
    return pdf


ruta = guardar_imagen()
dexined(ruta)





