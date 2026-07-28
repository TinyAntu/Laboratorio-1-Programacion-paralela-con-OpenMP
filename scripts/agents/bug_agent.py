import os
import json
import uuid
from github import Github, Auth
from google import genai

def analizar_codigo_con_ia(nombre_archivo, contenido_codigo):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return {"encontro_bug": False}

    client = genai.Client(api_key=api_key)
    
    prompt = f"""
    Eres un Agente Revisor de Bugs para código C++/CUDA de un simulador N-cuerpos.
    Analiza el siguiente código y busca específicamente: manejo de errores CUDA ausente (ej. falta de CUDA CHECK), desincronización host/device, o tests rotos por tolerancia.
    
    Debes clasificar el problema encontrado (si lo hay) en una de dos categorías:
    1. Mecánico: Falta de macro CUDA CHECK, o un typo evidente en un test.
    2. Complejo (Humano): Afecta la física, la API pública o la lógica del kernel de CUDA.

    Devuelve ÚNICAMENTE un JSON estricto:
    {{
        "encontro_bug": true/false,
        "es_mecanico": true/false,
        "descripcion": "Descripción del bug",
        "codigo_corregido": "Si es_mecanico es true, devuelve el archivo completo corregido. Si no, string vacío."
    }}

    Archivo ({nombre_archivo}):
    {contenido_codigo}
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=prompt
        )
        texto_limpio = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(texto_limpio)
    except Exception as e:
        print(f"Error detallado de la IA: {e}")
        return {"encontro_bug": False}

def obtener_archivos(repo, path):
    """
    Busca archivos de forma recursiva dentro de una ruta específica en el repositorio.
    """
    archivos_encontrados = []
    try:
        contenidos = repo.get_contents(path)
        for contenido in contenidos:
            if contenido.type == "dir":
                archivos_encontrados.extend(obtener_archivos(repo, contenido.path))
            else:
                archivos_encontrados.append(contenido)
    except Exception as e:
        print(f"No se pudo acceder a la ruta '{path}' o está vacía: {e}")
    return archivos_encontrados

def main():
    token = os.getenv("GITHUB_TOKEN")
    repo_name = os.getenv("GITHUB_REPOSITORY")
    
    if not token or not repo_name:
        print("Error: Faltan variables de entorno de GitHub.")
        exit(1)

    auth = Auth.Token(token)
    g = Github(auth=auth)
    repo = g.get_repo(repo_name)
    carpetas_objetivo = ["include", "src", "test"]
    todos_los_archivos = []
    
    for carpeta in carpetas_objetivo:
        todos_los_archivos.extend(obtener_archivos(repo, carpeta))
        
    if not todos_los_archivos:
        print("No se encontraron archivos en las carpetas especificadas.")
        return

    extensiones_validas = ('.cpp', '.h', '.cu', '.cuh')
    
    for archivo in todos_los_archivos:
        if archivo.name.endswith(extensiones_validas):
            print(f"Analizando {archivo.path}...")
            contenido = archivo.decoded_content.decode("utf-8")
            analisis = analizar_codigo_con_ia(archivo.path, contenido)
            
            if not analisis.get("encontro_bug", False):
                continue
                
            print(f"⚠️ Bug detectado en {archivo.path}.")
            
            if analisis.get("es_mecanico", False):
                rama_base = repo.get_branch("main")
                nueva_rama = f"auto-fix-bug-{uuid.uuid4().hex[:6]}"
                repo.create_git_ref(ref=f"refs/heads/{nueva_rama}", sha=rama_base.commit.sha)
                
                repo.update_file(
                    path=archivo.path,
                    message=f"fix(cuda): corrección mecánica en {archivo.path}",
                    content=analisis["codigo_corregido"],
                    sha=archivo.sha,
                    branch=nueva_rama
                )
                
                pr = repo.create_pull(
                    title=f"Fix automático de código en {archivo.path}",
                    body="El agente de bugs detectó un error mecánico (ej. falta CUDA CHECK) y propone este parche.",
                    head=nueva_rama,
                    base="main"
                )
                print(f"MR mecánico creado: {pr.html_url}")
                
            else:
                issue = repo.create_issue(
                    title=f"Bug lógico en {archivo.path}",
                    body=f"Requiere intervención humana: {analisis['descripcion']}\n\nNo se modificó main directamente.",
                    labels=["bug", "agent"]
                )
                print(f"Issue lógico creado: {issue.html_url}")

if __name__ == "__main__":
    main()