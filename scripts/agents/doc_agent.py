import os
import json
import uuid
import google.generativeai as genai
from github import Github

def analizar_texto_con_ia(contenido_archivo):
    """
    Envía el contenido a la IA y le pide que lo evalúe devolviendo un JSON estricto.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: No se encontró GEMINI_API_KEY. Saliendo...")
        exit(1)
        
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = f"""
    Eres un Agente Documentador para un proyecto de simulador N-cuerpos en CUDA.
    Analiza el siguiente archivo de documentación. Debes buscar:
    1. Errores mecánicos (typos, enlaces rotos, formato faltante obvio).
    2. Errores técnicos que requieren juicio (falta explicar cómo funciona el kernel de CUDA, decisiones de diseño, etc.).

    Devuelve ÚNICAMENTE un JSON con este formato exacto (sin markdown, sin bloques de código ```json):
    {{
        "encontro_problemas": true/false,
        "es_mecanico": true/false,
        "motivo": "Explicación breve del problema",
        "contenido_corregido": "Si es_mecanico es true, pon aquí el texto completo corregido. Si es false, deja un string vacío."
    }}

    Archivo a analizar:
    {contenido_archivo}
    """
    
    response = model.generate_content(prompt)
    
    try:
        # Limpiar la respuesta por si la IA incluye formato de markdown
        texto_limpio = response.text.replace("```json", "").replace("```", "").strip()
        resultado = json.loads(texto_limpio)
        return resultado
    except Exception as e:
        print(f"Error al parsear el JSON de la IA: {e}")
        return {"encontro_problemas": False}

def main():
    token = os.getenv("GHCR_TOKEN")
    repo_name = os.getenv("GITHUB_REPOSITORY")
    
    if not token or not repo_name:
        print("Error: Faltan variables de entorno de GitHub.")
        exit(1)

    g = Github(token)
    repo = g.get_repo(repo_name)
    
    # Archivos a analizar
    archivos_objetivo = ["README.md", "CHANGELOG.md"]
    
    for ruta_archivo in archivos_objetivo:
        try:
            archivo_repo = repo.get_contents(ruta_archivo)
            contenido_actual = archivo_repo.decoded_content.decode("utf-8")
        except Exception:
            print(f"No se pudo encontrar {ruta_archivo}. Saltando...")
            continue
            
        print(f"Analizando {ruta_archivo}...")
        analisis = analizar_texto_con_ia(contenido_actual)
        
        if not analisis.get("encontro_problemas", False):
            print(f"No hay problemas en {ruta_archivo}.")
            continue
            
        print(f" Problemas encontrados en {ruta_archivo}.")
        
        # Arreglo mecanico
        if analisis.get("es_mecanico", False):
            print("El problema es mecánico. Creando fix automático...")
            nuevo_contenido = analisis["contenido_corregido"]
            
            # 1. Crear nueva rama
            rama_base = repo.get_branch("main")
            nombre_nueva_rama = f"auto-fix-doc-{uuid.uuid4().hex[:6]}"
            repo.create_git_ref(ref=f"refs/heads/{nombre_nueva_rama}", sha=rama_base.commit.sha)
            
            # 2. Hacer commit en la nueva rama
            repo.update_file(
                path=archivo_repo.path,
                message=f"docs(agent): corrección automática en {ruta_archivo}",
                content=nuevo_contenido,
                sha=archivo_repo.sha,
                branch=nombre_nueva_rama
            )
            
            # 3. Crear el Pull Request / Merge Request
            pr = repo.create_pull(
                title=f"Fix automático de documentación en {ruta_archivo}",
                body="El agente documentador detectó y corrigió un error mecánico.",
                head=nombre_nueva_rama,
                base="main"
            )
            
            # 4. Etiquetar
            try:
                pr.add_to_labels("agent: auto-fix")
            except Exception:
                print("No se pudo añadir la etiqueta (asegúrate de que 'agent: auto-fix' exista en el repo).")
            
            print(f"PR creado con éxito: {pr.html_url}")
            
        # Requiere intervencion humana  
        else:
            print("El problema requiere juicio técnico. Abriendo Issue...")
            motivo = analisis.get("motivo", "Falta documentación técnica.")
            
            issue = repo.create_issue(
                title=f"Documentación deficiente en {ruta_archivo}",
                body=f"Requiere intervención humana: {motivo}",
                labels=["documentation", "agent"]
            )
            print(f"Issue creado con éxito: {issue.html_url}")

if __name__ == "__main__":
    main()