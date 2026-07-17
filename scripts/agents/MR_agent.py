import os
import json
from github import Github
from google import genai 

def evaluar_diff_pr_con_ia(diff_texto):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return {"es_mecanico": False, "razonamiento": "Falta GEMINI_API_KEY en el entorno."}
        
    client = genai.Client(api_key=api_key) 
    
    prompt = f"""
    Eres un Agente Revisor de Merge Requests para un proyecto CUDA.
    Analiza las diferencias (diff) de este PR y clasifícalo basándote estrictamente en esta regla:
    - Es 'mecánico': Solo cambia documentación, formato, o son cambios que no alteran la semántica física ni las firmas públicas.
    - Requiere revisión humana: Altera la lógica física, cambia llamadas de API o modifica el comportamiento del kernel.

    Devuelve ÚNICAMENTE un JSON estricto:
    {{
        "es_mecanico": true/false,
        "razonamiento": "Explica brevemente por qué tomaste la decisión."
    }}

    Diff del PR:
    {diff_texto}
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents=prompt
        )
        texto_limpio = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(texto_limpio)
    except Exception as e:
        print(f"Error en la IA: {e}")
        return {"es_mecanico": False, "razonamiento": "Fallo en el análisis de IA. Se requiere humano por seguridad."}

def main():
    token = os.getenv("GITHUB_TOKEN")
    repo_name = os.getenv("GITHUB_REPOSITORY")
    pr_number = os.getenv("PR_NUMBER")
    
    if not all([token, repo_name, pr_number]):
        print("Faltan variables de entorno de GitHub.")
        exit(1)
        
    g = Github(token)
    repo = g.get_repo(repo_name)
    pr = repo.get_pull(int(pr_number))
    
    # 1. Verificar estado del CI (Último commit)
    ultimo_commit = pr.get_commits().reversed[0]
    estados = ultimo_commit.get_statuses()
    ci_exitoso = True
    for estado in estados:
        if estado.state != "success" and estado.context != "Agente Revisor de MR IA":
            ci_exitoso = False
            break

    # 2. Leer las diferencias de código (Diff)
    archivos_cambiados = pr.get_files()
    diff_completo = ""
    for archivo in archivos_cambiados:
        diff_completo += f"--- {archivo.filename}\n+++ {archivo.filename}\n{archivo.patch}\n\n"

    # 3. Analizar con IA
    analisis = evaluar_diff_pr_con_ia(diff_completo)
    
    # 4. Construir el comentario de salida
    comentario = " **Evaluación del Agente Revisor de MR**\n\n"
    
    if not ci_exitoso:
        comentario += "❌ **Estado del CI:** El pipeline de integración continua ha fallado o está pendiente. Por favor, revisa los logs.\n\n"
    else:
        comentario += "✅ **Estado del CI:** Los tests pasaron correctamente.\n\n"

    if analisis.get("es_mecanico", False):
        comentario += "**Veredicto:** Mecánico y mergeable.\n"
    else:
        comentario += "**Veredicto:** Requiere revisión humana.\n"
        
    comentario += f"**Análisis:** {analisis.get('razonamiento', 'Sin razón provista.')}\n\n"
    comentario += "*Nota: Este agente nunca fusionará el código automáticamente a main.*"
    
    # 5. Publicar comentario en el PR
    pr.create_issue_comment(comentario)
    print("Comentario publicado exitosamente en el PR.")

if __name__ == "__main__":
    main()