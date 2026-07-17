import os
import json
import google.generativeai as genai
from github import Github

def evaluar_diff_pr_con_ia(diff_texto):
    api_key = os.getenv("GEMINI_API_KEY")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    
    prompt = f"""
    Eres un Agente Revisor de Merge Requests para un proyecto CUDA.
    Analiza las diferencias (diff) de este PR y clasifícalo basándote estrictamente en esta regla:
    - Es 'mecánico': Solo cambia documentación, formato, o son cambios que no alteran la semántica física ni las firmas públicas[cite: 150].
    - Requiere revisión humana: Altera la lógica física, cambia llamadas de API o modifica el comportamiento del kernel.

    Devuelve ÚNICAMENTE un JSON estricto:
    {{
        "es_mecanico": true/false,
        "razonamiento": "Explica brevemente por qué tomaste la decisión."
    }}

    Diff del PR:
    {diff_texto}
    """
    
    response = model.generate_content(prompt)
    try:
        texto_limpio = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(texto_limpio)
    except Exception:
        # Por defecto, requerir revisión humana si la IA falla
        return {"es_mecanico": False, "razonamiento": "Fallo en el análisis de IA. Se requiere humano por seguridad."}

def main():
    token = os.getenv("GHCR_TOKEN")
    repo_name = os.getenv("GITHUB_REPOSITORY")
    pr_number = os.getenv("PR_NUMBER")
    
    g = Github(token)
    repo = g.get_repo(repo_name)
    pr = repo.get_pull(int(pr_number))
    ultimo_commit = pr.get_commits().reversed[0]
    estados = ultimo_commit.get_statuses()
    ci_exitoso = True
    for estado in estados:
        if estado.state != "success" and estado.context != "Agente Revisor de MR IA":
            ci_exitoso = False
            break

    # Leer las diferencias de codigo 
    archivos_cambiados = pr.get_files()
    diff_completo = ""
    for archivo in archivos_cambiados:
        diff_completo += f"--- {archivo.filename}\n+++ {archivo.filename}\n{archivo.patch}\n\n"

    # Analizar con IA
    analisis = evaluar_diff_pr_con_ia(diff_completo)
    
    # Construir el comentario de salida 
    comentario = "🤖 **Evaluación del Agente Revisor de MR**\n\n"
    
    if not ci_exitoso:
        comentario += "❌ **Estado del CI:** El pipeline de integración continua ha fallado. Por favor, revisa los logs.\n\n"
    else:
        comentario += "✅ **Estado del CI:** Los tests pasaron correctamente.\n\n"

    if analisis["es_mecanico"]:
        comentario += f"**Veredicto:** Mecánico y mergeable.\n"
    else:
        comentario += f"**Veredicto:** Requiere revisión humana.\n"
        
    comentario += f"**Análisis:** {analisis['razonamiento']}\n\n"
    comentario += "*Nota: Este agente nunca fusionará el código automáticamente a main.*"
    
    # Publicar comentario en el PR
    pr.create_issue_comment(comentario)
    print("Comentario publicado exitosamente en el PR.")

if __name__ == "__main__":
    main()