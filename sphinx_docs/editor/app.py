import os
import subprocess
from flask import Flask, render_template, request, redirect, url_for, flash
from pathlib import Path

app = Flask(__name__)
app.secret_key = "lerobot_doc_editor_secret"

# The directory containing the markdown files
DOCS_DIR = Path(__file__).parent.parent.resolve()

def get_md_files():
    """List all markdown files in the documentation directory."""
    return sorted([f.name for f in DOCS_DIR.glob("*.md")])

@app.route("/")
def index():
    """Homepage listing all editable markdown files."""
    files = get_md_files()
    return render_template("index.html", files=files)

@app.route("/edit/<path:filename>", methods=["GET", "POST"])
def edit(filename):
    """Editor page for a specific markdown file."""
    # Security check: ensure the file is within DOCS_DIR and is a markdown file
    file_path = (DOCS_DIR / filename).resolve()
    if not str(file_path).startswith(str(DOCS_DIR)) or file_path.suffix != ".md":
        flash("Error: Invalid file path or type.")
        return redirect(url_for("index"))

    if request.method == "POST":
        content = request.form.get("content")
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            
            # Trigger Sphinx build after save
            # Using the same command as in build_sphinx_docs.sh
            build_cmd = ["sphinx-build", "-b", "html", ".", "_build/html"]
            subprocess.run(build_cmd, cwd=DOCS_DIR, check=True)
            
            flash(f"Successfully saved and rebuilt {filename}")
        except Exception as e:
            flash(f"Error saving {filename}: {str(e)}")
            
        return redirect(url_for("edit", filename=filename))

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        flash(f"Error reading file: {str(e)}")
        return redirect(url_for("index"))
    
    return render_template("edit.html", filename=filename, content=content)

if __name__ == "__main__":
    # Run on all interfaces so colleagues can access it
    app.run(host="0.0.0.0", port=5000, debug=False)
