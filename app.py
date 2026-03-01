from flask import Flask, render_template, request, session, redirect, url_for
from transformers import pipeline
import re

app = Flask(__name__)
app.secret_key = "secret123"

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def clean_text(text):
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^[^\w]+", "", text)
    return text.strip()

def generate_summaries(text):
    text = clean_text(text)

    if len(text.split()) < 30:
        return "", "", ""

    base_summary = summarizer(
        text,
        max_length=180,
        min_length=60,
        do_sample=False
    )[0]["summary_text"]

    expert = (
        "This analysis provides a comprehensive overview of the subject, "
        + base_summary
    )

    simplified = summarizer(
        "Explain in simple language: " + text,
        max_length=120,
        min_length=40,
        do_sample=False
    )[0]["summary_text"]

    insights_raw = summarizer(
        "Give key insights in bullet style: " + text,
        max_length=120,
        min_length=40,
        do_sample=False
    )[0]["summary_text"]

    insights_sentences = insights_raw.split(". ")
    insights = "\n".join([f"• {s.strip()}" for s in insights_sentences if s.strip()])

    return expert.strip(), simplified.strip(), insights.strip()


@app.route("/", methods=["GET", "POST"])
def home():
    if "history" not in session:
        session["history"] = []

    expert = simplified = insights = ""
    original_text = ""
    show_add = False

    if request.method == "POST":

        if "generate" in request.form:
            original_text = request.form.get("text")
            expert, simplified, insights = generate_summaries(original_text)
            session["history"].append(clean_text(original_text))
            session.modified = True

        elif "show_add" in request.form:
            original_text = request.form.get("existing_text")
            show_add = True

        elif "add_text" in request.form:
            original_text = request.form.get("existing_text")
            extra_text = request.form.get("extra_text")
            combined = original_text + " " + extra_text
            expert, simplified, insights = generate_summaries(combined)
            session["history"].append(clean_text(combined))
            session.modified = True
            original_text = combined

    return render_template(
        "index.html",
        expert=expert,
        simplified=simplified,
        insights=insights,
        original_text=original_text,
        show_add=show_add,
        history=session["history"]
    )

@app.route("/delete/<int:index>")
def delete(index):
    if index < len(session["history"]):
        session["history"].pop(index)
        session.modified = True
    return redirect(url_for("home"))

if __name__ == "__main__":
    app.run(debug=True)