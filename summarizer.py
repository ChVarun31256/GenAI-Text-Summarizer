from transformers import pipeline

summarizer = pipeline(
    "text2text-generation",
    model="t5-small"
)

text = """
Artificial Intelligence is transforming industries by automating tasks,
improving decision-making, and enabling new innovations.
Text summarization helps reduce long documents into short meaningful summaries.
"""

summary = summarizer("summarize: " + text, max_length=50, min_length=20)

print("Summary:\n", summary[0]['generated_text'])
