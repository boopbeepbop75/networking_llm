from flask import Flask, request, jsonify, render_template_string
from transformers import pipeline

app = Flask(__name__)
llm = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Simple HTML UI
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>LLM Chat</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; margin-top: 50px; }
        textarea { width: 80%; height: 100px; font-size: 16px; }
        button { padding: 10px 20px; font-size: 16px; cursor: pointer; }
        #response { margin-top: 20px; font-size: 18px; font-weight: bold; }
    </style>
</head>
<body>
    <h1>Interact with LLM</h1>
        <textarea id="context" placeholder="Enter context..."></textarea><br><br>
        <textarea id="question" placeholder="Enter your question..."></textarea><br><br>
        <button onclick="generateResponse()">Ask</button>
    <div id="response"></div>

    <script>
        async function generateResponse() {
            let context = document.getElementById("context").value;
            let question = document.getElementById("question").value;
            let responseDiv = document.getElementById("response");
            
            responseDiv.innerHTML = "Thinking...";

            let response = await fetch("/generate", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ question: question, context: context })
            });

            let data = await response.json();
            responseDiv.innerHTML = data.answer;
        }
    </script>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    question = data.get("question", "")
    context = data.get("context", "")

    if not question or not context:
        return jsonify({"error": "Both 'question' and 'context' are required"}), 400

    output = llm(question=question, context=context)
    return jsonify(output)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)