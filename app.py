# backend/app.py
print("✅ Flask app is starting...")

from flask import Flask, request, jsonify
from flask_cors import CORS
from summarizer import preprocess_text, tfidf_summary, llm_summary
from jobrecc import recommend_for_api
from qagenerator import QuestionGenerator
from tutorbot import tutor_chatbot

qg = QuestionGenerator()
app = Flask(__name__)
CORS(app)  # Allow all origins (for dev only)
print("abc")
@app.route('/summarizer', methods=['POST'])
def summarizer():
    print("123")
    data = request.json
    text = preprocess_text(data.get('text', ''))
    style = data.get('style', 'bullet')
    num = int(data.get('num', 5))

    # Extractive summary
    _, extracted_sentences = tfidf_summary(text, num_sentences=num)

    # LLM summary
    llm_result = llm_summary(text, style, num)
    print("🔎 LLM Result:", llm_result) 
    print("🔎 Extracted Sentences:", extracted_sentences)
    return jsonify({
        "extractive": extracted_sentences,
        "summary": llm_result
    })

@app.route('/qa-generator', methods=['POST'])
def generate_questions():
    data = request.get_json()
    text = data.get("text", "")
    qtype = data.get("type", "short")
    num = int(data.get("num", 3))

    if not text:
        return jsonify({"error": "Text is required"}), 400

    try:
        questions = qg.generate_questions(text, qtype, num)
        return jsonify({"questions": questions})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/job-recommend', methods=['POST'])
def job_recommendation():
    data = request.get_json()
    skills = data.get('skills', [])
    
    print("🔧 Received skills from frontend:", skills)

    if not skills:
        return jsonify({'error': 'No skills provided'}), 400

    recommendations = recommend_for_api(skills)
    for rec in recommendations:
        rec["match_score"] = float(rec["match_score"])

    print("🎯 Final job recommendations:", recommendations)
    
    return jsonify(recommendations)  # ✅ Add this line

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")
    bot_response = tutor_chatbot(user_message)
    return jsonify({"response": bot_response})





if __name__ == '__main__':
    app.run(debug=True)
