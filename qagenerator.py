import os
from typing import List, Dict
from langchain.schema import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

class QuestionGenerator:
    def __init__(self):
        self.llm = ChatGroq(
            model_name="llama3-70b-8192",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.7
        )

    def generate_questions(self, text: str, qtype: str = "short", num: int = 3) -> List[Dict]:
        """Generate questions using Groq API with robust error handling"""
        prompt = self._build_prompt(qtype, num)
        
        try:
            response = self.llm.invoke([
                SystemMessage(content=prompt),
                HumanMessage(content=text)
            ])
            return self._parse_response(response.content, qtype, num)
        except Exception as e:
            print(f"API Error: {e}")
            return self._fallback_questions(text, qtype, num)

    def _build_prompt(self, qtype: str, num: int) -> str:
        """Construct precise prompt for different question types"""
        prompts = {
            "short": f"""
            Generate {num} short-answer questions with answers in this exact format:
            Q: [question]
            A: [answer]
            """,
            "mcq": f"""
            Generate {num} multiple-choice questions with 3 options in this format:
            Q: [question]
            1) Option 1
            2) Option 2
            3) Option 3
            A: [correct option number]
            """,
            "fill": f"""
            Generate {num} fill-in-the-blank questions in this exact format:
            Q: [sentence with _____ where the blank is]
            A: [answer]
            Example:
            Q: The _____ is the powerhouse of the cell.
            A: mitochondria
            """,
            "tf": f"""
            Generate {num} true/false statements in this format:
            Q: [statement]
            A: True/False
            """
        }
        return prompts.get(qtype, prompts["short"])

    def _parse_response(self, response: str, qtype: str, expected_num: int) -> List[Dict]:
        """Parse API response with strict validation"""
        questions = []
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        
        i = 0
        while i < len(lines) and len(questions) < expected_num:
            if lines[i].startswith('Q:'):
                question = {
                    "type": qtype,
                    "question": lines[i][2:].strip(),
                    "answer": "",
                    "options": [] if qtype == "mcq" else None
                }
                
                # Find corresponding answer
                for j in range(i+1, len(lines)):
                    if lines[j].startswith('A:'):
                        question["answer"] = lines[j][2:].strip()
                        break
                
                # For fill-in-the-blank, ensure blank exists
                if qtype == "fill" and "_____" not in question["question"]:
                    question["question"] = question["question"].replace(
                        question["answer"], "_____"
                    )
                
                # For MCQ, collect options
                if qtype == "mcq":
                    question["options"] = [
                        line for line in lines[i+1:j] 
                        if line and line[0].isdigit() and ')' in line
                    ]
                
                questions.append(question)
                i = j + 1
            else:
                i += 1
        
        return questions[:expected_num]

    def _fallback_questions(self, text: str, qtype: str, num: int) -> List[Dict]:
        """Generate simple fallback questions if API fails"""
        return [{
            "type": qtype,
            "question": f"Sample {qtype} question about: {text[:50]}..." if qtype != "fill" 
                      else f"The _____ is related to: {text[:30]}...",
            "answer": "Sample answer",
            "options": ["Option 1", "Option 2", "Option 3"] if qtype == "mcq" else None
        } for _ in range(min(num, 3))]