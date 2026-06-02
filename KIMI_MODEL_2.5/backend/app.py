import os
import base64
import traceback
from datetime import datetime

from flask import (
    Flask,
    request,
    jsonify,
    render_template,
    send_file
)

from flask_sqlalchemy import SQLAlchemy
from reportlab.pdfgen import canvas
from openai import OpenAI

# ==================================
# Flask Setup
# ==================================

app = Flask(__name__)

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///app.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# ==================================
# Database Model
# ==================================

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    result = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

with app.app_context():
    db.create_all()

# ==================================
# HuggingFace Client
# ==================================

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key="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"
)
# ==================================
# Home
# ==================================

@app.route("/")
def home():
    return render_template("index.html")

# ==================================
# Chat Route
# ==================================
@app.route("/chat", methods=["POST"])
def chat():

    try:

        data = request.get_json()
        user_message = data.get("message", "")

        if not user_message.strip():
            return jsonify({
                "response": "Please enter a message."
            })

        response = client.chat.completions.create(
            model="Qwen/Qwen3-32B",

            messages=[
                {
                    "role": "system",
                    "content": """
                    You are Abhishek AI Vision Studio.

                    Reply directly.
                    Never show reasoning.
                    Never output <think>.
                    Never reveal internal thoughts.
                    Only provide final answers.
                    """
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ],

            temperature=0.3,
            max_tokens=800
        )

        result = response.choices[0].message.content

        if "</think>" in result:
            result = result.split("</think>")[-1].strip()

        db.session.add(
            Prediction(result=result)
        )

        db.session.commit()

        return jsonify({
            "response": result
        })

    except Exception as e:

        traceback.print_exc()

        return jsonify({
            "response": str(e)
        }), 500

# ==================================
# Image Analysis Route
# ==================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        file = request.files.get("image")

        if not file:
            return jsonify({
                "result": "No image uploaded"
            })

        mode = request.form.get(
            "mode",
            "describe"
        )

        image_bytes = file.read()

        base64_image = base64.b64encode(
            image_bytes
        ).decode("utf-8")

        prompt_map = {
            "describe":
                "Describe this image in detail.",
            "caption":
                "Generate a caption for this image.",
            "objects":
                "List all objects visible in this image."
        }

        prompt = prompt_map.get(
            mode,
            "Describe this image in detail."
        )

        response = client.chat.completions.create(

            model="moonshotai/Kimi-K2.5",

            messages=[
                {
                    "role": "user",
                    "content": [

                        {
                            "type": "text",
                            "text": prompt
                        },

                        {
                            "type": "image_url",
                            "image_url": {
                                "url":
                                f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],

            max_tokens=1000
        )

        result = response.choices[0].message.content

        # Remove think tags

        if "<think>" in result and "</think>" in result:
            result = result.split("</think>")[-1].strip()

        db.session.add(
            Prediction(result=result)
        )

        db.session.commit()

        return jsonify({
            "result": result
        })

    except Exception as e:

        traceback.print_exc()

        return jsonify({
            "result": str(e)
        }), 500

# ==================================
# History
# ==================================

@app.route("/history")
def history():

    records = Prediction.query.order_by(
        Prediction.id.desc()
    ).limit(20)

    history_data = [

        {
            "result": r.result,
            "timestamp":
            r.timestamp.strftime("%H:%M")
        }

        for r in records
    ]

    return jsonify(history_data)

# ==================================
# PDF Report
# ==================================

@app.route("/report", methods=["POST"])
def report():

    text = request.form["text"]

    filename = "report.pdf"

    c = canvas.Canvas(filename)

    y = 800

    for line in text.split("\n"):

        c.drawString(
            40,
            y,
            line[:120]
        )

        y -= 20

        if y < 50:
            c.showPage()
            y = 800

    c.save()

    return send_file(
        filename,
        as_attachment=True
    )

# ==================================
# Run
# ==================================

if __name__ == "__main__":
    app.run(
        debug=True,
        use_reloader=False
    )